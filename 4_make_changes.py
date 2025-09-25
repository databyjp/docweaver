from pathlib import Path
import json
from docweaver.agents import doc_writer_agent
import asyncio
from helpers import get_current_task_description, setup_logging
import logging
import time
from collections import defaultdict


async def run_doc_writer_agent(prompt: str):
    """Runs the doc writer agent."""
    response = await doc_writer_agent.run(prompt)
    return response


async def process_instruction(instruction_bundle: dict, accumulated_contents: dict):
    """Runs the doc writer agent for a single coordinated instruction bundle."""
    primary_path = instruction_bundle["primary_path"]
    file_instructions = instruction_bundle["file_instructions"]

    all_paths = {primary_path}
    for instr in file_instructions:
        if not instr["path"].startswith("docs/"):
            instr["path"] = "docs/" + instr["path"]
        all_paths.add(instr["path"])

    original_contents = {}
    current_contents = {}
    prompt_docs_list = []

    for path_str in all_paths:
        path = Path(path_str)

        # Use accumulated content if available, otherwise read from file
        if path_str in accumulated_contents:
            content = accumulated_contents[path_str]
        else:
            content = path.read_text()
            # Store original content for first-time files
            if path_str not in accumulated_contents:
                original_contents[path_str] = content

        current_contents[path_str] = content

        doc_type = "MAIN FILE" if path_str == primary_path else "REFERENCED FILE"
        prompt_docs_list.append(f"## File: {path_str} ({doc_type})\n{content}\n")

    prompt_docs = "\n".join(prompt_docs_list)

    # Format the structured instructions for the prompt
    formatted_instructions = []
    for instr in file_instructions:
        formatted_instructions.append(
            f"File: {instr['path']}\nInstructions:\n{instr['instructions']}\n"
        )
    instructions_prompt = "\n---\n".join(formatted_instructions)

    start_time = time.time()
    logging.info(f"Processing instructions for primary file: {primary_path}")
    task_description = get_current_task_description()
    prompt = f"""
Update the documentation based on the following task.

# Task Description
{task_description}

# Documentation Files
{prompt_docs}

# Update Instructions
{instructions_prompt}
"""
    response = await run_doc_writer_agent(prompt)
    logging.info(
        f"Token usage for doc_writer_agent (primary_path: {primary_path}): {response.usage()}"
    )

    end_time = time.time()
    logging.info(
        f"Finished processing instructions for primary file: {primary_path} in {end_time - start_time:.2f} seconds."
    )

    all_edits = [o.model_dump() for o in response.output]
    revised_contents = current_contents.copy()

    # Group all edits by file path
    edits_by_file = defaultdict(list)
    for doc_output in all_edits:
        if doc_output.get("edits"):
            edits_by_file[doc_output["path"]].extend(doc_output["edits"])
        for path, ref_edits in doc_output.get("referenced_file_edits", {}).items():
            edits_by_file[path].extend(ref_edits)

    # Apply edits to all files
    for path, edits in edits_by_file.items():
        content = revised_contents.get(path)
        if content is None:
            continue
        for edit in edits:
            if edit["replace_section"] in content:
                content = content.replace(
                    edit["replace_section"], edit["replacement_txt"]
                )
        revised_contents[path] = content
        # Update accumulated contents for next instruction bundle
        accumulated_contents[path] = content

    # Return only files that changed in this instruction bundle
    results = []
    for path, content in revised_contents.items():
        if current_contents.get(path) != content:
            results.append({
                "path": path,
                "revised_doc": content,
                "edits": all_edits,  # Keep original agent output for logging
            })

    return results


async def main():
    setup_logging(__file__)
    logging.info("Starting to make changes to the documentation.")
    start_time = time.time()
    outpath = Path("outputs/doc_instructor_agent.log")
    with outpath.open(mode="r") as f:
        doc_instructions: list[dict] = json.load(f)

    logging.info(f"Found {len(doc_instructions)} instruction bundles to process.")

    # Track accumulated file contents across all instruction bundles
    accumulated_file_contents = {}
    all_responses_with_edits = []

    for instruction_bundle in doc_instructions:
        result = await process_instruction(instruction_bundle, accumulated_file_contents)
        if result:
            all_responses_with_edits.extend(result)

    # Log the edits - Note: this might log the same agent output multiple times if one instruction edits multiple files
    edits_to_log = [
        {"path": r.get("path"), "edits": r.get("edits")}
        for r in all_responses_with_edits
    ]
    edit_outpath = Path("outputs/doc_writer_agent_edits.log")
    edit_outpath.parent.mkdir(parents=True, exist_ok=True)
    with edit_outpath.open(mode="w") as f:
        json.dump(edits_to_log, f, indent=4)
    logging.info(f"Raw edits from agent logged to {edit_outpath}")

    # Log the revised documents
    revised_docs_to_log = [
        {"path": r.get("path"), "revised_doc": r.get("revised_doc")}
        for r in all_responses_with_edits
    ]
    outpath = Path("outputs/doc_writer_agent.log")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open(mode="w") as f:
        json.dump(revised_docs_to_log, f, indent=4)

    end_time = time.time()
    logging.info(
        f"Finished making changes to the documentation in {end_time - start_time:.2f} seconds."
    )


if __name__ == "__main__":
    asyncio.run(main())
