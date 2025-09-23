from pathlib import Path
import json
from docweaver.agents import doc_writer_agent, parse_doc_refs, WeaviateDoc
import asyncio
from helpers import TECH_DESCRIPTION_RESHARDING
import logging
import time


async def process_instruction(instruction_bundle: dict, semaphore: asyncio.Semaphore):
    """Runs the doc writer agent for a single coordinated instruction bundle."""
    primary_path = instruction_bundle.get("primary_path")
    file_instructions = instruction_bundle.get("file_instructions", [])

    if not primary_path or not file_instructions:
        logging.warning(f"Skipping bundle due to missing primary_path or file_instructions: {instruction_bundle}")
        return []

    # The document bundle includes the primary path and all referenced files
    doc_bundle = parse_doc_refs(Path(primary_path))
    if not doc_bundle.doc_body and not doc_bundle.referenced_docs:
        logging.error(f"File not found or empty, skipping: {primary_path}")
        return []

    # Collect and format docs from the bundle
    original_contents = {}
    prompt_docs_list = []

    def collect_docs_recursive(doc: WeaviateDoc, is_main: bool):
        if doc.path in original_contents:
            return # Avoid cycles
        original_contents[doc.path] = doc.doc_body

        doc_type = "MAIN FILE" if is_main else "REFERENCED FILE"
        prompt_docs_list.append(
            f"[{doc_type}]\n"
            f"Filepath: {doc.path}\n"
            f"====== START-ORIGINAL CONTENT =====\n{doc.doc_body}\n====== END-ORIGINAL CONTENT =====\n"
        )
        for ref_doc in doc.referenced_docs:
            collect_docs_recursive(ref_doc, is_main=False)

    collect_docs_recursive(doc_bundle, is_main=True)
    prompt_docs = "\n".join(prompt_docs_list)

    # Format the structured instructions for the prompt
    formatted_instructions = []
    for instr in file_instructions:
        formatted_instructions.append(
            f"File: {instr['path']}\n"
            f"Instructions:\n{instr['instructions']}\n"
        )
    instructions_prompt = "\n---\n".join(formatted_instructions)

    async with semaphore:
        start_time = time.time()
        logging.info(f"Processing instructions for primary file: {primary_path}")
        prompt = f"""
        Weaviate has introduced this new feature.
        ====== START-FEATURE DESCRIPTION =====
        {TECH_DESCRIPTION_RESHARDING}
        ====== END-FEATURE DESCRIPTION =====

        As a result, the documentation page at `{primary_path}` and its dependent files need to be updated.
        Here is the original content of the page and its dependencies:
        ====== START-DOCUMENT BUNDLE =====
        {prompt_docs}
        ====== END-DOCUMENT BUNDLE =====

        Here are the high-level suggestions on how to update the page and its related files.
        Apply these changes across all relevant files as specified.
        ====== START-UPDATE INSTRUCTIONS =====
        {instructions_prompt}
        ====== END-UPDATE INSTRUCTIONS =====

        Please provide the revised content as a series of edits, from which we can build a file diff.
        """
        response = await doc_writer_agent.run(prompt)
        end_time = time.time()
        logging.info(
            f"Finished processing instructions for primary file: {primary_path} in {end_time - start_time:.2f} seconds."
        )

        all_edits = [o.model_dump() for o in response.output]
        revised_contents = original_contents.copy()

        # Apply edits to all files (main and referenced)
        for doc_output in all_edits:
            # Edits for the main file
            for edit in doc_output.get("edits", []):
                path = doc_output["path"]
                if path in revised_contents and edit["replace_section"] in revised_contents[path]:
                    revised_contents[path] = revised_contents[path].replace(edit["replace_section"], edit["replacement_txt"])
                else:
                    logging.warning(f"Could not find section to replace in {path}:\n{edit['replace_section']}")

            # Edits for referenced files
            for path, ref_edits in doc_output.get("referenced_file_edits", {}).items():
                for edit in ref_edits:
                    if path in revised_contents and edit["replace_section"] in revised_contents[path]:
                        revised_contents[path] = revised_contents[path].replace(edit["replace_section"], edit["replacement_txt"])
                    else:
                        logging.warning(f"Could not find section to replace in {path}:\n{edit['replace_section']}")

        return [
            {
                "path": path,
                "revised_doc": content,
                "edits": all_edits, # Keep original agent output for logging
            }
            for path, content in revised_contents.items() if original_contents.get(path) != content
        ]


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info("Starting to make changes to the documentation.")
    start_time = time.time()
    # logpath = Path("logs/doc_instructor_agent_shorter.log")
    logpath = Path("logs/doc_instructor_agent.log")
    with logpath.open(mode="r") as f:
        doc_instructions: list[dict] = json.load(f)

    logging.info(f"Found {len(doc_instructions)} instruction bundles to process.")

    semaphore = asyncio.Semaphore(1)  # Conservative concurrency; seems to fail often when concurrent :/
    tasks = [
        process_instruction(instruction_bundle, semaphore) for instruction_bundle in doc_instructions
    ]
    results = await asyncio.gather(*tasks)

    all_responses_with_edits = [item for sublist in results for item in sublist]

    # Log the edits - Note: this might log the same agent output multiple times if one instruction edits multiple files
    edits_to_log = [
        {"path": r.get("path"), "edits": r.get("edits")} for r in all_responses_with_edits
    ]
    edit_logpath = Path("logs/doc_writer_agent_edits.log")
    edit_logpath.parent.mkdir(parents=True, exist_ok=True)
    with edit_logpath.open(mode="w") as f:
        json.dump(edits_to_log, f, indent=4)
    logging.info(f"Raw edits from agent logged to {edit_logpath}")

    # Log the revised documents
    revised_docs_to_log = [
        {"path": r.get("path"), "revised_doc": r.get("revised_doc")}
        for r in all_responses_with_edits
    ]
    logpath = Path("logs/doc_writer_agent.log")
    logpath.parent.mkdir(parents=True, exist_ok=True)
    with logpath.open(mode="w") as f:
        json.dump(revised_docs_to_log, f, indent=4)

    end_time = time.time()
    logging.info(
        f"Finished making changes to the documentation in {end_time - start_time:.2f} seconds."
    )


if __name__ == "__main__":
    asyncio.run(main())
