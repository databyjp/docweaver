from pathlib import Path
import json
from docweaver.agents import doc_writer_agent
import asyncio
from helpers import TECH_DESCRIPTION_RESHARDING
import logging
import time


async def process_instruction(doc_instruction: dict[str, str], semaphore: asyncio.Semaphore):
    """Runs the doc writer agent for a single instruction."""
    filepath = doc_instruction.get("path")
    if not filepath:
        logging.warning(f"Skipping instruction due to missing path: {doc_instruction}")
        return []
    try:
        original_content = Path(filepath).read_text()
    except FileNotFoundError:
        logging.error(f"File not found, skipping: {filepath}")
        return []

    async with semaphore:
        start_time = time.time()
        logging.info(f"Processing instruction for file: {filepath}")
        prompt = f"""
        Weaviate has introduced this new feature.
        ====== START-FEATURE DESCRIPTION =====
        {TECH_DESCRIPTION_RESHARDING}
        ====== END-FEATURE DESCRIPTION =====

        As a result, the documentation page at `{filepath}` needs to be updated.
        Here is the original content of the page:
        ====== START-ORIGINAL CONTENT =====
        {original_content}
        ====== END-ORIGINAL CONTENT =====

        Here is a high-level suggestion on how to update the page.
        ====== START-UPDATE INSTRUCTIONS =====
        {doc_instruction.get('instructions')}
        ====== END-UPDATE INSTRUCTIONS =====

        Please provide the revised content, from which we can build a file diff.
        """
        response = await doc_writer_agent.run(prompt)
        end_time = time.time()
        logging.info(
            f"Finished processing instruction for file: {filepath} in {end_time - start_time:.2f} seconds."
        )

        all_edits = [o.model_dump() for o in response.output]

        # Apply the edits

        revised_content = original_content
        for edit_response in all_edits:
            for edit in edit_response["edits"]:
                replace_section = edit["replace_section"]
                replacement_txt = edit["replacement_txt"]
                if replace_section in revised_content:
                    revised_content = revised_content.replace(replace_section, replacement_txt)
                else:
                    logging.warning(f"Could not find section to replace in {filepath}:\n{replace_section}")

        return [{
            "path": filepath,
            "revised_doc": revised_content,
            "edits": all_edits,
        }]


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logging.info("Starting to make changes to the documentation.")
    start_time = time.time()
    logpath = Path("logs/doc_instructor_agent.log")
    with logpath.open(mode="r") as f:
        doc_instructions: list[dict[str, str]] = json.load(f)

    logging.info(f"Found {len(doc_instructions)} instructions to process.")

    semaphore = asyncio.Semaphore(1)  # Conservative concurrency; seems to fail often when concurrent :/
    tasks = [
        process_instruction(doc_instruction, semaphore) for doc_instruction in doc_instructions
    ]
    results = await asyncio.gather(*tasks)

    all_responses_with_edits = []
    for instruction_responses in results:
        all_responses_with_edits.extend(instruction_responses)

    # Log the edits
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
