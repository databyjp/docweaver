from pathlib import Path
import json
from docweaver.agents import doc_writer_agent, parse_doc_refs, WeaviateDoc
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

    doc_bundle = parse_doc_refs(Path(filepath))
    if not doc_bundle.doc_body and not doc_bundle.referenced_docs:
        logging.error(f"File not found or empty, skipping: {filepath}")
        return []

    # Collect and format docs
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

    async with semaphore:
        start_time = time.time()
        logging.info(f"Processing instruction for file: {filepath}")
        prompt = f"""
        Weaviate has introduced this new feature.
        ====== START-FEATURE DESCRIPTION =====
        {TECH_DESCRIPTION_RESHARDING}
        ====== END-FEATURE DESCRIPTION =====

        As a result, the documentation page at `{filepath}`,
        and its dependent files need to be updated.
        Here is the original content of the page and its dependencies:
        ====== START-DOCUMENT BUNDLE =====
        {prompt_docs}
        ====== END-DOCUMENT BUNDLE =====

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
    logpath = Path("logs/doc_instructor_agent.log")
    with logpath.open(mode="r") as f:
        doc_instructions: list[dict[str, str]] = json.load(f)

    logging.info(f"Found {len(doc_instructions)} instructions to process.")

    semaphore = asyncio.Semaphore(1)  # Conservative concurrency; seems to fail often when concurrent :/
    tasks = [
        process_instruction(doc_instruction, semaphore) for doc_instruction in doc_instructions
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
