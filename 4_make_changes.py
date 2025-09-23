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
        response = await doc_writer_agent.run(
            f"""
        The documentation page at `{filepath}` needs to be updated.
        Here is the original content of the page:
        ---
        {original_content}
        ---

        The update is for this new feature:
        ---
        {TECH_DESCRIPTION_RESHARDING}
        ---

        Here are the instructions on how to update the page:
        ---
        {doc_instruction.get('instructions')}
        ---

        Please provide the full, revised content of the documentation page.
        """
        )
        end_time = time.time()
        logging.info(
            f"Finished processing instruction for file: {filepath} in {end_time - start_time:.2f} seconds."
        )
        return [o.model_dump() for o in response.output]


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

    semaphore = asyncio.Semaphore(2)  # Conservative concurrency
    tasks = [
        process_instruction(doc_instruction, semaphore) for doc_instruction in doc_instructions
    ]
    results = await asyncio.gather(*tasks)

    all_responses = []
    for instruction_responses in results:
        all_responses.extend(instruction_responses)

    logpath = Path("logs/doc_writer_agent.log")
    logpath.parent.mkdir(parents=True, exist_ok=True)
    with logpath.open(mode="w") as f:
        json.dump(all_responses, f, indent=4)

    end_time = time.time()
    logging.info(
        f"Finished making changes to the documentation in {end_time - start_time:.2f} seconds."
    )


if __name__ == "__main__":
    asyncio.run(main())
