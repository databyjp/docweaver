from pathlib import Path
import json
from docweaver.agents import doc_writer_agent
import asyncio
from helpers import TECH_DESCRIPTION_RESHARDING
import logging
import time


async def process_instruction(doc_instruction: dict[str, str], semaphore: asyncio.Semaphore):
    """Runs the doc writer agent for a single instruction."""
    filepath = doc_instruction.get("path", "N/A")

    async with semaphore:
        start_time = time.time()
        logging.info(f"Processing instruction for file: {filepath}")
        response = await doc_writer_agent.run(
            f"""
        Write an updated version of the Weaviate documentation page
        for this new feature: {TECH_DESCRIPTION_RESHARDING}

        Here are the instructions by the doc manager:
        {doc_instruction}
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

    semaphore = asyncio.Semaphore(1)
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
