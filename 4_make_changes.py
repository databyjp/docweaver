from pathlib import Path
import json
from docweaver.agents import doc_writer_agent
import asyncio
from helpers import TECH_DESCRIPTION_RESHARDING



async def main():
    logpath = Path("logs/doc_instructor_agent.log")
    with logpath.open(mode="r") as f:
        doc_instructions: list[dict[str, str]] = json.load(f)

    for doc_instruction in doc_instructions:
        response = await doc_writer_agent.run(f"""
        Write an updated version of the Weaviate documentation page
        for this new feature: {TECH_DESCRIPTION_RESHARDING}

        Here are the instructions by the doc manager:
        {doc_instruction}
        """)

        logpath = Path("logs/doc_writer_agent.log")
        logpath.parent.mkdir(parents=True, exist_ok=True)
        responses = [o.model_dump() for o in response.output]
        with logpath.open(mode="a") as f:
            json.dump(responses, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
