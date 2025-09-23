from pathlib import Path
import json
from docweaver.agents import DocSearchReturn, doc_instructor_agent
import asyncio
from helpers import TECH_DESCRIPTION_COLLECTION_ALIASES



async def main():
    logpath = Path("logs/doc_search_agent.log")
    with logpath.open(mode="r") as f:
        doc_search_results: list[dict[str, str]] = json.load(f)

    response = await doc_instructor_agent.run(
        f"""
        Propose doc changes for {TECH_DESCRIPTION_COLLECTION_ALIASES}.

        Here is the set of search results:
        {doc_search_results}
        """
    )

    logpath = Path("logs/doc_instructor_agent.log")
    logpath.parent.mkdir(parents=True, exist_ok=True)
    responses = [o.model_dump() for o in response.output]
    with logpath.open(mode="w") as f:
        json.dump(responses, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
