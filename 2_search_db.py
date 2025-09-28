from docweaver.pipeline import search_documents
import asyncio
from helpers import get_current_task_description, setup_logging


async def main():
    setup_logging(__file__)
    task_description = get_current_task_description()
    result = await search_documents(task_description)

    print(f"Document search complete. Found {len(result['documents'])} documents:")
    for doc in result["documents"]:
        print(f"- {doc['path']}: {doc['reason']}")

    print(f"Results saved to: {result['output_path']}")
    print(f"Token usage: {result['token_usage']}")


if __name__ == "__main__":
    asyncio.run(main())
