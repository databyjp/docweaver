from docweaver.pipeline import coordinate_changes
import asyncio
from helpers import get_current_task_description, setup_logging


async def main():
    setup_logging(__file__)
    task_description = get_current_task_description()
    result = await coordinate_changes(task_description)

    print(f"Change coordination complete.")
    print(f"Generated {len(result['instructions'])} editing instructions")
    print(f"Processed {result['documents_processed']} documents")
    print(f"Instructions saved to: {result['output_path']}")
    print(f"Token usage: {result['token_usage']}")


if __name__ == "__main__":
    asyncio.run(main())
