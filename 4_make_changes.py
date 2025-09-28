from docweaver.pipeline import make_changes
import asyncio
from helpers import get_current_task_description, setup_logging


async def main():
    setup_logging(__file__)
    task_description = get_current_task_description()
    result = await make_changes(task_description)

    print(f"Document changes complete.")
    print(f"Files changed: {result['files_changed']}")
    print(f"Processing time: {result['total_processing_time']:.2f} seconds")
    print(f"Revised documents saved to: {result['output_path']}")
    print(f"Raw edits saved to: {result['edits_path']}")


if __name__ == "__main__":
    asyncio.run(main())
