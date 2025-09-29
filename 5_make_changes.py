from docweaver.pipeline import (
    search_documents,
    coordinate_changes,
    make_changes,
    create_diffs,
    create_pr,
)
from rich.console import Console
import asyncio
from helpers import setup_logging, load_task

# Current task configuration - change this to switch between tasks
# Task name is the name of the Python file in the tasks/ directory
CURRENT_TASK_NAME = "resharding_feature"


def get_current_task_description() -> str:
    """Returns formatted task description for agents."""
    task = load_task(CURRENT_TASK_NAME)
    return task.get_description()


async def main():
    setup_logging(__file__)
    task_description = get_current_task_description()
    result = await search_documents(task_description)

    print(f"Document search complete. Found {len(result['documents'])} documents:")
    for doc in result["documents"]:
        print(f"- {doc['path']}: {doc['reason']}")

    print(f"Results saved to: {result['output_path']}")
    print(f"Token usage: {result['token_usage']}")

    result = await coordinate_changes(task_description)

    print(f"Change coordination complete.")
    print(f"Generated {len(result['instructions'])} editing instructions")
    print(f"Processed {result['documents_processed']} documents")
    print(f"Instructions saved to: {result['output_path']}")
    print(f"Token usage: {result['token_usage']}")

    result = await make_changes(task_description)

    print(f"Document changes complete.")
    print(f"Files changed: {result['files_changed']}")
    print(f"Processing time: {result['total_processing_time']:.2f} seconds")
    print(f"Revised documents saved to: {result['output_path']}")
    print(f"Raw edits saved to: {result['edits_path']}")

    result = create_diffs()

    if result["has_changes"]:
        print(f"Created {result['diffs_created']} diffs")
        print(f"Diffs saved to: {result['output_path']}")
    else:
        print("No changes found to create diffs for")

    console = Console()

    console.print("📝 Applying diffs and creating PR...")
    result = create_pr()

    if result["success"]:
        console.print(f"✅ {result['message']}")
        console.print(f"Branch: {result['branch_name']}")
        if "pr_url" in result:
            console.print(f"PR URL: {result['pr_url']}")
    else:
        console.print(f"ℹ️ {result['message']}")


if __name__ == "__main__":
    asyncio.run(main())
