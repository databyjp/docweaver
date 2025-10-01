from docweaver.pipeline import (
    search_documents,
    coordinate_changes,
    make_changes,
    create_pr,
)
from rich.console import Console
import asyncio
from helpers import setup_logging, load_task
from pathlib import Path
import json
import sys

# Current task configuration - change this to switch between tasks
# Task name is the name of the Python file in the tasks/ directory
CURRENT_TASK_NAME = "prod_readiness"


def get_current_task_description() -> str:
    """Returns formatted task description for agents."""
    task = load_task(CURRENT_TASK_NAME)
    return task.get_description()


def clean_outputs():
    """Delete all intermediate output files."""
    console = Console()
    console.print("🧹 Cleaning all intermediate output files...")
    if Path("outputs").exists():
        for f in Path("outputs").glob("*.log"):
            f.unlink()
            console.print(f"   Deleted {f}")
        if Path("outputs/catalog.json").exists():
            Path("outputs/catalog.json").unlink()
            console.print("   Deleted outputs/catalog.json")
    console.print("✓ Clean complete\n")


async def run_search_stage(task_description: str, console: Console):
    """Run document search stage with caching."""
    output_path = "outputs/doc_search_agent.log"

    if Path(output_path).exists():
        console.print(f"✓ Using existing search results from {output_path}")
        with open(output_path) as f:
            search_data = json.load(f)
            return {"documents": search_data, "output_path": output_path}

    console.print("🔍 Searching documents...")
    result = await search_documents(task_description)
    console.print(f"   Token usage: {result['token_usage']}")
    return result


async def run_coordinate_stage(task_description: str, console: Console):
    """Run change coordination stage with caching."""
    output_path = "outputs/doc_instructor_agent.log"

    if Path(output_path).exists():
        console.print(f"✓ Using existing instructions from {output_path}")
        with open(output_path) as f:
            instructions_data = json.load(f)
            return {
                "instructions": instructions_data,
                "documents_processed": len(instructions_data),
                "output_path": output_path,
            }

    console.print("📋 Coordinating changes...")
    result = await coordinate_changes(task_description)
    console.print(f"   Token usage: {result['token_usage']}")
    return result


async def run_changes_stage(task_description: str, console: Console):
    """Run document changes stage with caching."""
    output_path = "outputs/doc_writer_agent.log"

    if Path(output_path).exists():
        console.print(f"✓ Using existing changes from {output_path}")
        with open(output_path) as f:
            changes_data = json.load(f)
            return {
                "revised_documents": changes_data,
                "files_changed": len(changes_data),
                "output_path": output_path,
            }

    console.print("✍️  Making changes...")
    result = await make_changes(task_description)
    console.print(f"   Processing time: {result['total_processing_time']:.2f} seconds")
    return result




async def run_pr_stage(task_description: str, console: Console):
    """Run PR creation stage."""
    console.print("📝 Applying changes and creating PR...")
    return await create_pr(feature_description=task_description)


async def main():
    setup_logging(__file__)
    console = Console()

    # Handle --clean flag
    if "--clean" in sys.argv:
        clean_outputs()

    task_description = get_current_task_description()

    # Stage 1: Search documents
    result = await run_search_stage(task_description, console)
    print(f"\nDocument search complete. Found {len(result['documents'])} documents:")
    for doc in result["documents"]:
        print(f"- {doc['path']}: {doc['reason']}")
    print(f"Results saved to: {result['output_path']}\n")

    # Stage 2: Coordinate changes
    result = await run_coordinate_stage(task_description, console)
    print(f"Change coordination complete.")
    print(f"Generated {len(result['instructions'])} editing instructions")
    print(f"Processed {result['documents_processed']} documents")
    print(f"Instructions saved to: {result['output_path']}\n")

    # Stage 3: Make changes
    result = await run_changes_stage(task_description, console)
    print(f"Document changes complete.")
    print(f"Files changed: {result['files_changed']}")
    print(f"Revised documents saved to: {result['output_path']}\n")

    # Stage 4: Create PR
    result = await run_pr_stage(task_description, console)
    if result["success"]:
        console.print(f"✅ {result['message']}")
        console.print(f"Branch: {result['branch_name']}")
        if "pr_url" in result:
            console.print(f"PR URL: {result['pr_url']}")
    else:
        console.print(f"ℹ️ {result['message']}")


if __name__ == "__main__":
    asyncio.run(main())
