import shutil
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

# List of tasks to run in sequence
TASKS_TO_RUN = [
    "training_schema_design",
    "training_backup",
    "training_monitoring",
    "training_deployment"
]


def get_task_description(task_name: str) -> str:
    """Returns formatted task description for agents."""
    task = load_task(task_name)
    return task.get_description()


def clean_task_outputs(task_name: str, console: Console):
    """Delete all intermediate output files for a specific task."""
    task_output_dir = Path("outputs") / f"task_{task_name}"
    console.print(f"🧹 Cleaning all intermediate output files for task: {task_name}...")
    if task_output_dir.exists():
        shutil.rmtree(task_output_dir)
        console.print(f"   Deleted directory: {task_output_dir}")
    console.print("✓ Clean complete\n")


async def run_search_stage(
    task_description: str, task_output_dir: Path, console: Console
):
    """Run document search stage with caching."""
    output_path = task_output_dir / "doc_search_agent.log"

    if output_path.exists():
        console.print(f"✓ Using existing search results from {output_path}")
        with open(output_path) as f:
            search_data = json.load(f)
            return {"documents": search_data, "output_path": output_path}

    console.print("🔍 Searching documents via MCP...")
    result = await search_documents(task_description, output_path=str(output_path))
    if result['token_usage'] is not None:
        console.print(f"   Token usage: {result['token_usage']}")
    return result


async def run_coordinate_stage(
    task_description: str, task_output_dir: Path, console: Console
):
    """Run change coordination stage with caching."""
    output_path = task_output_dir / "doc_instructor_agent.log"
    search_results_path = task_output_dir / "doc_search_agent.log"

    if output_path.exists():
        console.print(f"✓ Using existing instructions from {output_path}")
        with open(output_path) as f:
            instructions_data = json.load(f)
            return {
                "instructions": instructions_data,
                "documents_processed": len(instructions_data),
                "output_path": output_path,
            }

    console.print("📋 Coordinating changes...")
    result = await coordinate_changes(
        task_description,
        search_results_path=str(search_results_path),
        output_path=str(output_path),
    )
    console.print(f"   Token usage: {result['token_usage']}")
    return result


async def run_changes_stage(
    task_description: str, task_output_dir: Path, console: Console
):
    """Run document changes stage with caching."""
    output_path = task_output_dir / "doc_writer_agent.log"
    edits_path = task_output_dir / "doc_writer_agent_edits.log"
    instructions_path = task_output_dir / "doc_instructor_agent.log"

    if output_path.exists():
        console.print(f"✓ Using existing changes from {output_path}")
        with open(output_path) as f:
            changes_data = json.load(f)
            return {
                "revised_documents": changes_data,
                "files_changed": len(changes_data),
                "output_path": output_path,
            }

    console.print("✍️  Making changes...")
    result = await make_changes(
        task_description,
        instructions_path=str(instructions_path),
        output_path=str(output_path),
        edits_path=str(edits_path),
    )
    console.print(f"   Processing time: {result['total_processing_time']:.2f} seconds")
    return result


async def run_pr_stage(
    task_description: str, task_name: str, task_output_dir: Path, console: Console
):
    """Run PR creation stage."""
    changes_path = task_output_dir / "doc_writer_agent.log"
    console.print("📝 Applying changes and creating PR...")
    return await create_pr(
        feature_description=task_description,
        task_name=task_name,
        changes_path=str(changes_path),
    )


async def run_task(task_name: str, console: Console):
    """Runs the full pipeline for a single task."""
    console.rule(f"[bold green]Starting Task: {task_name}[/bold green]")
    task_output_dir = Path("outputs") / f"task_{task_name}"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    task_description = get_task_description(task_name)

    # Stage 1: Search documents
    result = await run_search_stage(task_description, task_output_dir, console)
    print(f"\nDocument search complete. Found {len(result['documents'])} documents:")
    for doc in result["documents"]:
        print(f"- {doc['path']}: {doc.get('reason', 'No reason provided')}")
    print(f"Results saved to: {result['output_path']}\n")

    # Stage 2: Coordinate changes
    result = await run_coordinate_stage(task_description, task_output_dir, console)
    print("Change coordination complete.")
    print(f"Generated {len(result['instructions'])} editing instructions")
    print(f"Processed {result['documents_processed']} documents")
    print(f"Instructions saved to: {result['output_path']}\n")

    # Stage 3: Make changes
    result = await run_changes_stage(task_description, task_output_dir, console)
    print("Document changes complete.")
    print(f"Files changed: {result['files_changed']}")
    print(f"Revised documents saved to: {result['output_path']}\n")

    # Stage 4: Create PR
    result = await run_pr_stage(task_description, task_name, task_output_dir, console)
    if result["success"]:
        console.print(f"✅ {result['message']}")
        console.print(f"Branch: {result['branch_name']}")
        if "pr_url" in result:
            console.print(f"PR URL: {result['pr_url']}")
    else:
        console.print(f"ℹ️ {result['message']}")
    console.rule(f"[bold green]Finished Task: {task_name}[/bold green]\n")


async def main():
    setup_logging(__file__)
    console = Console()

    # Handle --clean flag
    if "--clean" in sys.argv:
        for task_name in TASKS_TO_RUN:
            clean_task_outputs(task_name, console)

    for task_name in TASKS_TO_RUN:
        await run_task(task_name, console)


if __name__ == "__main__":
    asyncio.run(main())
