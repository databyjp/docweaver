import difflib
import json
from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax


def create_diff(old_content: str, new_content: str, file_path: str) -> str:
    """Creates a unified diff between old and new content with enhanced context."""
    return "".join(
        difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            n=5,  # Show 5 lines of context around changes
        )
    )


def main():
    writer_outpath = Path("outputs/doc_writer_agent.log")

    with writer_outpath.open(mode="r") as f:
        proposed_changes: list[dict[str, str]] = json.load(f)

    all_diffs = ""
    console = Console()

    for change in proposed_changes:
        file_path_str = change["path"]
        file_path = Path(file_path_str)

        original_content = ""
        if file_path.exists():
            original_content = file_path.read_text()

        diff = create_diff(original_content, change["revised_doc"], file_path_str)

        if diff:
            all_diffs += diff
            console.print(f"Diff for {file_path_str}:")
            syntax = Syntax(
                diff, "diff", theme="monokai", line_numbers=False, word_wrap=True
            )
            console.print(syntax)
            console.print("-" * 80)

    diff_outpath = Path("outputs/diffs.log")
    diff_outpath.parent.mkdir(parents=True, exist_ok=True)
    with diff_outpath.open(mode="w") as f:
        f.write(all_diffs)


if __name__ == "__main__":
    main()
