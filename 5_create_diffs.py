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


def apply_edits_to_content(original_content: str, edits: list[dict]) -> str:
    """Apply a list of edits to content."""
    revised_content = original_content
    for edit in edits:
        replace_section = edit["replace_section"]
        replacement_txt = edit["replacement_txt"]
        if replace_section in revised_content:
            revised_content = revised_content.replace(replace_section, replacement_txt)
    return revised_content


def main():
    # Load both logs to handle main files and referenced files
    writer_log_path = Path("logs/doc_writer_agent.log")
    edits_log_path = Path("logs/doc_writer_agent_edits.log")

    with writer_log_path.open(mode="r") as f:
        proposed_changes: list[dict[str, str]] = json.load(f)

    with edits_log_path.open(mode="r") as f:
        edits_data: list[dict] = json.load(f)

    all_diffs = ""
    console = Console()

    # Handle main document changes
    for change in proposed_changes:
        file_path_str = change["path"]
        file_path = Path(file_path_str)
        if file_path.exists():
            original_content = file_path.read_text()
            diff = create_diff(original_content, change["revised_doc"], file_path_str)
            all_diffs += diff
        else:
            # Handle case where the agent proposes a new file
            diff = create_diff("", change["revised_doc"], file_path_str)
            all_diffs += diff

        if diff:
            console.print(f"Diff for {file_path_str}:")
            syntax = Syntax(diff, "diff", theme="monokai", line_numbers=False, word_wrap=True)
            console.print(syntax)
            console.print("-" * 80)

    # Handle referenced file edits
    for edit_entry in edits_data:
        for edit_response in edit_entry["edits"]:
            referenced_file_edits = edit_response.get("referenced_file_edits", {})
            for ref_file_path, ref_edits in referenced_file_edits.items():
                file_path = Path(ref_file_path)
                if file_path.exists():
                    original_content = file_path.read_text()
                    revised_content = apply_edits_to_content(original_content, ref_edits)
                    diff = create_diff(original_content, revised_content, ref_file_path)
                else:
                    # Handle case where referenced file doesn't exist (new file)
                    revised_content = apply_edits_to_content("", ref_edits)
                    diff = create_diff("", revised_content, ref_file_path)

                if diff:
                    all_diffs += diff
                    console.print(f"Diff for referenced file {ref_file_path}:")
                    syntax = Syntax(diff, "diff", theme="monokai", line_numbers=False, word_wrap=True)
                    console.print(syntax)
                    console.print("-" * 80)

    diff_log_path = Path("logs/diffs.log")
    diff_log_path.parent.mkdir(parents=True, exist_ok=True)
    with diff_log_path.open(mode="w") as f:
        f.write(all_diffs)


if __name__ == "__main__":
    main()
