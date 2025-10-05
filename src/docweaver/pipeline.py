"""
Document processing pipeline operations.

This module contains high-level operations for the docweaver document processing pipeline.
Each function corresponds to a stage in the sequential pipeline process.
"""

from pathlib import Path
from typing import Dict, Any
import logging
import json
import re
from pydantic import TypeAdapter

from . import db
from .config import DOCS_PATHS, DOCS_BASE_PATH
from .utils import chunk_text
from .agents import (
    docs_search_agent,
    DocSearchDeps,
    doc_instructor_agent,
    parse_doc_refs,
    WeaviateDoc,
    doc_writer_agent,
    DocOutput,
    pr_generator_agent,
    PRContent,
)
from .catalog import DocCatalog, get_docs_to_update, generate_metadata
import time
from collections import defaultdict
from datetime import datetime
from rich.console import Console
import git
import os
from github import Github


def prep_database(
    docs_paths: list[str] | None = None, reset_collection: bool = True
) -> Dict[str, Any]:
    """
    Reset and populate the document database.

    This operation deletes the existing collection, creates a new one,
    and populates it with chunks from markdown files in the specified paths.

    Args:
        docs_paths: List of documentation directories (default: from config.DOCS_PATHS)
        reset_collection: Whether to reset the collection (default: True)

    Returns:
        Dict containing processing results with keys:
        - files_processed: Number of files successfully processed
        - files: List of file paths that were processed
    """
    logging.info("Starting database preparation...")

    if docs_paths is None:
        docs_paths = DOCS_PATHS

    if reset_collection:
        # For pipeline use, we want to reset without user prompts
        try:
            with db.connect() as client:
                from .config import COLLECTION_NAME

                client.collections.delete(COLLECTION_NAME)
                logging.info("Deleted existing collection")
        except Exception as e:
            logging.info(f"Collection deletion skipped: {e}")

    try:
        db.create_collection()
        logging.info("Created collection")
    except Exception as e:
        logging.info(f"Collection creation skipped (may already exist): {e}")

    md_files = []
    for docs_path in docs_paths:
        md_files.extend(Path(docs_path).rglob("*.md*"))
    md_files = [f for f in md_files if f.name[0] != "_"]

    base_path = Path(DOCS_BASE_PATH)
    processed_files = []
    for file in md_files:
        with open(file, "r") as f:
            relative_path = file.relative_to(base_path).as_posix()
            logging.info(f"Importing {relative_path}")
            text = f.read()
            chunks = chunk_text(text)
            chunk_texts = [
                {"path": relative_path, "chunk": chunk.text} for chunk in chunks
            ]
            db.add_chunks(chunk_texts)
            processed_files.append(relative_path)

    logging.info(
        f"Database preparation complete. Processed {len(processed_files)} files."
    )
    return {"files_processed": len(processed_files), "files": processed_files}


async def search_documents(
    feature_description: str,
    output_path: str = "outputs/doc_search_agent.log",
    catalog_path: str = "outputs/catalog.json",
) -> Dict[str, Any]:
    """
    Search for documents that may need editing for a given feature.

    This operation uses the docs search agent to find relevant documents
    that might require updates based on the feature description. It uses
    both chunk-based content search and catalog metadata search.

    Args:
        feature_description: Description of the feature to search for
        output_path: Path to save the search results (default: "outputs/doc_search_agent.log")
        catalog_path: Path to catalog JSON (default: "outputs/catalog.json")

    Returns:
        Dict containing search results with keys:
        - documents: List of document search results
        - token_usage: Token usage information from the agent
        - output_path: Path where results were saved
    """
    logging.info(f"Starting document search for feature: {feature_description}")

    # Load catalog if available
    catalog = None
    if Path(catalog_path).exists():
        try:
            catalog = DocCatalog.load(Path(catalog_path))
            logging.info(f"Loaded catalog with {len(catalog.docs)} documents")
        except Exception as e:
            logging.warning(f"Could not load catalog: {e}")

    response = await docs_search_agent.run(
        f"Find documents that may need editing, for this feature: {feature_description}",
        deps=DocSearchDeps(client=db.connect(), catalog=catalog),
    )

    logging.info(f"Token usage for docs_search_agent: {response.usage()}")

    # Convert results to serializable format
    results = [doc.model_dump() for doc in response.output]

    # Save output for pipeline continuity
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    logging.info(
        f"Document search complete. Found {len(results)} documents. Results saved to {output_path}"
    )

    return {
        "documents": results,
        "token_usage": response.usage(),
        "output_path": output_path,
    }


async def coordinate_changes(
    feature_description: str,
    search_results_path: str = "outputs/doc_search_agent.log",
    output_path: str = "outputs/doc_instructor_agent.log",
) -> Dict[str, Any]:
    """
    Generate editing instructions for documents based on search results.

    This operation processes the search results, parses document references,
    and uses the doc instructor agent to generate detailed editing instructions.

    Args:
        feature_description: Description of the feature for context
        search_results_path: Path to the search results file (default: "outputs/doc_search_agent.log")
        output_path: Path to save the instructions (default: "outputs/doc_instructor_agent.log")

    Returns:
        Dict containing coordination results with keys:
        - instructions: List of editing instructions
        - documents_processed: Number of documents processed
        - token_usage: Token usage information from the agent
        - output_path: Path where instructions were saved
    """
    logging.info(f"Starting change coordination for feature: {feature_description}")

    # Load search results
    with open(search_results_path, "r") as f:
        doc_search_results = json.load(f)

    logging.info(f"Found {len(doc_search_results)} search results to process.")

    # Collect and format all docs and their references
    prompt_docs_list = []
    all_docs_content = {}  # Use a dict to avoid processing the same file twice

    def add_doc_to_prompt(doc: WeaviateDoc, is_main: bool):
        if doc.path in all_docs_content or not doc.doc_body:
            return  # Skip if already processed or empty

        all_docs_content[doc.path] = doc.doc_body
        doc_type = "MAIN FILE" if is_main else "REFERENCED FILE"
        prompt_docs_list.append(
            f"[{doc_type}]\n"
            f"Filepath: {doc.path}\n"
            f"====== START-ORIGINAL CONTENT =====\n{doc.doc_body}\n====== END-ORIGINAL CONTENT =====\n"
        )

    for result in doc_search_results:
        filepath = result.get("path")
        if not filepath:
            logging.warning(f"Skipping search result due to missing path: {result}")
            continue

        logging.info(f"Parsing document and its references: {filepath}")
        # Resolve path relative to DOCS_BASE_PATH
        full_path = Path(DOCS_BASE_PATH) / filepath
        doc_bundle = parse_doc_refs(full_path, include_code_body=False)

        # Add main document
        add_doc_to_prompt(doc_bundle, is_main=True)

        # Add only first-level references
        for ref_doc in doc_bundle.referenced_docs:
            add_doc_to_prompt(ref_doc, is_main=False)

    document_bundle_prompt = "\n".join(prompt_docs_list)

    prompt = f"""
    Weaviate has introduced this new feature.
    ====== START-FEATURE DESCRIPTION =====
    {feature_description}
    ====== END-FEATURE DESCRIPTION =====

    A preliminary search has identified the following files as potentially needing updates:
    {json.dumps(doc_search_results, indent=2)}

    Here is the full content of those files and any other files they reference (like code examples):
    ====== START-DOCUMENT BUNDLE =====
    {document_bundle_prompt}
    ====== END-DOCUMENT BUNDLE =====

    Based on all of this context, instruct the writers on how to update the documentation.
    Focus on what needs to change and where, including any code examples in referenced files.
    """

    logging.info("Running doc_instructor_agent to generate edit instructions...")
    response = await doc_instructor_agent.run(prompt)
    logging.info(f"Token usage for doc_instructor_agent: {response.usage()}")

    # Save instructions
    instructions = [inst.model_dump() for inst in response.output]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(instructions, f, indent=4)

    logging.info(
        f"Change coordination complete. Generated {len(instructions)} instructions. Saved to {output_path}"
    )

    return {
        "instructions": instructions,
        "documents_processed": len(all_docs_content),
        "token_usage": response.usage(),
        "output_path": output_path,
    }


def _validate_and_log_edit(edit: dict, file_path: str) -> None:
    """
    Validate edit and log warnings for suspicious changes.

    Flags edits that appear to be overly aggressive (large deletions/replacements)
    without adequate justification.
    """
    start_line = edit.get("start_line", 0)
    end_line = edit.get("end_line", 0)
    replacement_txt = edit.get("replacement_txt", "")
    comment = edit.get("comment", "No comment provided")
    edit_type = edit.get("edit_type", "unknown")
    justification = edit.get("justification", "No justification provided")

    lines_affected = end_line - start_line + 1

    # Flag large deletions
    if lines_affected > 1 and not replacement_txt.strip():
        logging.warning(
            f"⚠️  LARGE DELETION in {file_path} (lines {start_line}-{end_line}, {lines_affected} lines)\n"
            f"   Edit type: {edit_type}\n"
            f"   Comment: {comment}\n"
            f"   Justification: {justification}"
        )

    # Flag large replacements (>10 lines) that claim to be updates
    elif lines_affected > 10 and edit_type == "update_outdated":
        logging.warning(
            f"⚠️  LARGE REPLACEMENT in {file_path} (lines {start_line}-{end_line}, {lines_affected} lines)\n"
            f"   Edit type: {edit_type}\n"
            f"   Comment: {comment}\n"
            f"   Justification: {justification}\n"
            f"   → Consider: Could this be ADD_NEW instead of UPDATE_OUTDATED?"
        )

    # Flag deletions with weak justification
    if edit_type == "delete_redundant" and len(justification) < 50:
        logging.warning(
            f"⚠️  DELETION WITH WEAK JUSTIFICATION in {file_path} (lines {start_line}-{end_line})\n"
            f"   Justification is too brief: {justification}\n"
            f"   → Deletions should have detailed justification"
        )

    # Log all non-ADD_NEW edits for review
    if edit_type in ["update_outdated", "delete_redundant"]:
        logging.info(
            f"📝 Non-additive edit in {file_path} (lines {start_line}-{end_line})\n"
            f"   Type: {edit_type} | Comment: {comment}\n"
            f"   Justification: {justification}"
        )


async def make_changes(
    feature_description: str,
    instructions_path: str = "outputs/doc_instructor_agent.log",
    output_path: str = "outputs/doc_writer_agent.log",
    edits_path: str = "outputs/doc_writer_agent_edits.log",
) -> Dict[str, Any]:
    """
    Apply editing instructions to generate revised documents.

    This operation processes the coordination instructions and uses the doc writer agent
    to generate actual document changes. Edits are applied cumulatively if multiple
    instructions touch the same file.

    Args:
        feature_description: Description of the feature for context
        instructions_path: Path to the instructions file (default: "outputs/doc_instructor_agent.log")
        output_path: Path to save revised documents (default: "outputs/doc_writer_agent.log")
        edits_path: Path to save raw edits (default: "outputs/doc_writer_agent_edits.log")

    Returns:
        Dict containing results with keys:
        - revised_documents: List of revised document data
        - files_changed: Number of files that were changed
        - total_processing_time: Total time taken in seconds
    """
    logging.info("Starting to make changes to the documentation.")
    start_time = time.time()

    with open(instructions_path, "r") as f:
        doc_instructions = json.load(f)

    logging.info(f"Found {len(doc_instructions)} instruction bundles to process.")

    # Collect all file paths from all bundles and load their original content
    all_paths = set()
    for instruction_bundle in doc_instructions:
        all_paths.add(instruction_bundle["primary_path"])
        for instr in instruction_bundle["file_instructions"]:
            all_paths.add(instr["path"])

    original_contents = {}
    for path_str in all_paths:
        path = Path(path_str)
        if not path.is_file():
            path = Path("docs") / path_str
        if not path.is_file():
            logging.warning(
                f"Could not find file: {path_str} or docs/{path_str}, skipping."
            )
            continue
        original_contents[path.as_posix()] = path.read_text()

    revised_contents = original_contents.copy()
    all_raw_edits = []

    for instruction_bundle in doc_instructions:
        primary_path = instruction_bundle["primary_path"]
        file_instructions = instruction_bundle["file_instructions"]

        bundle_paths = {primary_path}
        for instr in file_instructions:
            bundle_paths.add(instr["path"])

        prompt_docs_list = []
        bundle_contents = {}

        for path_str in bundle_paths:
            # Resolve path, trying both original and docs-prefixed versions
            canonical_path = None
            if path_str in revised_contents:
                canonical_path = path_str
            else:
                prefixed_path = (Path("docs") / path_str).as_posix()
                if prefixed_path in revised_contents:
                    canonical_path = prefixed_path

            if not canonical_path:
                logging.warning(
                    f"Content for {path_str} not found in revised_contents, skipping from bundle."
                )
                continue

            content = revised_contents[canonical_path]
            bundle_contents[canonical_path] = content

            lines = content.splitlines()
            numbered_content = "\n".join(
                f"{i+1}|{line}" for i, line in enumerate(lines)
            )
            doc_type = "MAIN FILE" if path_str == primary_path else "REFERENCED FILE"
            prompt_docs_list.append(
                f"## File: {canonical_path} ({doc_type})\n{numbered_content}\n"
            )

        prompt_docs = "\n".join(prompt_docs_list)

        # Format structured instructions
        formatted_instructions = []
        for instr in file_instructions:
            formatted_instructions.append(
                f"File: {instr['path']}\nInstructions:\n{instr['instructions']}\n"
            )
        instructions_prompt = "\n---\n".join(formatted_instructions)

        prompt = f"""
        Update the documentation for a new Weaviate feature.

        # Feature Description
        {feature_description}

        # Documentation Files
        {prompt_docs}

        # Update Instructions
        {instructions_prompt}
        """

        bundle_start_time = time.time()
        logging.info(f"Processing instructions for primary file: {primary_path}")

        try:
            response = await doc_writer_agent.run(prompt)
        except Exception as e:
            logging.error(f"Agent failed for primary_path {primary_path} after retries: {e}")
            continue # Skip to the next bundle

        logging.info(
            f"Token usage for doc_writer_agent (primary_path: {primary_path}): {response.usage()}"
        )

        # The agent now returns a list of DocOutput objects directly
        parsed_output = response.output

        # Save parsed output for debugging
        output_dir = Path(output_path).parent
        debug_output_path = (
            output_dir / f"doc_writer_agent_raw_output_{Path(primary_path).stem}.log"
        )
        debug_output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            output_for_log = [o.model_dump() for o in parsed_output]
            with open(debug_output_path, "w") as f:
                json.dump(output_for_log, f, indent=4)
            logging.info(f"Saved parsed agent output to {debug_output_path}")
        except Exception as e:
            logging.error(f"Could not serialize agent output for logging: {e}")


        try:
            all_edits = [o.model_dump() for o in parsed_output]

            # Validate and flag suspicious edits
            for doc_output in all_edits:
                file_path = doc_output["path"]
                for edit in doc_output.get("edits", []):
                    _validate_and_log_edit(edit, file_path)
                for ref_path, ref_edits in doc_output.get("referenced_file_edits", {}).items():
                    for edit in ref_edits:
                        _validate_and_log_edit(edit, ref_path)

            all_raw_edits.extend(all_edits)
        except Exception as e:
            logging.error(
                f"Failed to process doc_writer_agent output for {primary_path}: {e}"
            )
            logging.error(f"Parsed output was: \n{parsed_output}")
            continue  # Continue to the next instruction bundle

        # Group all edits by file path
        edits_by_file = defaultdict(list)
        for doc_output in all_edits:
            if doc_output.get("edits"):
                edits_by_file[doc_output["path"]].extend(doc_output["edits"])
            for path, ref_edits in doc_output.get(
                "referenced_file_edits", {}
            ).items():
                edits_by_file[path].extend(ref_edits)

        # Apply edits to all files using a line-based method
        for path, edits in edits_by_file.items():
            content_to_edit = revised_contents.get(path)
            if content_to_edit is None:
                logging.warning(
                    f"No original content found for {path}, skipping edits."
                )
                continue

            lines = content_to_edit.splitlines()

            # Sort edits by start_line in reverse order to avoid index shifting issues
            edits.sort(key=lambda e: e["start_line"], reverse=True)

            for edit in edits:
                start_line = edit["start_line"]
                end_line = edit["end_line"]
                replacement_txt = edit["replacement_txt"]
                comment = edit.get("comment", "No comment")

                logging.info(
                    f"Applying edit to {path} at lines {start_line}-{end_line}: {comment}"
                )

                # Adjust for 0-based indexing. end_line is inclusive.
                start_idx = start_line - 1

                # For insertions (ADD_NEW, ENHANCE), the intent is to insert text *before*
                # the specified start_line without deleting the line itself. To achieve this,
                # the slice should be empty (e.g., lines[i:i]).
                # For replacements and deletions, the slice should include the lines
                # from start_line to end_line.
                edit_type = edit.get("edit_type")
                is_insertion = edit_type in ["add_new", "enhance"]

                if is_insertion:
                    end_idx = start_idx  # Creates an empty slice for insertion
                else:
                    end_idx = end_line  # Creates a slice for replacement/deletion

                # Handle invalid line numbers
                if (
                    start_idx < 0
                    or start_idx > len(lines)
                    or end_idx < start_idx
                    or end_idx > len(lines)
                ):
                    logging.warning(
                        f"Invalid line numbers for edit in {path}: start={start_line}, end={end_line}. Skipping."
                    )
                    continue

                replacement_lines = replacement_txt.splitlines()
                lines[start_idx:end_idx] = replacement_lines

            # Clean trailing whitespace from all lines before saving
            cleaned_lines = [line.rstrip() for line in lines]
            revised_contents[path] = "\n".join(cleaned_lines)

        bundle_end_time = time.time()
        logging.info(
            f"Finished processing instructions for primary file: {primary_path} in {bundle_end_time - bundle_start_time:.2f} seconds."
        )

    # After processing all bundles, determine the final set of changed documents
    revised_docs_to_log = []
    for path, content in revised_contents.items():
        if original_contents.get(path) != content:
            revised_docs_to_log.append({"path": path, "revised_doc": content})

    # Save raw edits
    Path(edits_path).parent.mkdir(parents=True, exist_ok=True)
    with open(edits_path, "w") as f:
        json.dump(all_raw_edits, f, indent=4)
    logging.info(f"Raw edits from agent logged to {edits_path}")

    # Save revised documents
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(revised_docs_to_log, f, indent=4)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(
        f"Finished making changes to the documentation in {total_time:.2f} seconds."
    )

    return {
        "revised_documents": revised_docs_to_log,
        "files_changed": len(revised_docs_to_log),
        "total_processing_time": total_time,
        "output_path": output_path,
        "edits_path": edits_path,
    }




async def generate_pr_content(
    feature_description: str,
    changes_path: str = "outputs/doc_writer_agent.log",
    docs_path: str = "docs",
) -> PRContent:
    """
    Generate PR title and description using LLM based on the changes made.

    Args:
        feature_description: Description of the feature that was implemented
        changes_path: Path to the revised documents file
        docs_path: Path to the docs repository for generating diffs

    Returns:
        PRContent object with generated title and description
    """
    logging.info("Generating PR content using LLM...")

    # Load the changes made
    import difflib
    from pathlib import Path
    import json

    changes_data = []
    if Path(changes_path).exists():
        with open(changes_path, "r") as f:
            changes_data = json.load(f)

    if not changes_data:
        # Fallback for no changes
        timestamp_readable = datetime.now().strftime("%Y-%m-%d %H:%M")
        return PRContent(
            title=f"📚 DocWeaver updates - {timestamp_readable}",
            description="This PR contains automated documentation improvements generated by DocWeaver.",
        )

    # Generate diffs to provide concise context to the LLM
    diffs = []
    for change in changes_data:
        file_path_str = change["path"]
        # The path in the change log is relative to the docs/ dir, but when applying changes,
        # we need to make sure we're inside docs/
        file_path = Path(docs_path) / Path(file_path_str).relative_to(Path(docs_path))
        original_content = ""
        if file_path.exists():
            original_content = file_path.read_text()

        revised_content = change["revised_doc"]
        diff = "".join(
            difflib.unified_diff(
                original_content.splitlines(keepends=True),
                revised_content.splitlines(keepends=True),
                fromfile=f"a/{file_path_str}",
                tofile=f"b/{file_path_str}",
            )
        )
        diffs.append(diff)

    diff_summary = "\n".join(filter(None, diffs))

    prompt = f"""
    Analyze the following documentation changes and generate an appropriate PR title and description.
    The changes are provided in the form of a git diff.

    ## Feature Context
    {feature_description}

    ## Changes Made (in diff format)
    ```diff
    {diff_summary}
    ```

    Please generate a clear, concise PR title and a comprehensive description that will help reviewers understand what was changed and why.
    The description should summarize the changes shown in the diff, focusing on what was changed and why.
    """

    try:
        response = await pr_generator_agent.run(prompt)
        logging.info(f"Generated PR content. Token usage: {response.usage()}")
        return response.output
    except Exception as e:
        logging.error(f"Failed to generate PR content: {e}")
        # Fallback to default content
        timestamp_readable = datetime.now().strftime("%Y-%m-%d %H:%M")
        return PRContent(
            title=f"📚 DocWeaver updates - {timestamp_readable}",
            description="""This PR contains automated documentation improvements generated by DocWeaver.
## Changes Summary
- The LLM was unable to generate a summary. Please review the diff for details.""",
        )


async def create_pr(
    feature_description: str = None,
    task_name: str = None,
    title: str = None,
    body: str = None,
    branch_name: str = None,
    changes_path: str = "outputs/doc_writer_agent.log",
    docs_path: str = "docs",
) -> Dict[str, Any]:
    """
    Apply diffs and create a pull request.

    This operation applies the generated diffs to the actual files and creates
    a pull request with the changes. Diffs are generated and applied individually.

    Args:
        feature_description: Description of the feature for LLM-generated PR content (optional)
        task_name: The name of the task being executed (optional)
        title: PR title (LLM-generated if None and feature_description provided)
        body: PR description (LLM-generated if None and feature_description provided)
        branch_name: Branch name (auto-generated if None)
        changes_path: Path to the revised documents file (default: "outputs/doc_writer_agent.log")
        docs_path: Path to the docs repository (default: "docs")

    Returns:
        Dict containing PR results with keys:
        - pr_url: URL of the created pull request (if successful)
        - branch_name: Name of the created branch
        - success: Whether the operation succeeded
    """
    logging.info("Creating pull request with changes.")
    console = Console()

    # Load changes
    if not Path(changes_path).exists():
        raise FileNotFoundError(f"No changes log found at {changes_path}")

    with open(changes_path, "r") as f:
        proposed_changes = json.load(f)

    if not proposed_changes:
        console.print("✅ No changes found in log, skipping PR creation.")
        return {"success": False, "message": "No changes to apply"}

    # Work in the docs directory
    docs_path_obj = Path(docs_path)
    if not docs_path_obj.exists():
        raise FileNotFoundError("docs/ directory not found")

    repo = git.Repo(docs_path_obj)
    original_branch = repo.active_branch
    changes_applied = False
    try:
        # Overwrite files with their revised content
        for change in proposed_changes:
            file_path_str = change["path"]
            file_path = Path(file_path_str)

            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the revised document content
            with open(file_path, "w") as f:
                f.write(change["revised_doc"])

            console.print(f"✅ Changes for {file_path_str} written to disk.")
            changes_applied = True

        if not changes_applied:
            console.print("✅ No changes to apply, skipping.")
            return {"success": False, "message": "No changes to apply"}

        # Get GitHub token
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_TOKEN environment variable not set")

        # Generate branch name if not provided
        if branch_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            branch_name = f"docweaver-updates-{timestamp}"

        # Ensure we're on the default branch
        try:
            # Get default branch from remote 'origin'
            default_branch_name = repo.remotes.origin.refs.HEAD.ref.name.split("/")[-1]
            repo.heads[default_branch_name].checkout()
            logging.info(f"Checked out default branch: {default_branch_name}")
        except Exception as e:
            logging.warning(
                f"Could not determine default branch from remote. Falling back to main/master. Error: {e}"
            )
            try:
                repo.heads["main"].checkout()
            except:
                repo.heads["master"].checkout()

        # Create and checkout new branch
        if branch_name in repo.heads:
            console.print(f"Branch '{branch_name}' already exists. Recreating it.")
            repo.delete_head(branch_name, force=True)

        new_branch = repo.create_head(branch_name)
        new_branch.checkout()

        # Stage and commit changes
        repo.git.add(".")

        if not repo.is_dirty() and not repo.untracked_files:
            console.print("⚠️  No changes to commit")
            return {"success": False, "message": "No changes to commit"}

        # Generate title and body if not provided
        if title is None or body is None:
            if feature_description and (body is None):
                # Use LLM to generate PR content
                console.print("🤖 Generating PR title and description using LLM...")
                pr_content = await generate_pr_content(
                    feature_description, changes_path, docs_path=docs_path
                )

                llm_description = pr_content.description
                body = f"""{llm_description}

---

## Task Description

{feature_description}"""
            elif body is None:
                # Fallback to default content
                body = """This PR contains automated documentation improvements generated by DocWeaver.

## Changes Summary
- Content updates and clarifications
- Formatting improvements
- Enhanced code examples
- Better structural organization

🤖 Generated with DocWeaver"""

            if title is None:
                timestamp_readable = datetime.now().strftime("%Y-%m-%d %H:%M")
                if task_name:
                    title = f"DocWeaver draft: {task_name} - {timestamp_readable}"
                else:
                    title = f"📚 DocWeaver updates - {timestamp_readable}"

        repo.index.commit(title)
        repo.remotes.origin.push(new_branch)

        console.print(f"✅ Branch '{branch_name}' created and pushed")
        logging.info(f"Created branch '{branch_name}' with changes")

        # Create Pull Request
        remote_url = repo.remotes.origin.url
        match = re.search(r"github\.com[/:]([\w.-]+)/([\w.-]+?)(?:\.git)?$", remote_url)
        if not match:
            raise ValueError(
                f"Could not parse GitHub owner/repo from remote URL: {remote_url}"
            )
        repo_owner, repo_name = match.groups()

        g = Github(github_token)
        gh_repo = g.get_repo(f"{repo_owner}/{repo_name}")

        # Get default branch from remote 'origin'
        default_branch_name = repo.remotes.origin.refs.HEAD.ref.name.split("/")[-1]

        pull = gh_repo.create_pull(
            title=title,
            body=body,
            head=branch_name,
            base=default_branch_name,
            draft=True,
        )

        console.print(f"✅ Draft pull request created: {pull.html_url}")

        return {
            "success": True,
            "branch_name": branch_name,
            "pr_url": pull.html_url,
            "title": title,
            "body": body,
            "message": f"Draft pull request for branch '{branch_name}' created successfully.",
        }

    except Exception as e:
        raise RuntimeError(f"Failed to process changes: {e}")
    finally:
        # Switch back to the original branch and clean up worktree
        try:
            # Use git checkout and clean to reset the state
            repo.git.checkout(original_branch.name, "--force")
            repo.git.clean("-fd")
            console.print(
                f"✅ Switched back to original branch: {original_branch.name} and cleaned worktree."
            )
        except Exception as e:
            console.print(f"⚠️ Could not switch back to original branch: {e}")


async def update_catalog(
    docs_paths: list[str] | None = None,
    catalog_path: str = "outputs/catalog.json",
    limit: int | None = None,
) -> Dict[str, Any]:
    """
    Update the document catalog with metadata for new or modified documents.
    This operation generates metadata (topics, doctype, summary) for documents
    and stores them both locally (JSON) and in Weaviate for vector search.
    It also removes documents from the catalog that no longer exist.
    Args:
        docs_paths: List of documentation directories (default: from config.DOCS_PATHS)
        catalog_path: Path to save catalog JSON (default: "outputs/catalog.json")
        limit: Optional limit on number of documents to process
    Returns:
        Dict containing results with keys:
        - total_docs: Total documents in catalog
        - updated_docs: Number of documents updated
        - removed_docs: Number of documents removed
        - catalog_path: Path where catalog was saved
    """
    logging.info("Starting catalog update...")

    if docs_paths is None:
        docs_paths = DOCS_PATHS

    catalog = DocCatalog.load(Path(catalog_path))
    logging.info(f"Loaded catalog with {len(catalog.docs)} existing documents")

    base_path = Path(DOCS_BASE_PATH)

    # Get all documents on disk from all specified paths
    all_docs_on_disk = set()
    for docs_path in docs_paths:
        doc_root = Path(docs_path)
        all_docs_on_disk.update(
            p.relative_to(base_path).as_posix()
            for p in doc_root.rglob("*.md*")
            if not p.name.startswith("_")
        )

    # Find documents to remove (in catalog but not on disk)
    docs_in_catalog = set(catalog.docs.keys())
    docs_to_remove = list(docs_in_catalog - all_docs_on_disk)

    removed_count = 0
    if docs_to_remove:
        logging.info(f"Found {len(docs_to_remove)} documents to remove")
        for path in docs_to_remove:
            if path in catalog.docs:
                del catalog.docs[path]
        try:
            db.remove_catalog_entries(docs_to_remove)
            removed_count = len(docs_to_remove)
            logging.info(f"Removed {removed_count} entries from Weaviate catalog")
        except Exception as e:
            logging.warning(f"Could not remove from Weaviate catalog: {e}")

    # Find documents that need updating across all paths
    # Use common base path for consistent relative paths
    all_docs_to_update = []
    for docs_path in docs_paths:
        doc_root = Path(docs_path)
        docs_to_update = get_docs_to_update(doc_root, catalog, base_path)
        all_docs_to_update.extend(docs_to_update)

    if limit:
        all_docs_to_update = all_docs_to_update[:limit]
        logging.info(f"Limited to {limit} documents")

    if not all_docs_to_update:
        logging.info("No documents to update")
        return {
            "total_docs": len(catalog.docs),
            "updated_docs": 0,
            "removed_docs": removed_count,
            "catalog_path": catalog_path,
        }

    logging.info(f"Found {len(all_docs_to_update)} documents to update")

    # Generate metadata for each document with incremental saving
    updated_entries = []
    for i, doc_path in enumerate(all_docs_to_update):
        try:
            metadata = await generate_metadata(doc_path, base_path)
            catalog.docs[metadata.path] = metadata
            updated_entries.append(metadata.model_dump())
            logging.info(f"Updated metadata for {metadata.path} ({i+1}/{len(all_docs_to_update)})")

            # Save catalog incrementally every 10 documents
            if (i + 1) % 10 == 0:
                catalog.save(Path(catalog_path))
                logging.info(f"Incremental save: {i+1}/{len(all_docs_to_update)} documents processed")

        except Exception as e:
            logging.error(f"Failed to process {doc_path}: {e}")
            # Continue processing remaining documents

    # Final save to ensure all updates are persisted
    catalog.save(Path(catalog_path))
    logging.info(f"Final save: catalog saved to {catalog_path}")

    # Store in Weaviate for vector search
    if updated_entries:
        try:
            db.add_catalog_entries(updated_entries)
            logging.info(f"Added {len(updated_entries)} entries to Weaviate catalog")
        except Exception as e:
            logging.warning(f"Could not update Weaviate catalog: {e}")

    return {
        "total_docs": len(catalog.docs),
        "updated_docs": len(updated_entries),
        "removed_docs": removed_count,
        "catalog_path": catalog_path,
    }
