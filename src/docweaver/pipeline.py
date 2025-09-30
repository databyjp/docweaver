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
)
from .catalog import DocCatalog, get_docs_to_update, generate_metadata
import time
import difflib
from collections import defaultdict
from datetime import datetime
from rich.console import Console
from rich.syntax import Syntax
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
        doc_bundle = parse_doc_refs(Path(filepath), include_code_body=False)

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


async def make_changes(
    feature_description: str,
    instructions_path: str = "outputs/doc_instructor_agent.log",
    output_path: str = "outputs/doc_writer_agent.log",
    edits_path: str = "outputs/doc_writer_agent_edits.log",
) -> Dict[str, Any]:
    """
    Apply editing instructions to generate revised documents.

    This operation processes the coordination instructions and uses the doc writer agent
    to generate actual document changes.

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

    async def process_instruction(instruction_bundle: dict):
        """Process a single instruction bundle."""
        primary_path = instruction_bundle["primary_path"]
        file_instructions = instruction_bundle["file_instructions"]

        all_paths = {primary_path}
        for instr in file_instructions:
            if not instr["path"].startswith("docs/"):
                instr["path"] = "docs/" + instr["path"]
            all_paths.add(instr["path"])

        original_contents = {}
        prompt_docs_list = []

        for path_str in all_paths:
            path = Path(path_str)
            content = path.read_text()
            original_contents[path_str] = content

            doc_type = "MAIN FILE" if path_str == primary_path else "REFERENCED FILE"
            prompt_docs_list.append(f"## File: {path_str} ({doc_type})\n{content}\n")

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

        start_time = time.time()
        logging.info(f"Processing instructions for primary file: {primary_path}")
        response = await doc_writer_agent.run(prompt)
        logging.info(
            f"Token usage for doc_writer_agent (primary_path: {primary_path}): {response.usage()}"
        )

        end_time = time.time()
        logging.info(
            f"Finished processing instructions for primary file: {primary_path} in {end_time - start_time:.2f} seconds."
        )

        all_edits = [o.model_dump() for o in response.output]
        revised_contents = original_contents.copy()

        # Group all edits by file path
        edits_by_file = defaultdict(list)
        for doc_output in all_edits:
            if doc_output.get("edits"):
                edits_by_file[doc_output["path"]].extend(doc_output["edits"])
            for path, ref_edits in doc_output.get("referenced_file_edits", {}).items():
                edits_by_file[path].extend(ref_edits)

        # Apply edits to all files using a more robust method
        for path, edits in edits_by_file.items():
            original_content = original_contents.get(path)
            if original_content is None:
                logging.warning(
                    f"No original content found for {path}, skipping edits."
                )
                continue

            processed_edits = []
            for edit in edits:
                replace_section = edit["replace_section"]
                # Use finditer to locate all occurrences
                matches = list(
                    re.finditer(re.escape(replace_section), original_content)
                )

                if not matches:
                    logging.warning(
                        f"Edit section not found in {path}, skipping edit. "
                        f"Section: {replace_section[:100]}..."
                    )
                    continue

                if len(matches) > 1:
                    logging.warning(
                        f"Edit section is not unique in {path} ({len(matches)} occurrences), "
                        f"applying to first one. Section: {replace_section[:100]}..."
                    )

                # Process the first match
                match = matches[0]
                processed_edits.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "replacement": edit["replacement_txt"],
                    }
                )

            # Sort edits by start position in reverse order to avoid index shifting issues
            processed_edits.sort(key=lambda e: e["start"], reverse=True)

            # Apply edits to the original content
            content_list = list(original_content)
            for edit in processed_edits:
                content_list[edit["start"] : edit["end"]] = list(edit["replacement"])

            revised_contents[path] = "".join(content_list)

        return [
            {
                "path": path,
                "revised_doc": content,
                "edits": all_edits,
            }
            for path, content in revised_contents.items()
            if original_contents.get(path) != content
        ]

    all_responses_with_edits = []
    for instruction_bundle in doc_instructions:
        result = await process_instruction(instruction_bundle)
        if result:
            all_responses_with_edits.extend(result)

    # Save raw edits
    edits_to_log = [
        {"path": r.get("path"), "edits": r.get("edits")}
        for r in all_responses_with_edits
    ]
    Path(edits_path).parent.mkdir(parents=True, exist_ok=True)
    with open(edits_path, "w") as f:
        json.dump(edits_to_log, f, indent=4)
    logging.info(f"Raw edits from agent logged to {edits_path}")

    # Save revised documents
    revised_docs_to_log = [
        {"path": r.get("path"), "revised_doc": r.get("revised_doc")}
        for r in all_responses_with_edits
    ]
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
        "files_changed": len(all_responses_with_edits),
        "total_processing_time": total_time,
        "output_path": output_path,
        "edits_path": edits_path,
    }


def create_diffs(
    changes_path: str = "outputs/doc_writer_agent.log",
    output_path: str = "outputs/diffs.log",
) -> Dict[str, Any]:
    """
    Create unified diffs from the revised documents.

    This operation compares the original files with the revised versions
    and generates unified diffs for review.

    Args:
        changes_path: Path to the revised documents file (default: "outputs/doc_writer_agent.log")
        output_path: Path to save the diffs (default: "outputs/diffs.log")

    Returns:
        Dict containing diff results with keys:
        - diffs_created: Number of diffs created
        - output_path: Path where diffs were saved
        - has_changes: Whether any changes were found
    """
    logging.info("Creating diffs from revised documents.")

    with open(changes_path, "r") as f:
        proposed_changes = json.load(f)

    all_diffs = ""
    console = Console()
    diffs_created = 0

    for change in proposed_changes:
        file_path_str = change["path"]
        file_path = Path(file_path_str)

        original_content = ""
        if file_path.exists():
            original_content = file_path.read_text()

        diff = "".join(
            difflib.unified_diff(
                original_content.splitlines(keepends=True),
                change["revised_doc"].splitlines(keepends=True),
                fromfile=f"a/{file_path_str}",
                tofile=f"b/{file_path_str}",
                n=5,  # Show 5 lines of context around changes
            )
        )

        if diff:
            all_diffs += diff
            diffs_created += 1
            console.print(f"Diff for {file_path_str}:")
            syntax = Syntax(
                diff, "diff", theme="monokai", line_numbers=False, word_wrap=True
            )
            console.print(syntax)
            console.print("-" * 80)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(all_diffs)

    logging.info(f"Created {diffs_created} diffs. Saved to {output_path}")

    return {
        "diffs_created": diffs_created,
        "output_path": output_path,
        "has_changes": diffs_created > 0,
    }


def create_pr(
    title: str = None,
    body: str = None,
    branch_name: str = None,
    diffs_path: str = "outputs/diffs.log",
    docs_path: str = "docs",
) -> Dict[str, Any]:
    """
    Apply diffs and create a pull request.

    This operation applies the generated diffs to the actual files and creates
    a pull request with the changes.

    Args:
        title: PR title (auto-generated if None)
        body: PR description (auto-generated if None)
        branch_name: Branch name (auto-generated if None)
        diffs_path: Path to the diffs file (default: "outputs/diffs.log")
        docs_path: Path to the docs repository (default: "docs")

    Returns:
        Dict containing PR results with keys:
        - pr_url: URL of the created pull request (if successful)
        - branch_name: Name of the created branch
        - success: Whether the operation succeeded
    """
    logging.info("Creating pull request with changes.")
    console = Console()

    # Check if diffs exist
    if not Path(diffs_path).exists():
        raise FileNotFoundError("No diffs.log found. Run create_diffs first.")

    with open(diffs_path, "r") as f:
        diff_content = f.read()

    # Check if there are any diffs to apply
    if Path(diffs_path).stat().st_size == 0:
        console.print("✅ No diffs to apply, skipping.")
        return {"success": False, "message": "No changes to apply"}

    # Work in the docs directory
    docs_path = Path(docs_path)
    if not docs_path.exists():
        raise FileNotFoundError("docs/ directory not found")

    repo = git.Repo(docs_path)
    original_branch = repo.active_branch
    temp_diff_path = None

    try:
        # Apply diffs
        modified_diff = diff_content.replace("--- a/docs/", "--- a/").replace(
            "+++ b/docs/", "+++ b/"
        )

        temp_diff_path = docs_path / "temp_diffs.patch"
        with open(temp_diff_path, "w") as f:
            f.write(modified_diff)

        repo.git.apply("--verbose", "temp_diffs.patch")
        console.print("✅ Diffs applied successfully")

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
        if title is None:
            timestamp_readable = datetime.now().strftime("%Y-%m-%d %H:%M")
            title = f"📚 DocWeaver updates - {timestamp_readable}"

        if body is None:
            body = """This PR contains automated documentation improvements generated by DocWeaver.

## Changes Summary
- Content updates and clarifications
- Formatting improvements
- Enhanced code examples
- Better structural organization

🤖 Generated with DocWeaver"""

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
        )

        console.print(f"✅ Pull request created: {pull.html_url}")

        return {
            "success": True,
            "branch_name": branch_name,
            "pr_url": pull.html_url,
            "title": title,
            "body": body,
            "message": f"Pull request for branch '{branch_name}' created successfully.",
        }

    except git.exc.GitCommandError as e:
        raise RuntimeError(f"Failed to apply diffs: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to process changes: {e}")
    finally:
        # Clean up temp file if it exists
        if temp_diff_path and temp_diff_path.exists():
            temp_diff_path.unlink()
            console.print("✅ Temporary diff file removed.")

        # Switch back to the original branch and clean up worktree
        try:
            original_branch.checkout(force=True)
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

    Args:
        docs_paths: List of documentation directories (default: from config.DOCS_PATHS)
        catalog_path: Path to save catalog JSON (default: "outputs/catalog.json")
        limit: Optional limit on number of documents to process

    Returns:
        Dict containing results with keys:
        - total_docs: Total documents in catalog
        - updated_docs: Number of documents updated
        - catalog_path: Path where catalog was saved
    """
    logging.info("Starting catalog update...")

    if docs_paths is None:
        docs_paths = DOCS_PATHS

    catalog = DocCatalog.load(Path(catalog_path))
    logging.info(f"Loaded catalog with {len(catalog.docs)} existing documents")

    # Find documents that need updating across all paths
    # Use common base path for consistent relative paths
    base_path = Path(DOCS_BASE_PATH)
    all_docs_to_update = []
    for docs_path in docs_paths:
        doc_root = Path(docs_path)
        docs_to_update = get_docs_to_update(doc_root, catalog)
        all_docs_to_update.extend(docs_to_update)

    if limit:
        all_docs_to_update = all_docs_to_update[:limit]
        logging.info(f"Limited to {limit} documents")

    if not all_docs_to_update:
        logging.info("No documents to update")
        return {
            "total_docs": len(catalog.docs),
            "updated_docs": 0,
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
        "catalog_path": catalog_path,
    }
