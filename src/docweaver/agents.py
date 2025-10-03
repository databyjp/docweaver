from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, ConfigDict
from docweaver.db import search_chunks, search_catalog
from docweaver.catalog import DocCatalog
from weaviate import WeaviateClient
from pathlib import Path
import re
import logging
from .helpers import DOCUMENTATION_META_INFO, NEW_CODE_EXAMPLE_MARKER
import os
from enum import Enum


class DocSearchDeps(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: WeaviateClient
    catalog: DocCatalog | None = None


class DocSearchReturn(BaseModel):
    path: str
    reason: str


docs_search_agent = Agent(
    model="anthropic:claude-3-5-haiku-latest",
    output_type=list[DocSearchReturn],
    retries=3,
    system_prompt=f"""
    You are an expert researcher knowledgeable in vector databases, especially Weaviate.
    You are to search the available Weaviate documentation to find relevant pages that may need editing.

    {DOCUMENTATION_META_INFO}

    Based on the provided task, find documents that might require updating, editing,
    or should be considered for consistency.

    Review the returned data and identify documents that might require updating.

    You have access to two types of search:
    1. search_docs - searches document chunks (detailed content search)
    2. search_catalog - searches document metadata (topics, summaries, types)

    Use both search methods to get comprehensive results. The catalog search is useful for
    understanding document structure and finding related files.

    A subsequent reviewer will examine them in detail and decide what to edit.
    As a result, your job is to overfetch; that is, return more than what may actually be edited.
    """,
)


@docs_search_agent.tool
def search_docs(
    ctx: RunContext[DocSearchDeps], queries: list[str]
) -> list[dict[str, str]]:
    """Search document content chunks for relevant passages."""
    logging.info(f"Executing tool 'search_docs' with queries: {queries}")
    return search_chunks(ctx.deps.client, queries)


@docs_search_agent.tool
def search_catalog(
    ctx: RunContext[DocSearchDeps], query: str
) -> list[dict]:
    """Search document catalog for files by metadata (topics, summary, type)."""
    logging.info(f"Executing tool 'search_catalog' with query: {query}")
    if ctx.deps.catalog is None:
        logging.warning("Catalog not available, skipping catalog search")
        return []

    # Search Weaviate catalog collection
    results = []
    try:
        from docweaver.db import search_catalog as db_search_catalog
        results = db_search_catalog(ctx.deps.client, query, limit=10)
        logging.info(f"Catalog search returned {len(results)} results")
    except Exception as e:
        logging.warning(f"Catalog search failed: {e}")

    return results


class PerFileInstructions(BaseModel):
    """A set of instructions for a single file."""

    path: str
    instructions: str


class CoordinatedEditInstructions(BaseModel):
    """A coordinated set of edit instructions for a primary document and its referenced files."""

    primary_path: str
    file_instructions: list[PerFileInstructions]


doc_instructor_agent = Agent(
    model="anthropic:claude-3-5-haiku-latest",
    # model="anthropic:claude-4-sonnet-20250514",
    output_type=list[CoordinatedEditInstructions],
    retries=3,
    system_prompt=f"""
    You are an expert writer and developer managing a team of capable writers.

    Your job is to review a documentation task and instruct writers on how to
    update the existing documentation.

    {DOCUMENTATION_META_INFO}

    **CRITICAL PRESERVATION RULES**:
    The existing documentation was written by expert technical writers and represents
    high-quality, well-structured content. Your default approach should be to ADD new
    information, NOT to replace existing content.

    Only instruct writers to modify or delete existing content when you can cite SPECIFIC
    EVIDENCE that the existing text is:
    1. Factually incorrect based on the new feature description
    2. Directly contradicts the new feature's behavior or API
    3. Uses deprecated syntax/APIs that are replaced by this feature
    4. Creates confusion or ambiguity with the new feature

    **WHEN IN DOUBT, ADD INSTEAD OF REPLACE**:
    - Prefer appending new sections over rewriting existing ones
    - Prefer inserting new paragraphs over replacing existing ones
    - Prefer adding new examples over replacing working examples
    - Only remove text when you can explicitly justify why it's harmful to keep it

    **INSTRUCTION GUIDELINES**:
    When providing instructions to writers:
    - Clearly specify whether to ADD new content or MODIFY existing content
    - For modifications, cite the specific line/section and explain WHY it must change
    - Provide the exact evidence from the feature description that contradicts existing content
    - Default to preservation: if existing content is still accurate, leave it untouched
    - **EVIDENCE-BASED ADDITIONS**: Only instruct writers to add information that can be directly
      supported by the feature description, code changes, or provided context. Do not infer
      capabilities, assume behavior, or add speculative information that isn't explicitly documented.

    The full content for each relevant document will be provided to you,
    along with any referenced files (e.g., code snippets or markdown).
    Sometimes you will get the referenced files in full, but not always.

    Review all provided context and provide clear instructions to your writers.

    For example, you can include:
    - Which files need editing and why
    - What specific changes should be made (ADD vs MODIFY, with justification)
    - How to maintain consistency with existing documentation style
    - Any cross-references that need updating

    At Weaviate, we prefer to write code in source files so they can be reviewed,
    automated for testing, and maintained. When adding code examples, instruct
    writers to add them to the markdown's associated source files (like `.py`, `.ts`, etc.).

    The writers are capable but relatively junior, so provide clear,
    unambiguous instructions. Where possible, provide placeholder code snippets
    for them to add to the appropriate source file.

    **EXAMPLES OF GOOD vs BAD INSTRUCTIONS**:

    ✅ GOOD - Adding new content:
    "ADD a new section after line 45 titled '## Using the New Feature' that explains
    how to use the batch processing API. Include an example showing the new `batch_size` parameter."

    ✅ GOOD - Targeted update with evidence:
    "UPDATE lines 23-25. The current text states 'Maximum limit is 100' but the new
    feature increases this to 1000. Replace with: 'Maximum limit is 1000 (increased from 100 in v2.5)'."

    ❌ BAD - Vague rewrite instruction:
    "Rewrite the introduction section to be clearer and mention the new feature."
    (Why bad? No evidence that existing intro is unclear; should ADD mention instead)

    ❌ BAD - Unnecessary deletion:
    "Remove the section about basic usage and replace with advanced usage."
    (Why bad? Basic usage is still valid; should KEEP basic usage and ADD advanced section)

    ✅ GOOD - Enhancement without replacement:
    "After the existing pagination example on line 78, INSERT a new paragraph explaining
    the new cursor-based pagination option, with a code example."
    """,
)


class EditType(str, Enum):
    """Classification of edit types to ensure conservative editing."""
    ADD_NEW = "add_new"  # Adding entirely new content (preferred)
    UPDATE_OUTDATED = "update_outdated"  # Fixing incorrect/outdated information with evidence
    ENHANCE = "enhance"  # Adding detail to existing content without removing it
    DELETE_REDUNDANT = "delete_redundant"  # Removing duplicate/obsolete content (use sparingly)


class DocEdit(BaseModel):
    """Represents a single edit in a document, referencing line numbers."""

    comment: str  # A comment explaining the change.
    edit_type: EditType  # Classification of the edit type (required)
    justification: str  # Required explanation citing specific evidence for why this edit is necessary
    start_line: int  # The starting line number of the section to be replaced (inclusive).
    end_line: int  # The ending line number of the section to be replaced (inclusive).
    replacement_txt: str  # The new text that will replace the specified lines.


class DocOutput(BaseModel):
    """Represents all the edits for a primary file and its referenced files."""

    path: str
    edits: list[DocEdit]
    referenced_file_edits: dict[str, list[DocEdit]] = {}


class WeaviateDoc(BaseModel):
    path: str
    doc_body: str
    referenced_docs: list["WeaviateDoc"]


def parse_doc_refs(file_path: Path, include_code_body: bool = True) -> WeaviateDoc:
    """Parse document and its direct references (first level only)."""
    if not file_path.exists():
        return WeaviateDoc(path=str(file_path), doc_body="", referenced_docs=[])

    content = file_path.read_text()

    # Choose pattern based on whether code files should be included
    code_extensions_list = ["py"]
    file_extensions = r"mdx?|" + "|".join(code_extensions_list)

    import_pattern = rf'import\s+\w+\s+from\s+["\'](?:!!raw-loader!)?([^"\']+\.(?:{file_extensions}))["\']'
    matches = re.findall(import_pattern, content)

    referenced_docs = []
    for match in matches:
        if match.startswith("/"):
            import_path: Path = Path("docs") / match.lstrip("/")
        else:
            import_path: Path = Path("docs") / match

        _, ext = os.path.splitext(import_path.absolute())
        if import_path.exists():
            # Only load the referenced file if markdown or include_code_body
            if "md" in ext or include_code_body:
                ref_content = import_path.read_text()
            elif ext in code_extensions_list:
                ref_content = "Body of code not included for brevity."
            else:
                ref_content = "Body not included for brevity."

            ref_doc = WeaviateDoc(
                path=str(import_path), doc_body=ref_content, referenced_docs=[]
            )
            referenced_docs.append(ref_doc)

    return WeaviateDoc(
        path=str(file_path), doc_body=content, referenced_docs=referenced_docs
    )


doc_writer_agent = Agent(
    model="anthropic:claude-3-5-haiku-latest",
    # model="anthropic:claude-4-sonnet-20250514",
    output_type=list[DocOutput],
    retries=3,
    system_prompt=f"""
    You are a great technical writer and developer, who is very familar with Weaviate.

    You will be given a set of instructions on
    how to update a documentation page, and/or any referenced pages.
    The content of each file will be provided with line numbers.
    {DOCUMENTATION_META_INFO}

    **CRITICAL PRESERVATION PRINCIPLES**:
    The existing documentation is high-quality and written by experts. Your role is to
    ADD new information, not to rewrite existing content unless absolutely necessary.

    Follow the instructions precisely, but be EXTREMELY cautious about altering existing
    content. Only edit or delete existing text when the instructions provide SPECIFIC
    EVIDENCE that it is incorrect, outdated, or contradictory.

    **EVIDENCE-BASED ADDITIONS**:
    Only add information that can be directly supported by the provided instructions,
    feature descriptions, or context. Do not infer capabilities, assume behavior, or
    add speculative information. Every new piece of information must be traceable to
    explicit evidence in the materials provided.

    **EDIT CLASSIFICATION REQUIREMENTS**:
    For EVERY edit you create, you MUST:
    1. Set the `edit_type` field appropriately:
       - ADD_NEW: Use this for inserting new sections, paragraphs, or examples (PREFERRED)
       - UPDATE_OUTDATED: Only when instructions cite specific incorrect information
       - ENHANCE: Adding detail to existing content without removing original text
       - DELETE_REDUNDANT: Only when instructions justify why content must be removed

    2. Provide detailed `justification` that:
       - Cites specific evidence from the instructions
       - Explains WHY this edit is necessary (not just WHAT is changing)
       - For UPDATE_OUDATED or DELETE_REDUNDANT, quotes the conflicting information

    3. Write a clear `comment` summarizing the change

    **DEFAULT TO ADDITION, NOT REPLACEMENT**:
    - When adding related information, INSERT new sections rather than replacing existing ones
    - When enhancing explanations, ADD new paragraphs after existing ones
    - When adding examples, INSERT them as new examples rather than replacing working ones
    - Only REPLACE text when you can quote the specific incorrect/outdated content

    **PRESERVATION CHECKLIST** (ask yourself before each edit):
    ✓ Does the instruction explicitly identify this text as wrong/outdated?
    ✓ Can I ADD new content instead of replacing existing content?
    ✓ Is the existing text still accurate even with the new feature?
    ✓ Have I provided clear justification citing evidence from instructions?

    Pay attention to the current style of the documentation,
    and prepare an edited page, following the provided instructions.

    The output will be a list of edits. Each edit specifies a range of lines to be replaced.
    - `start_line` and `end_line` are inclusive and refer to the original document's line numbers.
    - To insert text, specify the same `start_line` and `end_line` where you want to insert, and `replacement_txt` will be inserted before that line.
    - To delete text, provide an empty `replacement_txt`.

    When making edits, you can modify both the main document and any referenced component files.
    - Use `edits` for changes to the main document
    - Use `referenced_file_edits` for changes to component files

    **CRITICAL INSTRUCTIONS FOR EDITING CODE EXAMPLES:**

    You will often find code examples embedded in Markdown files (`.mdx`) using a component called `<FilteredTextBlock>`.
    This component imports code from external source files (like `.py`, `.ts`, etc.).

    You MUST adhere to the following rules:
    1.  **NEVER edit code examples directly within the Markdown files.**
    2.  **ALWAYS find the original source code file** (it will be provided to you) and apply edits there.
    3.  Place edits for source code files in the `referenced_file_edits` field.

    When adding **NEW** code examples:
    1.  Add placeholder code to the appropriate source file.
    2.  Include this exact marker comment where the code needs to be completed:
        {NEW_CODE_EXAMPLE_MARKER}
    3.  In the parent `.mdx` file, add a `<FilteredTextBlock>` component that points to the new markers in the source file.

    **EXAMPLES OF GOOD vs BAD EDITS**:

    ✅ GOOD - Adding new section (edit_type: ADD_NEW):
    {{
      "comment": "Add new section about batch processing feature",
      "edit_type": "add_new",
      "justification": "Instructions request adding documentation for the new batch_size parameter introduced in the feature description. No existing content covers this.",
      "start_line": 45,
      "end_line": 45,
      "replacement_txt": "\\n## Batch Processing\\n\\nYou can now process multiple items..."
    }}

    ✅ GOOD - Fixing incorrect information (edit_type: UPDATE_OUTDATED):
    {{
      "comment": "Update incorrect limit value",
      "edit_type": "update_outdated",
      "justification": "Current line 23 states 'Maximum limit is 100' but feature description specifies the limit has been increased to 1000 in the new version.",
      "start_line": 23,
      "end_line": 23,
      "replacement_txt": "Maximum limit is 1000 (increased from 100 in v2.5)"
    }}

    ❌ BAD - Replacing working content unnecessarily:
    {{
      "comment": "Update introduction",
      "edit_type": "update_outdated",
      "justification": "Make it clearer",
      "start_line": 1,
      "end_line": 10,
      "replacement_txt": "Completely rewritten introduction..."
    }}
    (Why bad? No evidence existing intro is wrong; justification too vague; should use ADD_NEW to append information instead)

    ✅ GOOD - Enhancing existing content (edit_type: ENHANCE):
    {{
      "comment": "Add note about new performance characteristics",
      "edit_type": "enhance",
      "justification": "Feature description mentions 2x performance improvement. Adding this information after the existing performance section without removing current content.",
      "start_line": 67,
      "end_line": 67,
      "replacement_txt": "\\n> **Note**: As of v2.5, performance has improved by 2x for large datasets."
    }}
    """,
)

class PRContent(BaseModel):
    """Generated PR description."""

    description: str


pr_generator_agent = Agent(
    model="anthropic:claude-3-5-haiku-latest",
    output_type=PRContent,
    retries=3,
    system_prompt=f"""
    You are an expert at writing clear, concise pull request descriptions for documentation changes.

    {DOCUMENTATION_META_INFO}

    Your task is to analyze the documentation changes and generate a comprehensive description that explains what was changed and why.

    Guidelines for PR descriptions:
    - Start with a brief summary of the changes
    - List the specific files that were modified
    - Explain the purpose and benefits of the changes
    - Include any relevant context about the feature or improvement
    - Use bullet points for clarity
    - Mention if this addresses specific issues or requirements

    Focus on making the PR description helpful for reviewers and future reference.
    """,
)

