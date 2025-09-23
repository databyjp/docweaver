from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, ConfigDict
from docweaver.db import search_chunks
from weaviate import WeaviateClient
from pathlib import Path
import re
from helpers import DOCUMENTATION_META_INFO, NEW_CODE_EXAMPLE_MARKER


class DocSearchDeps(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: WeaviateClient


class DocSearchReturn(BaseModel):
    path: str
    reason: str


docs_search_agent = Agent(
    model="anthropic:claude-3-5-haiku-latest",
    output_type=list[DocSearchReturn],
    system_prompt=f"""
    You are a research assistant.

    You are given a user query, and you are to search the available Weaviate documentation.
    {DOCUMENTATION_META_INFO}

    Perform multiple searches with different query strings, if necessary.

    Review the returned data, and return the documents that may require updating.

    Return a list of file paths, and why each path may need to be updated.
    """
)


@docs_search_agent.tool
def search_docs(ctx: RunContext[DocSearchDeps], query=str) -> list[dict[str, str]]:
    return search_chunks(ctx.deps.client, query)


class PerFileInstructions(BaseModel):
    """A set of instructions for a single file."""
    path: str
    instructions: str

class CoordinatedEditInstructions(BaseModel):
    """A coordinated set of edit instructions for a primary document and its referenced files."""
    primary_path: str
    file_instructions: list[PerFileInstructions]


doc_instructor_agent = Agent(
    # model="anthropic:claude-3-5-haiku-latest",
    model="anthropic:claude-4-sonnet-20250514",
    output_type=list[CoordinatedEditInstructions],
    system_prompt=f"""
    You are an expert writer, who is now managing a team of writers.

    Review a technical summary of a feature,
    and a preliminary research of existing documents.
    {DOCUMENTATION_META_INFO}

    The full content for each relevant document and
    its referenced files (e.g., code snippets) has been provided to you.
    Review all of the provided context.

    Then, provide a set of suggestions to your writers regarding
    how to edit the documentation to reflect the feature.

    At Weaviate, we prefer to write code in source files
    so that they can be reviewed, automated for testing, and maintained.

    So, when adding any code examples,
    you should instruct the writer to add them to
    the markdown's associated source files (like `.py`, `.ts`, etc.).

    In other words, be sure to include instructions to edit the source files directly,
    rather than the markdown files.

    The instructions are to be succinct, and in bullet points,
    so that they are easy to review and understand.

    Leave the implementation to the writers.

    **Output Structure:**
    For each primary document that needs changes, you MUST generate one `CoordinatedEditInstructions` object.
    - `primary_path` should be the path to the main document file (e.g., an `.mdx` file).
    - `file_instructions` should be a list containing instructions for the primary file AND for each referenced file (like code files) that also needs to be changed.
    - Each item in `file_instructions` must contain the `path` and the specific `instructions` for that single file.
    - If a primary document and its two referenced code files need changes, you will create one `CoordinatedEditInstructions` object with three items in its `file_instructions` list.
    """
)


class DocEdit(BaseModel):
    """Represents a single edit in a document."""

    replace_section: str  # A verbatim section of the original document to be replaced.
    replacement_txt: str  # The new text that will replace the replace_section.


class DocOutput(BaseModel):
    path: str
    edits: list[DocEdit]
    referenced_file_edits: dict[str, list[DocEdit]] = {}


class WeaviateDoc(BaseModel):
    path: str
    doc_body: str
    referenced_docs: list["WeaviateDoc"]


def parse_doc_refs(file_path: Path, max_depth: int = 2, current_depth: int = 0) -> WeaviateDoc:
    if not file_path.exists():
        return WeaviateDoc(path=str(file_path), doc_body="", referenced_docs=[])

    content = file_path.read_text()

    if current_depth >= max_depth:
        return WeaviateDoc(path=str(file_path), doc_body=content, referenced_docs=[])

    # Find only code files and .mdx/.md includes
    import_pattern = r'import\s+\w+\s+from\s+["\'](?:!!raw-loader!)?([^"\']+\.(?:mdx?|py|ts|js|java|go|cpp|c|rb|php|rs))["\']'
    matches = re.findall(import_pattern, content)

    referenced_docs = []
    for match in matches:
        # Simple path resolution - adjust as needed
        if match.startswith('/'):
            import_path = Path("docs") / match.lstrip('/')
        else:
            import_path = Path("docs") / match

        ref_doc = parse_doc_refs(import_path, max_depth, current_depth + 1)
        referenced_docs.append(ref_doc)

    return WeaviateDoc(
        path=str(file_path),
        doc_body=content,
        referenced_docs=referenced_docs
    )


doc_writer_agent = Agent(
    # model="anthropic:claude-3-5-haiku-latest",
    model="anthropic:claude-4-sonnet-20250514",
    output_type=list[DocOutput],
    system_prompt=f"""
    You are an expert technical writer and a good developer.

    You will be given a set of instructions on
    how to update a documentation page, and/or any referenced pages.
    {DOCUMENTATION_META_INFO}

    Pay attention to the current style of the documentation,
    and prepare an edited page, following the provided instructions.

    The output will be a list of edits. Each edit consists of a section to replace and the replacement text.
    Make sure that `replace_section` is a verbatim copy of a section in the original document, so that the new section(s) can be placed at the right location.

    When making edits, you can modify both the main document and any referenced component files.
    - Use `edits` for changes to the main document
    - Use `referenced_file_edits` for changes to component files
    - Make sure each edit's `replace_section` exactly matches content in the target file

    **CRITICAL INSTRUCTIONS FOR EDITING CODE EXAMPLES:**

    You will often find code examples embedded in Markdown files (`.mdx`) using a component called `<FilteredTextBlock>`.
    This component imports code from external source files (like `.py`, `.ts`, etc.).

    You MUST adhere to the following rules:
    1.  **NEVER edit code examples directly within the Markdown files.**
    2.  **ALWAYS find the original source code file** (it will be provided to you) and apply edits there.
    3.  Place edits for source code files in the `referenced_file_edits` field.

    For example, if `docs/main.mdx` contains:
    ```mdx
    <FilteredTextBlock
      code={{'/path/to/example.py'}}
      startMarker="START: SomeExample"
      endMarker="END: SomeExample"
    />
    ```
    And you need to change the code, you will find `/path/to/example.py` in the context and create an edit for it. Your output for this change would be:
    ```json
    {{
      "path": "docs/main.mdx",
      "edits": [],
      "referenced_file_edits": {{
        "/path/to/example.py": [
          {{
            "replace_section": "...", // old code
            "replacement_txt": "..." // new code
          }}
        ]
      }}
    }}
    ```

    When adding **NEW** code examples:
    1.  Add placeholder code to the appropriate source file.
    2.  Include this exact marker comment where the code needs to be completed:
        {NEW_CODE_EXAMPLE_MARKER}
    3.  In the parent `.mdx` file, add a `<FilteredTextBlock>` component that points to the new markers in the source file.
    """
)


@doc_instructor_agent.tool
@doc_writer_agent.tool
def read_doc_page(ctx: RunContext[None], path=str):
    docpath = Path(path)
    weaviate_doc = parse_doc_refs(docpath)
    return weaviate_doc
