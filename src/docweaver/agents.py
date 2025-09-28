from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, ConfigDict
from docweaver.db import search_chunks
from weaviate import WeaviateClient
from pathlib import Path
import re
import logging
from helpers import DOCUMENTATION_META_INFO, NEW_CODE_EXAMPLE_MARKER
import os


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
    You are an expert researcher knowledgeable in vector databases, especially Weaviate.
    You are to search the available Weaviate documentation to find relevant pages that may need editing.

    {DOCUMENTATION_META_INFO}

    Based on the provided task, find documents that might require updating, editing,
    or should be considered for consistency.

    Review the returned data and identify documents that might require updating.

    A subsequent reviewer will examine them in detail and decide what to edit.
    As a result, your job is to overfetch; that is, return more than what may actually be edited.
    """,
)


@docs_search_agent.tool
def search_docs(
    ctx: RunContext[DocSearchDeps], queries=list[str]
) -> list[dict[str, str]]:
    logging.info(f"Executing tool 'search_docs' with queries: {queries}")
    return search_chunks(ctx.deps.client, queries)


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
    You are an expert writer and developer managing a team of capable writers.

    Your job is to review a documentation task and instruct writers on how to
    update the existing documentation.

    {DOCUMENTATION_META_INFO}

    The full content for each relevant document will be provided to you,
    along with any referenced files (e.g., code snippets or markdown).
    Sometimes you will get the referenced files in full, but not always.

    Review all provided context and provide clear instructions to your writers.

    For example, you can include:
    - Which files need editing and why
    - What specific changes should be made
    - How to maintain consistency with existing documentation style
    - Any cross-references that need updating

    At Weaviate, we prefer to write code in source files so they can be reviewed,
    automated for testing, and maintained. When adding code examples, instruct
    writers to add them to the markdown's associated source files (like `.py`, `.ts`, etc.).

    The writers are capable but relatively junior, so provide clear,
    unambiguous instructions. Where possible, provide placeholder code snippets
    for them to add to the appropriate source file.
    """,
)


class DocEdit(BaseModel):
    """Represents a single edit in a document."""

    replace_section: str  # A verbatim section of the original document to be replaced.
    replacement_txt: str  # The new text that will replace the replace_section.


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
    # model="anthropic:claude-3-5-haiku-latest",
    model="anthropic:claude-4-sonnet-20250514",
    output_type=list[DocOutput],
    system_prompt=f"""
    You are a great technical writer and developer, who is very familar with Weaviate.

    You will be given a set of instructions on
    how to update a documentation page, and/or any referenced pages.
    {DOCUMENTATION_META_INFO}

    Pay attention to the current style of the documentation,
    and prepare an edited page, following the provided instructions.

    The output will be a list of edits. Each edit consists of a section to replace and the replacement text.
    Make sure that `replace_section` is a verbatim copy of a section in the original document,
    so that the new section(s) can be placed at the right location.

    When making edits, you can modify both the main document and any referenced component files.
    - Use `edits` for changes to the main document
    - Use `referenced_file_edits` for changes to component files
    - Make sure each edit's `replace_section` exactly matches content in the target file

    Your final output must be ONLY a valid JSON list of objects, conforming to the specified schema.
    Do not include any other text, markdown formatting, or explanations.

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
    And you need to change the code, you will find `/path/to/example.py` in the context and create an edit for it. Your output for this change would be a list containing one object like this:
    ```json
    [
      {{
        "path": "docs/main.mdx",
        "edits": [],
        "referenced_file_edits": {{
          "/path/to/example.py": [
            {{
              "replace_section": "# Old code to be replaced...",
              "replacement_txt": "# New replacement code..."
            }}
          ]
        }}
      }}
    ]
    ```

    When adding **NEW** code examples:
    1.  Add placeholder code to the appropriate source file.
    2.  Include this exact marker comment where the code needs to be completed:
        {NEW_CODE_EXAMPLE_MARKER}
    3.  In the parent `.mdx` file, add a `<FilteredTextBlock>` component that points to the new markers in the source file.
    """,
)
