from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, ConfigDict
from docweaver.db import search_chunks
from weaviate import WeaviateClient
from pathlib import Path
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


class DocEditInstructions(BaseModel):
    path: str
    instructions: str


doc_instructor_agent = Agent(
    model="anthropic:claude-3-5-haiku-latest",
    output_type=list[DocEditInstructions],
    system_prompt=f"""
    You are an expert writer, who is now managing a team of writers.

    Review a technical summary of a feature,
    and a preliminary research of existing documents.
    {DOCUMENTATION_META_INFO}

    For each file, you can review the full document
    if it would help the decision making.

    Then, provide a set of suggestions to your writers regarding
    how to edit the documentation to reflect the feature.

    The instructions are to be succinct, and in bullet points,
    so that they are easy to review and understand.

    Leave the implementation to the writers.
    """
)


class DocEdit(BaseModel):
    """Represents a single edit in a document."""

    replace_section: str  # A verbatim section of the original document to be replaced.
    replacement_txt: str  # The new text that will replace the replace_section.


class DocOutput(BaseModel):
    path: str
    edits: list[DocEdit]


doc_writer_agent = Agent(
    model="anthropic:claude-3-5-haiku-latest",
    # model="anthropic:claude-4-sonnet-20250514",
    output_type=list[DocOutput],
    system_prompt=f"""
    You are an expert technical writer and a good developer.

    You will be given a set of instructions on
    how to update a documentation page.
    {DOCUMENTATION_META_INFO}

    Pay attention to the current style of the documentation,
    and prepare an edited page, following the provided instructions.

    The output will be a list of edits. Each edit consists of a section to replace and the replacement text.
    This will be used to programmatically apply changes to the document.
    So, please make sure that `replace_section` is a verbatim copy of a section in the original document.

    If a change is to be an addition, include the verbatim text of the section before or after,
    so that the new section(s) can be placed at the right location.

    Often, you will see that the documentation includes SDK and/or other code examples,
    which is built on top of the raw API. This is often shown with the `FilteredTextBlock` MDX component.

    Such examples can be very helpful for the users.

    Where a set of code examples should be shown for the new feature,
    indicate as such to the writer by adding this Docusaurus admonition in the documentation,
    and the writer will take care of it.
    ===== START-NEW CODE EXAMPLE ADMONITION =====
    {NEW_CODE_EXAMPLE_MARKER}
    ===== END-NEW CODE EXAMPLE ADMONITION =====
    """
)


@doc_instructor_agent.tool
@doc_writer_agent.tool
def read_doc_page(ctx: RunContext[None], path=str):
    docpath = Path(path)
    return docpath.read_text()
