from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, ConfigDict
from docweaver.db import search_chunks
from weaviate import WeaviateClient
from pathlib import Path


class DocSearchDeps(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: WeaviateClient


class DocSearchReturn(BaseModel):
    path: str
    reason: str


docs_search_agent = Agent(
    model="anthropic:claude-3-5-haiku-latest",
    output_type=list[DocSearchReturn],
    system_prompt="""
    You are a research assistant.

    You are given a user query, and you are to search the available Weaviate documentation.
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
    system_prompt="""
    You are an expert writer, who is now managing a team of writers.

    Review a technical summary of a feature,
    and a preliminary research of existing documents.

    For each file, you can review the full document
    if it would help the decision making.

    Then, provide a set of suggestions to your writers regarding
    how to edit the documentation to reflect the feature.

    The instructions are to be succinct, and in bullet points,
    so that they are easy to review and understand.

    Leave the implementation to the writers.
    """
)


class DocOutput(BaseModel):
    path: str
    revised_doc: str


doc_writer_agent = Agent(
    model="anthropic:claude-3-5-haiku-latest",
    # model="anthropic:claude-4-sonnet-20250514",
    output_type=list[DocOutput],
    system_prompt="""
    You are an expert technical writer and a good developer.

    You will be given a set of instructions on
    how to update a documentation page.

    Pay attention to the current style of the documentation,
    and prepare an edited page, following the provided instructions.

    The output will be used to produce a git-style diff.
    So, if you are truncating any existing parts of the documentation,
    please make sure to include existing lines so that the diff will clearly pick those up.
    """
)


@doc_instructor_agent.tool
@doc_writer_agent.tool
def read_doc_page(ctx: RunContext[None], path=str):
    docpath = Path(path)
    return docpath.read_text()
