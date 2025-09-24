from pathlib import Path
import json
from docweaver.agents import doc_instructor_agent, parse_doc_refs, WeaviateDoc
import asyncio
from helpers import TECH_DESCRIPTION_RESHARDING
import logging


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    doc_search_results_path = Path("outputs/doc_search_agent.log")
    with doc_search_results_path.open(mode="r") as f:
        doc_search_results: list[dict[str, str]] = json.load(f)

    logging.info(f"Found {len(doc_search_results)} search results to process.")

    # Collect and format all docs and their references
    prompt_docs_list = []
    all_docs_content = {}  # Use a dict to avoid processing the same file twice

    def collect_docs_recursive(doc: WeaviateDoc, is_main: bool):
        if doc.path in all_docs_content:
            return  # Avoid cycles and redundant processing

        all_docs_content[doc.path] = doc.doc_body
        if not doc.doc_body:
            return  # Don't add empty files to prompt

        doc_type = "MAIN FILE" if is_main else "REFERENCED FILE"
        prompt_docs_list.append(
            f"[{doc_type}]\n"
            f"Filepath: {doc.path}\n"
            f"====== START-ORIGINAL CONTENT =====\n{doc.doc_body}\n====== END-ORIGINAL CONTENT =====\n"
        )
        for ref_doc in doc.referenced_docs:
            collect_docs_recursive(ref_doc, is_main=False)

    for result in doc_search_results:
        filepath = result.get("path")
        if not filepath:
            logging.warning(f"Skipping search result due to missing path: {result}")
            continue

        logging.info(f"Parsing document and its references: {filepath}")
        doc_bundle = parse_doc_refs(Path(filepath))
        collect_docs_recursive(doc_bundle, is_main=True)

    document_bundle_prompt = "\n".join(prompt_docs_list)

    prompt = f"""
    Weaviate has introduced this new feature.
    ====== START-FEATURE DESCRIPTION =====
    {TECH_DESCRIPTION_RESHARDING}
    ====== END-FEATURE DESCRIPTION =====

    A preliminary search has identified the following files as potentially needing updates:
    {json.dumps(doc_search_results, indent=2)}

    Here is the full content of those files and any other files they reference (like code examples):
    ====== START-DOCUMENT BUNDLE =====
    {document_bundle_prompt}
    ====== END-DOCUMENT BUNDLE =====

    Based on all of this context, please provide a set of high-level instructions for a writer to update the documentation.
    Focus on what needs to change and where, including any code examples in referenced files.
    """

    logging.info("Running doc_instructor_agent to generate edit instructions...")
    response = await doc_instructor_agent.run(prompt)

    outpath = Path("outputs/doc_instructor_agent.log")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    responses = [o.model_dump() for o in response.output]
    with outpath.open(mode="w") as f:
        json.dump(responses, f, indent=4)

    logging.info(f"Instructions saved to {outpath}")


if __name__ == "__main__":
    asyncio.run(main())
