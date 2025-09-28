"""
Document processing pipeline operations.

This module contains high-level operations for the docweaver document processing pipeline.
Each function corresponds to a stage in the sequential pipeline process.
"""
from pathlib import Path
from typing import Dict, Any
import logging
import json

from . import db
from .utils import chunk_text
from .agents import docs_search_agent, DocSearchDeps


def prep_database(
    docs_path: str = "docs/docs/weaviate/",
    reset_collection: bool = True
) -> Dict[str, Any]:
    """
    Reset and populate the document database.

    This operation deletes the existing collection, creates a new one,
    and populates it with chunks from markdown files in the specified path.

    Args:
        docs_path: Path to the documentation directory (default: "docs/docs/weaviate/")
        reset_collection: Whether to reset the collection (default: True)

    Returns:
        Dict containing processing results with keys:
        - files_processed: Number of files successfully processed
        - files: List of file paths that were processed
    """
    logging.info("Starting database preparation...")

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

    md_files = Path(docs_path).rglob("*.md*")
    md_files = [f for f in md_files if f.name[0] != "_"]

    processed_files = []
    for file in md_files:
        with open(file, "r") as f:
            logging.info(f"Importing {file}")
            text = f.read()
            chunks = chunk_text(text)
            chunk_texts = [
                {"path": file.as_posix(), "chunk": chunk.text} for chunk in chunks
            ]
            db.add_chunks(chunk_texts)
            processed_files.append(file.as_posix())

    logging.info(f"Database preparation complete. Processed {len(processed_files)} files.")
    return {"files_processed": len(processed_files), "files": processed_files}


async def search_documents(
    feature_description: str,
    output_path: str = "outputs/doc_search_agent.log"
) -> Dict[str, Any]:
    """
    Search for documents that may need editing for a given feature.

    This operation uses the docs search agent to find relevant documents
    that might require updates based on the feature description.

    Args:
        feature_description: Description of the feature to search for
        output_path: Path to save the search results (default: "outputs/doc_search_agent.log")

    Returns:
        Dict containing search results with keys:
        - documents: List of document search results
        - token_usage: Token usage information from the agent
        - output_path: Path where results were saved
    """
    logging.info(f"Starting document search for feature: {feature_description}")

    response = await docs_search_agent.run(
        f"Find documents that may need editing, for this feature: {feature_description}",
        deps=DocSearchDeps(client=db.connect()),
    )

    logging.info(f"Token usage for docs_search_agent: {response.usage()}")

    # Convert results to serializable format
    results = [doc.model_dump() for doc in response.output]

    # Save output for pipeline continuity
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    logging.info(f"Document search complete. Found {len(results)} documents. Results saved to {output_path}")

    return {
        "documents": results,
        "token_usage": response.usage(),
        "output_path": output_path
    }
