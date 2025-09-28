"""
Document processing pipeline operations.

This module contains high-level operations for the docweaver document processing pipeline.
Each function corresponds to a stage in the sequential pipeline process.
"""
from pathlib import Path
from typing import Dict, Any
import logging

from . import db
from .utils import chunk_text


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
