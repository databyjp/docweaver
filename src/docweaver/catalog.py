"""Document catalog for storing and retrieving document metadata."""

import hashlib
import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent

logger = logging.getLogger(__name__)

DOCTYPES = Literal["concept", "how-to", "reference", "tutorial"]


class DocMetadata(BaseModel):
    """Metadata for a single documentation file."""

    path: str = Field(description="Path to the documentation file")
    title: Optional[str] = Field(default=None, description="Title of the document")
    topics: Optional[list[str]] = Field(
        default=None, description="Topics covered by this document"
    )
    doctype: Optional[DOCTYPES] = Field(
        default=None, description="Document type according to Diataxis framework"
    )
    summary: Optional[str] = Field(
        default=None, description="Short summary describing what this document covers"
    )
    hash: Optional[str] = Field(default=None, description="Hash of the document file")


class DocCatalog(BaseModel):
    """Collection of document metadata."""

    docs: dict[str, DocMetadata] = Field(
        default_factory=dict,
        description="Dictionary of document metadata indexed by path",
    )

    @classmethod
    def load(cls, path: Path) -> "DocCatalog":
        """Load catalog from JSON file, or create empty if doesn't exist."""
        if path.exists():
            return cls.model_validate_json(path.read_text())
        return cls()

    def save(self, path: Path):
        """Save catalog to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))


# Agent for generating document metadata
doc_metadata_agent = Agent(
    model="anthropic:claude-3-5-haiku-latest",
    output_type=DocMetadata,
    system_prompt="""You are a documentation metadata generator.

    Generate metadata for documentation files:
    - topics: Technical topics covered (list of strings)
    - doctype: Document type per Diataxis framework (must be one of: concept, how-to, reference, tutorial)
    - summary: Brief 1-2 sentence description of the document content

    Be concise and accurate.""",
)


def get_docs_to_update(
    doc_root: Path, catalog: DocCatalog, base_path: Path
) -> list[Path]:
    """Find documents that are new or have changed since last cataloging."""
    all_docs = [p for p in doc_root.rglob("*.md*") if not p.name.startswith("_")]
    docs_to_update = []

    for doc_path in all_docs:
        relative_path = doc_path.relative_to(base_path).as_posix()
        current_hash = hashlib.sha256(doc_path.read_bytes()).hexdigest()

        if relative_path not in catalog.docs:
            docs_to_update.append(doc_path)
        elif catalog.docs[relative_path].hash != current_hash:
            docs_to_update.append(doc_path)

    return docs_to_update


def get_docs_to_remove(doc_root: Path, catalog: DocCatalog, base_path: Path) -> list[str]:
    """Find documents in the catalog that no longer exist on disk."""
    # Build a set of all current document paths relative to the base_path
    all_docs_on_disk = {
        p.relative_to(base_path).as_posix()
        for p in doc_root.rglob("*.md*")
        if not p.name.startswith("_")
    }

    # Find paths that are in the catalog but not on disk anymore
    docs_in_catalog = set(catalog.docs.keys())
    docs_to_remove = [
        path for path in docs_in_catalog if path not in all_docs_on_disk
    ]
    return docs_to_remove


async def generate_metadata(doc_path: Path, base_path: Path) -> DocMetadata:
    """Generate metadata for a single document using AI."""
    content = doc_path.read_text()
    relative_path = doc_path.relative_to(base_path).as_posix()

    prompt = f"Generate metadata for this documentation file at '{relative_path}':\n\n{content}"

    try:
        response = await doc_metadata_agent.run(prompt)
        metadata = response.output

        # Add computed fields
        metadata.path = relative_path
        metadata.hash = hashlib.sha256(content.encode()).hexdigest()

        # Extract title from frontmatter if available
        if "title: " in content:
            metadata.title = content.split("title: ")[1].split("\n")[0].strip()
        else:
            metadata.title = doc_path.stem

        logger.info(f"Generated metadata for {relative_path}")
        return metadata

    except Exception as e:
        logger.error(f"Failed to generate metadata for {relative_path}: {e}")
        # Return minimal metadata on failure
        return DocMetadata(
            path=relative_path,
            title=doc_path.stem,
            hash=hashlib.sha256(content.encode()).hexdigest(),
        )
