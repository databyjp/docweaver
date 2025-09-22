from typing import Sequence
from chonkie import RecursiveChunker
from chonkie.types import (
    Chunk,
)


def chunk_text(text: str) -> Sequence[Chunk]:
    chunker = RecursiveChunker.from_recipe("markdown", lang="en")
    return chunker.chunk(text)
