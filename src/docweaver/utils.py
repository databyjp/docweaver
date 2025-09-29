from typing import Sequence
from chonkie import TokenChunker
from chonkie.types import (
    Chunk,
)


def chunk_text(text: str) -> Sequence[Chunk]:
    # Basic initialization with default parameters
    chunker = TokenChunker(
        tokenizer="word",  # Default tokenizer (or use "gpt2", etc.)
        chunk_size=512,    # Maximum tokens per chunk
        chunk_overlap=64  # Overlap between chunks
    )
    return chunker.chunk(text)
