"""Three chunking strategies, evaluated head to head by the harness.

The strategy names describe the unit of retrieval:

    fixed:    sliding window of N words with overlap, ignoring structure
    section:  one chunk per document section (structure aware)
    document: one chunk per whole page (the lazy baseline)

Chunk ids embed provenance (doc_id, section span) so retrieval hits can
be scored against the planted ground truth and cited by the answerer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    section_ids: tuple[str, ...]
    text: str


def chunk_fixed(documents: list[dict], size_words: int, overlap_words: int) -> list[Chunk]:
    if size_words <= 0 or not 0 <= overlap_words < size_words:
        raise ValueError("need size_words > 0 and 0 <= overlap_words < size_words")
    chunks: list[Chunk] = []
    for doc in documents:
        words: list[tuple[str, str]] = []  # (word, section_id)
        for section in doc["sections"]:
            words.extend((w, section["section_id"]) for w in section["text"].split())
        step = size_words - overlap_words
        for start in range(0, max(len(words) - overlap_words, 1), step):
            window = words[start:start + size_words]
            if not window:
                continue
            sections = tuple(dict.fromkeys(s for _, s in window))
            chunks.append(Chunk(
                chunk_id=f"{doc['doc_id']}:w{start}",
                doc_id=doc["doc_id"], section_ids=sections,
                text=" ".join(w for w, _ in window)))
    return chunks


def chunk_section(documents: list[dict]) -> list[Chunk]:
    return [Chunk(chunk_id=f"{doc['doc_id']}:{s['section_id']}",
                  doc_id=doc["doc_id"], section_ids=(s["section_id"],),
                  text=s["text"])
            for doc in documents for s in doc["sections"]]


def chunk_document(documents: list[dict]) -> list[Chunk]:
    return [Chunk(chunk_id=f"{doc['doc_id']}:full",
                  doc_id=doc["doc_id"],
                  section_ids=tuple(s["section_id"] for s in doc["sections"]),
                  text=" ".join(s["text"] for s in doc["sections"]))
            for doc in documents]


STRATEGIES = {"fixed": chunk_fixed, "section": chunk_section, "document": chunk_document}


def build_chunks(strategy: str, documents: list[dict], size_words: int = 120,
                 overlap_words: int = 30) -> list[Chunk]:
    if strategy == "fixed":
        return chunk_fixed(documents, size_words, overlap_words)
    if strategy == "section":
        return chunk_section(documents)
    if strategy == "document":
        return chunk_document(documents)
    raise ValueError(f"unknown chunking strategy: {strategy}")
