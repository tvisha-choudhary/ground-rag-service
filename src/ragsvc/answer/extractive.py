"""Extractive answerer with a citation contract and an abstention gate.

The contract this module enforces is the point of the repo:

1. Every answer carries citations: the chunk ids whose text supports it.
2. If retrieval confidence falls below the abstention threshold, the
   service says "not found in the handbook" instead of guessing. The
   threshold trades coverage against wrong answers; the harness measures
   that curve on questions known to be unanswerable.

Generation is pluggable (providers.py). This extractive answerer is the
offline default: it selects the best supporting sentence from the top
chunks by weighted term overlap with the question. It cannot hallucinate
by construction, which makes it the honest baseline the faithfulness
gate is calibrated against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragsvc.corpus.chunking import Chunk
from ragsvc.retrieval.bm25 import tokenize

STOPWORDS = {"the", "a", "an", "is", "are", "for", "to", "of", "in", "on",
             "what", "which", "how", "who", "where", "does", "do", "its"}


@dataclass(frozen=True)
class Answer:
    question: str
    text: str | None
    citations: tuple[str, ...]
    confidence: float
    abstained: bool


def _content_terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOPWORDS}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def answer_question(question: str, ranked_chunks: list[tuple[Chunk, float]],
                    abstain_below: float) -> Answer:
    if not ranked_chunks:
        return Answer(question, None, (), 0.0, True)
    top_confidence = ranked_chunks[0][1]
    if top_confidence < abstain_below:
        return Answer(question, None, (), top_confidence, True)

    q_terms = _content_terms(question)
    best_sentence, best_chunk, best_score = None, None, -1.0
    for chunk, retrieval_score in ranked_chunks:
        for sentence in split_sentences(chunk.text):
            overlap = len(q_terms & _content_terms(sentence))
            score = overlap + 0.25 * retrieval_score
            if score > best_score:
                best_sentence, best_chunk, best_score = sentence, chunk, score

    if best_sentence is None:
        return Answer(question, None, (), top_confidence, True)
    return Answer(question=question, text=best_sentence,
                  citations=(best_chunk.chunk_id,),
                  confidence=top_confidence, abstained=False)
