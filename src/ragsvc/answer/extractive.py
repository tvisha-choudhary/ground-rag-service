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
    """Confidence is answer-term coverage, not retrieval score.

    The first version gated on retrieval confidence and the measured
    abstention curve was flat: unanswerable questions about real
    services kept a 47 to 54 percent false answer rate at every
    threshold, because the service's page matches strongly whether or
    not the asked-about attribute exists anywhere. Retrieval score
    answers "did I find the right page"; it cannot answer "does the
    page contain this fact". Coverage of the question's content terms
    by the selected sentence answers the second question, and it is
    already on a [0, 1] scale, so the gate needs no score calibration
    across retrievers.
    """
    if not ranked_chunks:
        return Answer(question, None, (), 0.0, True)

    q_terms = _content_terms(question)
    if not q_terms:
        return Answer(question, None, (), 0.0, True)

    best = None  # (coverage, retrieval_score, sentence, chunk)
    for chunk, retrieval_score in ranked_chunks:
        for sentence in split_sentences(chunk.text):
            coverage = len(q_terms & _content_terms(sentence)) / len(q_terms)
            key = (coverage, retrieval_score)
            if best is None or key > best[:2]:
                best = (coverage, retrieval_score, sentence, chunk)

    coverage, _score, sentence, chunk = best
    if coverage < abstain_below:
        return Answer(question, None, (), round(coverage, 4), True)
    return Answer(question=question, text=sentence,
                  citations=(chunk.chunk_id,),
                  confidence=round(coverage, 4), abstained=False)
