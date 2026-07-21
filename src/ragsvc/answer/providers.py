"""Pluggable generation providers behind one protocol.

The service and harness depend only on this protocol. The extractive
answerer is the zero-dependency offline default and the one every
number in the README was measured with. The Anthropic provider shows
the integration seam: it receives the question plus the retrieved
chunks, must answer only from them, and must emit citations, which the
faithfulness gate then verifies exactly as it does for the extractive
path. It requires ANTHROPIC_API_KEY at runtime and is exercised by an
integration test that skips without the key; no README metric depends
on it.
"""

from __future__ import annotations

import os
from typing import Protocol

from ragsvc.answer.extractive import Answer, answer_question
from ragsvc.corpus.chunking import Chunk

PROMPT = """Answer the question using only the numbered context chunks.
Cite the chunk ids you used in square brackets after the answer, like
[doc-amber-vault:limits]. If the context does not contain the answer,
reply exactly: NOT_FOUND

Question: {question}

Context:
{context}
"""


class AnswerProvider(Protocol):
    def answer(self, question: str, ranked_chunks: list[tuple[Chunk, float]],
               abstain_below: float) -> Answer: ...


class ExtractiveProvider:
    def answer(self, question: str, ranked_chunks: list[tuple[Chunk, float]],
               abstain_below: float) -> Answer:
        return answer_question(question, ranked_chunks, abstain_below)


class AnthropicProvider:
    def __init__(self, model: str = "claude-sonnet-4-5") -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("AnthropicProvider requires ANTHROPIC_API_KEY")
        import anthropic  # deferred so the base install stays light

        self._client = anthropic.Anthropic()
        self._model = model

    def answer(self, question: str, ranked_chunks: list[tuple[Chunk, float]],
               abstain_below: float) -> Answer:
        if not ranked_chunks or ranked_chunks[0][1] < abstain_below:
            confidence = ranked_chunks[0][1] if ranked_chunks else 0.0
            return Answer(question, None, (), confidence, True)
        context = "\n".join(f"[{c.chunk_id}] {c.text}" for c, _ in ranked_chunks)
        response = self._client.messages.create(
            model=self._model, max_tokens=300,
            messages=[{"role": "user", "content": PROMPT.format(
                question=question, context=context)}])
        text = response.content[0].text.strip()
        if text == "NOT_FOUND":
            return Answer(question, None, (), ranked_chunks[0][1], True)
        cited = tuple(c.chunk_id for c, _ in ranked_chunks if f"[{c.chunk_id}]" in text)
        clean = text
        for chunk_id in cited:
            clean = clean.replace(f"[{chunk_id}]", "")
        return Answer(question, clean.strip(), cited, ranked_chunks[0][1], False)
