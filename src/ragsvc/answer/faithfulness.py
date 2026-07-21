"""The citation gate: does the cited text actually support the answer?

An answer is faithful when every content term it asserts appears in the
text of its cited chunks (numbers and entities carry the claim; glue
words do not). The extractive answerer passes by construction, which is
what calibrates the gate. A generative provider must clear the same bar,
so a model that paraphrases beyond its citations, or cites chunk A while
asserting a value from chunk B, gets caught before the answer ships.

This is a lexical entailment proxy, not NLI. Its failure modes are
documented in the README's honest boundaries; its virtue is that it runs
in microseconds, offline, on every single response.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragsvc.answer.extractive import _content_terms
from ragsvc.corpus.chunking import Chunk


@dataclass(frozen=True)
class FaithfulnessVerdict:
    supported: bool
    support_ratio: float
    unsupported_terms: tuple[str, ...]


def check_support(answer_text: str, cited_chunks: list[Chunk],
                  threshold: float = 0.5) -> FaithfulnessVerdict:
    if not answer_text or not cited_chunks:
        return FaithfulnessVerdict(False, 0.0, ())
    claim_terms = _content_terms(answer_text)
    if not claim_terms:
        return FaithfulnessVerdict(False, 0.0, ())
    cited_terms = set()
    for chunk in cited_chunks:
        cited_terms |= _content_terms(chunk.text)
    supported = claim_terms & cited_terms
    ratio = len(supported) / len(claim_terms)
    missing = tuple(sorted(claim_terms - cited_terms))
    return FaithfulnessVerdict(ratio >= threshold, round(ratio, 4), missing)
