"""Okapi BM25, implemented from scratch.

About sixty lines of the algorithm that still wins or ties most
enterprise retrieval bakeoffs. Written out rather than imported so
every scoring decision is inspectable and the harness's numbers trace
to arithmetic in this file. k1 and b are the textbook defaults.
"""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; hyphenated service names stay whole tokens,
    which matters in a corpus whose entities look like quartz-courier."""
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[list[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._doc_len: list[int] = []
        self._avg_len: float = 0.0
        self._term_freqs: list[Counter[str]] = []

    def fit(self, texts: list[str]) -> "BM25Index":
        if not texts:
            raise ValueError("cannot index an empty corpus")
        self._docs = [tokenize(t) for t in texts]
        self._term_freqs = [Counter(d) for d in self._docs]
        self._doc_len = [len(d) for d in self._docs]
        self._avg_len = sum(self._doc_len) / len(self._doc_len)
        self._doc_freq = Counter()
        for tf in self._term_freqs:
            self._doc_freq.update(tf.keys())
        return self

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._doc_freq.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = [0.0] * len(self._docs)
        for term in terms:
            idf = self._idf(term)
            for i, tf in enumerate(self._term_freqs):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                norm = f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self._doc_len[i] / self._avg_len))
                out[i] += idf * norm
        return out

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        scored = self.scores(query)
        order = sorted(range(len(scored)), key=lambda i: scored[i], reverse=True)
        return [(i, scored[i]) for i in order[:k]]
