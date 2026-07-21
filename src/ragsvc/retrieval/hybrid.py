"""Reciprocal rank fusion of the lexical and dense retrievers.

RRF(d) = sum over retrievers of 1 / (rrf_k + rank_r(d))

Rank based fusion needs no score calibration between retrievers whose
scores live on different scales (BM25 is unbounded, cosine is [-1, 1]),
which is why it beats naive score averaging without a tuning step.
The fused score is normalized by its theoretical maximum so downstream
consumers (the abstention gate) see a stable [0, 1] confidence scale.
"""

from __future__ import annotations

from ragsvc.retrieval.bm25 import BM25Index
from ragsvc.retrieval.dense import LSAIndex


class HybridIndex:
    def __init__(self, bm25: BM25Index, dense: LSAIndex, rrf_k: int = 60,
                 pool_size: int = 50) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k
        self.pool_size = pool_size

    def fit(self, texts: list[str]) -> "HybridIndex":
        self.bm25.fit(texts)
        self.dense.fit(texts)
        return self

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        fused: dict[int, float] = {}
        for ranked in (self.bm25.top_k(query, self.pool_size),
                       self.dense.top_k(query, self.pool_size)):
            for rank, (idx, _score) in enumerate(ranked):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        max_possible = 2.0 / (self.rrf_k + 1)
        order = sorted(fused, key=lambda i: fused[i], reverse=True)
        return [(i, fused[i] / max_possible) for i in order[:k]]
