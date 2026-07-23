"""Retrieval latency and throughput on the full generated corpus.

Measures what the service actually does per request: one top_k query
against the fitted index. Reports index fit time, then per-query p50
and p95 latency and queries per second over the full question set,
for each retriever on the serving chunking (fixed). Run on the same
machine as the pipeline; numbers are hardware-dependent and the
committed results state the environment.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

from ragsvc.config import Config
from ragsvc.corpus.chunking import build_chunks
from ragsvc.corpus.generator import generate_corpus
from ragsvc.eval.harness import make_retriever


def main() -> None:
    cfg = Config.load(None)
    c = cfg.section("corpus")
    r = cfg.section("retrieval")
    ch = cfg.section("chunking")
    documents, _facts, questions = generate_corpus(
        int(c["n_services"]), int(c["sections_per_doc"]),
        float(c["hard_negative_rate"]), int(c["n_answerable"]),
        int(c["n_unanswerable"]), int(c["seed"]))
    chunks = build_chunks("fixed", documents, int(ch["fixed_size_words"]),
                          int(ch["fixed_overlap_words"]))
    texts = [chunk.text for chunk in chunks]
    queries = [q.text for q in questions]
    top_k = int(r["top_k"])

    results = {"environment": {"python": platform.python_version(),
                               "machine": platform.machine(),
                               "n_chunks": len(chunks),
                               "n_queries": len(queries)},
               "retrievers": {}}
    for name in ("bm25", "lsa", "hybrid"):
        index = make_retriever(name, int(r["lsa_dims"]), int(r["rrf_k"]),
                               int(c["seed"]))
        t0 = time.perf_counter()
        index.fit(texts)
        fit_seconds = time.perf_counter() - t0

        latencies = []
        t0 = time.perf_counter()
        for query in queries:
            q0 = time.perf_counter()
            index.top_k(query, top_k)
            latencies.append(time.perf_counter() - q0)
        wall = time.perf_counter() - t0
        latencies.sort()
        results["retrievers"][name] = {
            "fit_seconds": round(fit_seconds, 3),
            "queries_per_second": round(len(queries) / wall, 1),
            "p50_ms": round(statistics.median(latencies) * 1000, 2),
            "p95_ms": round(latencies[int(len(latencies) * 0.95)] * 1000, 2),
        }
        print(name, results["retrievers"][name])

    out = Path("benchmark/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
