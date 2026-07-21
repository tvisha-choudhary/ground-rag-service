"""The evaluation harness: every README number comes from here.

Three measurements:

1. Retrieval grid: {fixed, section, document} chunking x {bm25, lsa,
   hybrid} retrievers, scored with recall@1, recall@5, and MRR against
   the planted (doc_id, section_id) ground truth. A hit means the
   retrieved chunk contains the gold section of the gold document.
2. Answer quality on answerable questions: exact substring match of the
   gold value in the produced answer, plus faithfulness of citations.
3. The abstention curve: sweep the gate threshold and measure coverage
   (share of answerable questions answered) against the false answer
   rate (share of unanswerable questions that got an answer anyway).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ragsvc.answer.extractive import answer_question
from ragsvc.answer.faithfulness import check_support
from ragsvc.corpus.chunking import Chunk, build_chunks
from ragsvc.corpus.generator import Question
from ragsvc.retrieval.bm25 import BM25Index
from ragsvc.retrieval.dense import LSAIndex
from ragsvc.retrieval.hybrid import HybridIndex

logger = logging.getLogger(__name__)

CHUNKINGS = ("fixed", "section", "document")
RETRIEVERS = ("bm25", "lsa", "hybrid")


def make_retriever(name: str, lsa_dims: int, rrf_k: int, seed: int):
    if name == "bm25":
        return BM25Index()
    if name == "lsa":
        return LSAIndex(dims=lsa_dims, seed=seed)
    if name == "hybrid":
        return HybridIndex(BM25Index(), LSAIndex(dims=lsa_dims, seed=seed), rrf_k=rrf_k)
    raise ValueError(f"unknown retriever: {name}")


def chunk_is_hit(chunk: Chunk, question: Question) -> bool:
    return chunk.doc_id == question.doc_id and question.section_id in chunk.section_ids


def eval_retrieval(chunks: list[Chunk], index, questions: list[Question],
                   top_k: int) -> dict:
    answerable = [q for q in questions if q.answerable]
    r1 = r5 = 0
    mrr = 0.0
    for q in answerable:
        ranked = index.top_k(q.text, top_k)
        hits = [rank for rank, (i, _s) in enumerate(ranked, start=1)
                if chunk_is_hit(chunks[i], q)]
        if hits:
            first = hits[0]
            mrr += 1.0 / first
            r1 += int(first == 1)
            r5 += int(first <= top_k)
    n = len(answerable)
    return {"recall_at_1": round(r1 / n, 4), f"recall_at_{top_k}": round(r5 / n, 4),
            "mrr": round(mrr / n, 4), "n_questions": n}


def _ranked_chunks(index, chunks, query, top_k):
    """Retrieval scores are passed through untouched: since the gate
    moved to answer-term coverage they only break ties between equally
    covering sentences, so no cross-retriever calibration is needed."""
    return [(chunks[i], s) for i, s in index.top_k(query, top_k)]


def eval_answers(chunks: list[Chunk], index, questions: list[Question], top_k: int,
                 abstain_below: float, support_threshold: float) -> dict:
    by_id = {c.chunk_id: c for c in chunks}
    answered = correct = faithful = 0
    false_answers = n_unanswerable = 0
    for q in questions:
        ranked = _ranked_chunks(index, chunks, q.text, top_k)
        ans = answer_question(q.text, ranked, abstain_below)
        if q.answerable:
            if not ans.abstained:
                answered += 1
                if q.answer and q.answer.lower() in (ans.text or "").lower():
                    correct += 1
                verdict = check_support(ans.text or "",
                                        [by_id[c] for c in ans.citations],
                                        support_threshold)
                faithful += int(verdict.supported)
        else:
            n_unanswerable += 1
            if not ans.abstained:
                false_answers += 1
    n_ans = sum(q.answerable for q in questions)
    return {
        "coverage": round(answered / n_ans, 4),
        "answer_accuracy_when_answered": round(correct / max(answered, 1), 4),
        "faithfulness_when_answered": round(faithful / max(answered, 1), 4),
        "false_answer_rate_on_unanswerable": round(false_answers / max(n_unanswerable, 1), 4),
    }


def abstention_curve(chunks: list[Chunk], index, questions: list[Question],
                     top_k: int, support_threshold: float,
                     thresholds: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4,
                                                      0.5, 0.6, 0.7, 0.8)) -> list[dict]:
    curve = []
    for t in thresholds:
        point = eval_answers(chunks, index, questions, top_k, t, support_threshold)
        point["threshold"] = t
        curve.append(point)
    return curve


def run_harness(documents: list[dict], questions: list[Question], cfg: dict,
                mlflow_tracking: bool = False) -> dict:
    r = cfg["retrieval"]
    a = cfg["answer"]
    c = cfg["chunking"]
    seed = int(cfg["corpus"]["seed"])
    top_k = int(r["top_k"])

    report: dict = {"grid": {}, "answers": {}, "abstention_curve": []}
    timings: dict[str, float] = {}
    best = None
    for chunking in CHUNKINGS:
        chunks = build_chunks(chunking, documents,
                              int(c["fixed_size_words"]), int(c["fixed_overlap_words"]))
        texts = [ch.text for ch in chunks]
        for retriever in RETRIEVERS:
            index = make_retriever(retriever, int(r["lsa_dims"]), int(r["rrf_k"]), seed)
            t0 = time.perf_counter()
            index.fit(texts)
            timings[f"fit_{chunking}_{retriever}"] = round(time.perf_counter() - t0, 3)
            metrics = eval_retrieval(chunks, index, questions, top_k)
            metrics["n_chunks"] = len(chunks)
            end_task = eval_answers(chunks, index, questions, top_k,
                                    float(a["abstain_below"]),
                                    float(a["support_overlap_threshold"]))
            metrics["answer_accuracy"] = end_task["answer_accuracy_when_answered"]
            metrics["coverage"] = end_task["coverage"]
            report["grid"][f"{chunking}/{retriever}"] = metrics
            # Select the serving config on the end task, not on retrieval
            # metrics: document-level chunking scores perfect recall@1 on
            # a small corpus while producing the worst answers, because a
            # whole page is a weak citation and a noisy extraction target.
            key = (end_task["answer_accuracy_when_answered"], metrics["mrr"])
            if best is None or key > best[0]:
                best = (key, chunking, retriever, chunks, index)
            if mlflow_tracking:
                _log_mlflow(chunking, retriever, metrics)

    _key, chunking, retriever, chunks, index = best
    report["best_config"] = {"chunking": chunking, "retriever": retriever}
    report["answers"] = eval_answers(chunks, index, questions, top_k,
                                     float(a["abstain_below"]),
                                     float(a["support_overlap_threshold"]))
    report["abstention_curve"] = abstention_curve(
        chunks, index, questions, top_k, float(a["support_overlap_threshold"]))
    report["fit_timings_seconds"] = timings
    logger.info("harness complete", extra={"extra_fields": report["best_config"]})
    return report


def _log_mlflow(chunking: str, retriever: str, metrics: dict) -> None:
    import mlflow

    with mlflow.start_run(run_name=f"{chunking}-{retriever}"):
        mlflow.log_params({"chunking": chunking, "retriever": retriever})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})


def write_report(report: dict, artifacts_dir: str | Path) -> Path:
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "eval_report.json"
    path.write_text(json.dumps(report, indent=2))
    return path
