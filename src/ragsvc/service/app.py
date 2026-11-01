"""FastAPI service: retrieval, grounded answer, citations, abstention.

Startup builds the index from the generated corpus using the serving
config (fixed chunking, hybrid retrieval, the grid's selected cell).
Responses always carry citations and a confidence; when the coverage
gate abstains, the service says so honestly instead of guessing, with
HTTP 200 and abstained=true, because "not in the handbook" is a valid
answer, not a server error.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ragsvc.answer.faithfulness import check_support
from ragsvc.answer.providers import ExtractiveProvider
from ragsvc.config import Config
from ragsvc.corpus.chunking import build_chunks
from ragsvc.logging_config import configure_logging
from ragsvc.retrieval.bm25 import BM25Index
from ragsvc.retrieval.dense import LSAIndex
from ragsvc.retrieval.hybrid import HybridIndex

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)


class AskResponse(BaseModel):
    answer: str | None
    abstained: bool
    confidence: float
    citations: list[str]
    supported: bool | None


def build_app(config_path: str | None = None) -> FastAPI:
    configure_logging()
    cfg = Config.load(config_path)
    r = cfg.section("retrieval")
    a = cfg.section("answer")
    c = cfg.section("chunking")
    corpus_dir = Path(str(cfg.section("corpus")["output_dir"]))
    docs_path = corpus_dir / "documents.json"
    if not docs_path.exists():
        raise RuntimeError(
            f"no corpus at {docs_path}; run `python -m ragsvc.pipeline` first")
    documents = json.loads(docs_path.read_text())

    chunks = build_chunks("fixed", documents, int(c["fixed_size_words"]),
                          int(c["fixed_overlap_words"]))
    index = HybridIndex(BM25Index(),
                        LSAIndex(dims=int(r["lsa_dims"]),
                                 seed=int(cfg.section("corpus")["seed"])),
                        rrf_k=int(r["rrf_k"]))
    index.fit([ch.text for ch in chunks])
    by_id = {ch.chunk_id: ch for ch in chunks}
    provider = ExtractiveProvider()
    top_k = int(r["top_k"])
    abstain_below = float(a["abstain_below"])
    support_threshold = float(a["support_overlap_threshold"])

    app = FastAPI(title="grounded-rag-service", version="0.1.0")

    ui_path = Path(__file__).parent / "index.html"

    @app.get("/", response_class=HTMLResponse)
    def ui() -> str:
        if ui_path.exists():
            return ui_path.read_text(encoding="utf-8")
        return "<h1>Grounded RAG Service</h1><p>Visit <a href='/docs'>/docs</a></p>"

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "n_chunks": len(chunks)}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            ranked = [(by_id[chunks[i].chunk_id], s)
                      for i, s in index.top_k(request.question, top_k)]
            answer = provider.answer(request.question, ranked, abstain_below)
        except Exception:
            logger.exception("answer pipeline failed")
            raise HTTPException(status_code=500, detail="internal error") from None
        supported = None
        if not answer.abstained and answer.text:
            verdict = check_support(answer.text,
                                    [by_id[cid] for cid in answer.citations],
                                    support_threshold)
            supported = verdict.supported
        logger.info("ask", extra={"extra_fields": {
            "abstained": answer.abstained, "confidence": answer.confidence,
            "n_citations": len(answer.citations)}})
        return AskResponse(answer=answer.text, abstained=answer.abstained,
                           confidence=answer.confidence,
                           citations=list(answer.citations), supported=supported)

    return app


def create_app() -> FastAPI:
    """uvicorn entry point: uvicorn ragsvc.service.app:create_app --factory"""
    return build_app(None)
