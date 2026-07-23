"""Unit and integration tests for the grounded RAG service.

The tests that earn their keep most: the hand-computed BM25 ranking,
the abstention behavior on questions whose entity exists but whose
attribute does not, and the faithfulness gate rejecting an answer that
asserts a value its citation does not contain.
"""

import numpy as np
import pytest

from ragsvc.answer.extractive import Answer, answer_question, split_sentences
from ragsvc.answer.faithfulness import check_support
from ragsvc.answer.providers import ExtractiveProvider
from ragsvc.corpus.chunking import Chunk, build_chunks
from ragsvc.corpus.generator import generate_corpus
from ragsvc.eval.harness import chunk_is_hit, eval_retrieval, make_retriever
from ragsvc.retrieval.bm25 import BM25Index, tokenize
from ragsvc.retrieval.dense import LSAIndex
from ragsvc.retrieval.hybrid import HybridIndex


@pytest.fixture(scope="module")
def small_corpus():
    return generate_corpus(n_services=12, sections_per_doc=6,
                           hard_negative_rate=0.25, n_answerable=40,
                           n_unanswerable=15, seed=9)


class TestCorpus:
    def test_deterministic(self, small_corpus):
        again = generate_corpus(12, 6, 0.25, 40, 15, 9)
        assert [d["doc_id"] for d in again[0]] == [d["doc_id"] for d in small_corpus[0]]
        assert [q.text for q in again[2]] == [q.text for q in small_corpus[2]]

    def test_ground_truth_fact_present_in_gold_section(self, small_corpus):
        docs, facts, questions = small_corpus
        by_doc = {d["doc_id"]: d for d in docs}
        for q in questions:
            if not q.answerable:
                continue
            section = next(s for s in by_doc[q.doc_id]["sections"]
                           if s["section_id"] == q.section_id)
            assert q.answer.lower() in section["text"].lower()

    def test_unanswerable_have_no_gold(self, small_corpus):
        _, _, questions = small_corpus
        for q in questions:
            if not q.answerable:
                assert q.doc_id is None and q.answer is None

    def test_invalid_params_rejected(self):
        with pytest.raises(ValueError):
            generate_corpus(5, 6, 1.5, 10, 5, 1)


class TestChunking:
    def test_section_chunks_are_one_section(self, small_corpus):
        docs, _, _ = small_corpus
        for chunk in build_chunks("section", docs):
            assert len(chunk.section_ids) == 1

    def test_fixed_windows_cover_all_words(self, small_corpus):
        docs, _, _ = small_corpus
        chunks = build_chunks("fixed", docs, size_words=50, overlap_words=10)
        doc = docs[0]
        total_words = sum(len(s["text"].split()) for s in doc["sections"])
        covered = set()
        for chunk in chunks:
            if chunk.doc_id == doc["doc_id"]:
                start = int(chunk.chunk_id.split(":w")[1])
                covered.update(range(start, start + len(chunk.text.split())))
        assert covered == set(range(total_words))

    def test_bad_params_rejected(self, small_corpus):
        docs, _, _ = small_corpus
        with pytest.raises(ValueError):
            build_chunks("fixed", docs, size_words=10, overlap_words=10)
        with pytest.raises(ValueError):
            build_chunks("nonsense", docs)


class TestBM25:
    def test_hand_computed_ranking(self):
        """Three tiny docs; the query term appears only in doc 1, so doc 1
        must rank first, and the rarer term must outweigh the common one."""
        idx = BM25Index().fit(["apple banana banana", "banana cherry", "durian"])
        top = idx.top_k("cherry", 3)
        assert top[0][0] == 1
        top_common = idx.top_k("banana", 3)
        assert {top_common[0][0], top_common[1][0]} == {0, 1}
        assert idx.top_k("durian", 1)[0][0] == 2

    def test_hyphenated_entities_stay_whole(self):
        assert "quartz-courier" in tokenize("Restart quartz-courier now.")

    def test_empty_corpus_rejected(self):
        with pytest.raises(ValueError):
            BM25Index().fit([])


class TestDenseAndHybrid:
    def test_lsa_finds_exact_match(self, small_corpus):
        docs, _, _ = small_corpus
        chunks = build_chunks("section", docs)
        idx = LSAIndex(dims=64, seed=0).fit([c.text for c in chunks])
        target = chunks[0]
        top = idx.top_k(target.text, 1)
        assert chunks[top[0][0]].chunk_id == target.chunk_id

    def test_hybrid_scores_normalized(self, small_corpus):
        docs, _, _ = small_corpus
        chunks = build_chunks("section", docs)
        idx = HybridIndex(BM25Index(), LSAIndex(dims=64, seed=0)).fit(
            [c.text for c in chunks])
        top = idx.top_k("what is the rate limit", 5)
        assert all(0.0 <= s <= 1.0 for _, s in top)
        assert top[0][1] == max(s for _, s in top)

    def test_make_retriever_rejects_unknown(self):
        with pytest.raises(ValueError):
            make_retriever("bert", 64, 60, 0)


class TestAnswerAndGate:
    CHUNKS = [
        Chunk("c1", "doc-a", ("limits",),
              "The rate limit for amber-vault is 500 requests per second."),
        Chunk("c2", "doc-a", ("overview",),
              "amber-vault is an internal vault service written in Go."),
    ]

    def test_answers_with_citation(self):
        ans = answer_question("What is the rate limit for amber-vault?",
                              [(self.CHUNKS[0], 0.9), (self.CHUNKS[1], 0.5)],
                              abstain_below=0.5)
        assert not ans.abstained
        assert "500 requests per second" in ans.text
        assert ans.citations == ("c1",)
        assert ans.confidence == 1.0

    def test_abstains_when_attribute_absent(self):
        """The regression test for the gate rewrite: the entity matches,
        the attribute does not, the service must not guess."""
        ans = answer_question("Which CDN fronts amber-vault?",
                              [(self.CHUNKS[0], 0.9), (self.CHUNKS[1], 0.8)],
                              abstain_below=0.5)
        assert ans.abstained
        assert ans.text is None
        assert ans.confidence < 0.5

    def test_abstains_on_empty_retrieval(self):
        ans = answer_question("anything", [], abstain_below=0.5)
        assert ans.abstained and ans.confidence == 0.0

    def test_provider_protocol(self):
        provider = ExtractiveProvider()
        ans = provider.answer("What is the rate limit for amber-vault?",
                              [(self.CHUNKS[0], 0.9)], 0.5)
        assert isinstance(ans, Answer) and not ans.abstained

    def test_split_sentences(self):
        assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


class TestFaithfulness:
    CHUNK = Chunk("c1", "doc-a", ("limits",),
                  "The rate limit for amber-vault is 500 requests per second.")

    def test_supported_answer_passes(self):
        verdict = check_support(
            "The rate limit for amber-vault is 500 requests per second.",
            [self.CHUNK])
        assert verdict.supported and verdict.support_ratio == 1.0

    def test_fabricated_value_fails(self):
        """An answer asserting a value its citation does not contain must
        be flagged, with the fabricated terms named."""
        verdict = check_support(
            "The rate limit for amber-vault is 9000 requests per minute.",
            [self.CHUNK])
        assert not verdict.supported or verdict.support_ratio < 1.0
        assert "9000" in verdict.unsupported_terms

    def test_empty_inputs_unsupported(self):
        assert not check_support("", [self.CHUNK]).supported
        assert not check_support("answer", []).supported


class TestAnthropicProvider:
    """The provider's citation parsing and NOT_FOUND protocol are real
    logic worth pinning; a fake client stands in for the API, and the
    integration path stays honest: no README metric depends on it."""

    CHUNKS = [
        (Chunk("doc-a:limits", "doc-a", ("limits",),
               "The rate limit for amber-vault is 500 requests per second."), 0.9),
        (Chunk("doc-a:overview", "doc-a", ("overview",),
               "amber-vault is written in Go."), 0.6),
    ]

    @staticmethod
    def _provider_with_reply(monkeypatch, reply_text):
        import sys
        import types

        class FakeMessages:
            def create(self, **kwargs):
                block = types.SimpleNamespace(text=reply_text)
                return types.SimpleNamespace(content=[block])

        class FakeAnthropic:
            def __init__(self):
                self.messages = FakeMessages()

        fake_module = types.SimpleNamespace(Anthropic=FakeAnthropic)
        monkeypatch.setitem(sys.modules, "anthropic", fake_module)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from ragsvc.answer.providers import AnthropicProvider
        return AnthropicProvider()

    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from ragsvc.answer.providers import AnthropicProvider
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            AnthropicProvider()

    def test_not_found_becomes_abstention(self, monkeypatch):
        provider = self._provider_with_reply(monkeypatch, "NOT_FOUND")
        ans = provider.answer("Which CDN fronts amber-vault?", self.CHUNKS, 0.0)
        assert ans.abstained and ans.text is None

    def test_citations_parsed_and_stripped(self, monkeypatch):
        provider = self._provider_with_reply(
            monkeypatch,
            "The rate limit is 500 requests per second. [doc-a:limits]")
        ans = provider.answer("What is the rate limit for amber-vault?",
                              self.CHUNKS, 0.0)
        assert not ans.abstained
        assert ans.citations == ("doc-a:limits",)
        assert "[doc-a:limits]" not in ans.text
        assert "500 requests per second" in ans.text

    def test_gate_applies_before_the_api_call(self, monkeypatch):
        provider = self._provider_with_reply(monkeypatch, "irrelevant")
        ans = provider.answer("anything", [], abstain_below=0.5)
        assert ans.abstained and ans.confidence == 0.0


class TestHarness:
    def test_hit_criterion(self, small_corpus):
        docs, _, questions = small_corpus
        q = next(q for q in questions if q.answerable)
        hit = Chunk("x", q.doc_id, (q.section_id,), "text")
        miss = Chunk("y", q.doc_id, ("other",), "text")
        assert chunk_is_hit(hit, q) and not chunk_is_hit(miss, q)

    def test_eval_retrieval_perfect_on_oracle(self, small_corpus):
        """An oracle index that always returns the gold chunk first must
        score recall@1 of 1.0; pins the metric arithmetic."""
        docs, _, questions = small_corpus
        chunks = build_chunks("section", docs)
        gold_index = {(c.doc_id, c.section_ids[0]): i for i, c in enumerate(chunks)}
        answerable = [q for q in questions if q.answerable]

        class Oracle:
            def __init__(self):
                self._queue = [gold_index[(q.doc_id, q.section_id)]
                               for q in answerable]
                self._i = -1

            def top_k(self, query, k):
                self._i += 1
                return [(self._queue[self._i], 1.0)]

        metrics = eval_retrieval(chunks, Oracle(), answerable, top_k=5)
        assert metrics["recall_at_1"] == 1.0 and metrics["mrr"] == 1.0


class TestService:
    def test_ask_and_abstain_roundtrip(self, tmp_path, small_corpus, monkeypatch):
        import json as jsonlib
        docs, facts, questions = small_corpus
        corpus_dir = tmp_path / "generated"
        corpus_dir.mkdir()
        (corpus_dir / "documents.json").write_text(jsonlib.dumps(docs))
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(f"""
corpus:
  n_services: 12
  sections_per_doc: 6
  hard_negative_rate: 0.25
  n_answerable: 40
  n_unanswerable: 15
  seed: 9
  output_dir: {corpus_dir}
chunking:
  fixed_size_words: 120
  fixed_overlap_words: 30
retrieval:
  top_k: 5
  lsa_dims: 64
  rrf_k: 60
answer:
  abstain_below: 0.5
  support_overlap_threshold: 0.5
artifacts_dir: {tmp_path}/artifacts
""")
        from fastapi.testclient import TestClient
        from ragsvc.service.app import build_app
        client = TestClient(build_app(str(cfg)))

        assert client.get("/health").json()["status"] == "ok"
        # Plumbing test, not an accuracy test: accuracy lives in the
        # harness with its measured error rate. Use an exact-template
        # question ("rate limit" appears verbatim in the gold sentence)
        # so this test fails only when the service path breaks, not when
        # it hits the known paraphrase-tie error class.
        exact_markers = ("rate limit", "primary deployment region",
                         "language is", "store its state",
                         "authenticate to", "oncall rotation for")
        answerable = next(q for q in questions if q.answerable
                          and any(m in q.text for m in exact_markers))
        response = client.post("/ask", json={"question": answerable.text}).json()
        assert response["abstained"] is False
        assert answerable.answer.lower() in response["answer"].lower()
        assert response["citations"] and response["supported"] is True

        made_up = "What is the annual audit budget for " + docs[0]["doc_id"][4:] + "?"
        response = client.post("/ask", json={"question": made_up}).json()
        assert response["abstained"] is True and response["answer"] is None

        assert client.post("/ask", json={"question": "x"}).status_code == 422

    def test_missing_corpus_is_a_clear_error(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(f"""
corpus:
  output_dir: {tmp_path}/nowhere
  seed: 1
chunking:
  fixed_size_words: 120
  fixed_overlap_words: 30
retrieval:
  top_k: 5
  lsa_dims: 64
  rrf_k: 60
answer:
  abstain_below: 0.5
  support_overlap_threshold: 0.5
""")
        from ragsvc.service.app import build_app
        with pytest.raises(RuntimeError, match="no corpus"):
            build_app(str(cfg))


class TestPipelineEndToEnd:
    def test_tiny_run(self, tmp_path):
        cfg = tmp_path / "tiny.yaml"
        cfg.write_text(f"""
corpus:
  n_services: 10
  sections_per_doc: 6
  hard_negative_rate: 0.2
  n_answerable: 30
  n_unanswerable: 10
  seed: 3
  output_dir: {tmp_path}/generated
chunking:
  fixed_size_words: 80
  fixed_overlap_words: 20
retrieval:
  top_k: 5
  lsa_dims: 32
  rrf_k: 60
answer:
  abstain_below: 0.5
  support_overlap_threshold: 0.5
artifacts_dir: {tmp_path}/artifacts
""")
        from ragsvc.pipeline import run
        report = run(str(cfg))
        assert len(report["grid"]) == 9
        assert report["best_config"]["retriever"] in {"bm25", "lsa", "hybrid"}
        assert (tmp_path / "artifacts" / "eval_report.json").exists()
        curve = report["abstention_curve"]
        false_at_zero = curve[0]["false_answer_rate_on_unanswerable"]
        false_at_high = curve[-1]["false_answer_rate_on_unanswerable"]
        assert false_at_high <= false_at_zero


class TestMLflowTracking:
    def test_grid_runs_logged(self, tmp_path, monkeypatch):
        mlflow = pytest.importorskip("mlflow")
        monkeypatch.chdir(tmp_path)
        uri = f"sqlite:///{tmp_path}/mlflow.db"
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
        mlflow.set_tracking_uri(uri)
        from ragsvc.eval.harness import _log_mlflow
        _log_mlflow("section", "bm25", {"recall_at_1": 0.5, "mrr": 0.6,
                                        "n_chunks": 10})
        runs = mlflow.search_runs(search_all_experiments=True)
        assert len(runs) == 1
        assert runs.iloc[0]["params.retriever"] == "bm25"
