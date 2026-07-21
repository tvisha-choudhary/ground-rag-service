"""End to end: generate the corpus, run the evaluation harness, write artifacts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ragsvc.config import Config
from ragsvc.corpus.generator import generate_corpus, write_corpus
from ragsvc.eval.harness import run_harness, write_report
from ragsvc.logging_config import configure_logging

logger = logging.getLogger(__name__)


def run(config_path: str | None = None, mlflow_tracking: bool = False) -> dict:
    cfg = Config.load(config_path)
    c = cfg.section("corpus")
    documents, facts, questions = generate_corpus(
        n_services=int(c["n_services"]), sections_per_doc=int(c["sections_per_doc"]),
        hard_negative_rate=float(c["hard_negative_rate"]),
        n_answerable=int(c["n_answerable"]), n_unanswerable=int(c["n_unanswerable"]),
        seed=int(c["seed"]))
    write_corpus(documents, facts, questions, c["output_dir"])

    report = run_harness(documents, questions, cfg.raw, mlflow_tracking)
    artifacts = Path(str(cfg.raw.get("artifacts_dir", "artifacts")))
    write_report(report, artifacts)
    logger.info("pipeline complete", extra={"extra_fields": {"artifacts": str(artifacts)}})
    return report


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mlflow", action="store_true", help="log grid runs to MLflow")
    args = parser.parse_args()
    run(args.config, args.mlflow)


if __name__ == "__main__":
    main()
