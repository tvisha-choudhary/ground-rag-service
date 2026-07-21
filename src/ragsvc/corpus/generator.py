"""Synthetic internal engineering handbook with planted ground truth.

Why synthetic (docs/adr/0002): retrieval evaluation needs to know, for
every question, exactly which document and section contains the answer.
Public QA datasets provide that for their own domains; this generator
provides it for a controllable one, and lets us plant the failure mode
that makes enterprise retrieval hard: near-duplicate documents.

Structure: one handbook page per internal service. Every page has the
same section skeleton (overview, deployment, oncall, limits, security,
storage), which is exactly what makes company wikis hard to search:
every page about every service says "to deploy this service run". The
facts differ only in entities and values. A question about service A's
rate limit has dozens of near-duplicate distractors about B's and C's.

Ground truth: each fact sentence records its (doc_id, section_id). Each
answerable question targets one fact. Unanswerable questions ask about
planted-but-absent services and attributes, for measuring abstention.

Everything is seeded and deterministic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ADJECTIVES = ["amber", "birch", "cobalt", "delta", "ember", "flint", "granite",
              "harbor", "iris", "juniper", "krypton", "lumen", "meridian",
              "nimbus", "onyx", "pluto", "quartz", "raven", "sable", "tundra"]
NOUNS = ["gateway", "ledger", "indexer", "scheduler", "router", "vault",
         "courier", "beacon", "forge", "relay"]

REGIONS = ["us-east-1", "us-west-2", "eu-central-1", "ap-south-1"]
LANGUAGES = ["Python", "Go", "Java", "Rust"]
DATASTORES = ["PostgreSQL", "DynamoDB", "Cassandra", "MySQL"]
QUEUES = ["Kafka", "SQS", "Pub/Sub"]
ONCALL_ROTATIONS = ["weekly", "biweekly", "daily"]


@dataclass(frozen=True)
class Fact:
    doc_id: str
    section_id: str
    attribute: str
    value: str
    sentence: str


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    answerable: bool
    doc_id: str | None
    section_id: str | None
    answer: str | None


def _service_names(rng: np.random.Generator, n: int) -> list[str]:
    pairs = [(a, b) for a in ADJECTIVES for b in NOUNS]
    idx = rng.choice(len(pairs), size=n, replace=False)
    return [f"{pairs[i][0]}-{pairs[i][1]}" for i in idx]


def _facts_for_service(rng: np.random.Generator, name: str) -> list[Fact]:
    doc_id = f"doc-{name}"
    p95 = int(rng.choice([50, 80, 120, 200, 250, 400]))
    rate = int(rng.choice([100, 250, 500, 1000, 2000, 5000]))
    replicas = int(rng.integers(2, 13))
    retention = int(rng.choice([7, 14, 30, 90, 365]))
    port = int(rng.integers(7000, 9500))
    values = {
        "overview": (
            "language", str(rng.choice(LANGUAGES)),
            "{name} is an internal {noun} service written in {v}. "
            "It serves requests on port " + str(port) + "."),
        "deployment": (
            "region", str(rng.choice(REGIONS)),
            "To deploy {name}, run the standard release pipeline. "
            "The primary deployment region for {name} is {v} "
            "with " + str(replicas) + " replicas."),
        "oncall": (
            "rotation", str(rng.choice(ONCALL_ROTATIONS)),
            "The oncall rotation for {name} is {v}. Page the owning team "
            "through the standard escalation policy."),
        "limits": (
            "rate_limit", f"{rate} requests per second",
            "The rate limit for {name} is {v}. The p95 latency budget "
            "is " + str(p95) + " milliseconds."),
        "security": (
            "auth", str(rng.choice(["mTLS", "OAuth2 service tokens", "signed JWTs"])),
            "Authentication for {name} uses {v}. Rotate credentials with "
            "the standard secrets workflow."),
        "storage": (
            "datastore", str(rng.choice(DATASTORES)),
            "{name} persists state in {v} and publishes change events to "
            + str(rng.choice(QUEUES)) + ". Backups are retained for "
            + str(retention) + " days."),
    }
    noun = name.split("-")[1]
    facts = []
    for section_id, (attribute, value, template) in values.items():
        sentence = template.format(name=name, v=value, noun=noun)
        facts.append(Fact(doc_id=doc_id, section_id=section_id,
                          attribute=attribute, value=value, sentence=sentence))
    return facts


BOILERPLATE = {
    "overview": ["This page follows the standard handbook template.",
                 "Ownership and escalation details live in the service catalog.",
                 "Architecture diagrams are attached to the design review."],
    "deployment": ["Deployments follow the standard release pipeline with canary "
                   "analysis before full rollout.",
                   "Rollbacks use the previous immutable image tag.",
                   "Config changes ship through the same pipeline as code."],
    "oncall": ["Runbooks are linked from every alert.",
               "Sev1 incidents page the incident commander rotation.",
               "Alert thresholds are reviewed after every incident."],
    "limits": ["Load tests run before every capacity change.",
               "Clients should implement exponential backoff on 429 responses.",
               "Quota increases require a capacity review."],
    "security": ["Access follows least privilege and quarterly review.",
                 "Secrets never appear in logs or environment dumps.",
                 "Dependency scanning runs in the build pipeline."],
    "storage": ["Schema changes require a migration review.",
                "Restore drills run quarterly against the latest backup.",
                "Data classification is documented in the retention policy."],
}

CROSS_REFS = [
    "Requests typically arrive via {other} before reaching {name}.",
    "Coordinate maintenance windows with the {other} team.",
    "{name} sits upstream of {other} in the request path.",
    "Dashboards for {name} and {other} share the same folder.",
]


def _section_text(rng: np.random.Generator, name: str, fact_sentence: str,
                  section_id: str, other_names: list[str]) -> str:
    """A realistic section: the fact, surrounded by shared boilerplate and
    cross references that mention other services by name. The cross
    references carry no attribute values, so ground truth stays unique,
    but they inject other entities into every page, which is what makes
    entity matching insufficient and retrieval genuinely hard."""
    extras = list(rng.choice(BOILERPLATE[section_id],
                             size=min(2, len(BOILERPLATE[section_id])), replace=False))
    others = rng.choice(len(other_names), size=2, replace=False)
    cross = str(rng.choice(CROSS_REFS)).format(
        name=name, other=other_names[int(others[0])])
    sentences = [fact_sentence] + extras + [cross]
    order = rng.permutation(len(sentences))
    # keep the fact sentence findable anywhere in the section, not always first
    return " ".join(sentences[i] for i in order)


QUESTION_TEMPLATES = {
    "language": ["What language is {name} written in?",
                 "Which programming language powers {name}?",
                 "What is {name} implemented in?"],
    "region": ["What is the primary deployment region for {name}?",
               "Which cloud region hosts {name}?",
               "Where does {name} primarily run?"],
    "rotation": ["How often does the oncall rotation for {name} change?",
                 "What is the oncall cadence for {name}?",
                 "How frequently does {name} rotate its pager duty?"],
    "rate_limit": ["What is the rate limit for {name}?",
                   "How many requests per second can {name} accept?",
                   "What throughput is {name} capped at?"],
    "auth": ["How do clients authenticate to {name}?",
             "What auth mechanism does {name} require?",
             "How is access to {name} verified?"],
    "datastore": ["Where does {name} store its state?",
                  "Which database backs {name}?",
                  "What does {name} persist its data in?"],
}

UNANSWERABLE_ATTRS = [
    "What is the disaster recovery objective for {name}?",
    "Who reviewed the launch of {name}?",
    "What is the monthly cloud cost of {name}?",
    "Which CDN fronts {name}?",
    "What is the GDPR data residency plan for {name}?",
]


def generate_corpus(n_services: int, sections_per_doc: int, hard_negative_rate: float,
                    n_answerable: int, n_unanswerable: int, seed: int
                    ) -> tuple[list[dict], list[Fact], list[Question]]:
    """Return (documents, facts, questions).

    Documents are dicts: {doc_id, title, sections: [{section_id, text}]}.
    hard_negative_rate controls how many extra services get generated
    purely to serve as distractors (their facts exist in the corpus but
    no questions target them).
    """
    if not 0 <= hard_negative_rate < 1:
        raise ValueError("hard_negative_rate must be in [0, 1)")
    rng = np.random.default_rng(seed)
    n_distractors = int(n_services * hard_negative_rate)
    names = _service_names(rng, n_services + n_distractors)
    target_names, distractor_names = names[:n_services], names[n_services:]

    documents, all_facts = [], []
    for name in names:
        facts = _facts_for_service(rng, name)[:sections_per_doc]
        all_facts.extend(facts)
        other_names = [n for n in names if n != name]
        documents.append({
            "doc_id": f"doc-{name}",
            "title": f"{name} service handbook",
            "sections": [{"section_id": f.section_id,
                          "text": _section_text(rng, name, f.sentence,
                                                f.section_id, other_names)}
                         for f in facts],
        })

    fact_by_key = {(f.doc_id, f.attribute): f for f in all_facts}
    questions: list[Question] = []
    qid = 0
    while len([q for q in questions if q.answerable]) < n_answerable:
        name = str(rng.choice(target_names))
        attribute = str(rng.choice(list(QUESTION_TEMPLATES)))
        fact = fact_by_key.get((f"doc-{name}", attribute))
        if fact is None:
            continue
        template = str(rng.choice(QUESTION_TEMPLATES[attribute]))
        questions.append(Question(
            question_id=f"q{qid:04d}",
            text=template.format(name=name),
            answerable=True, doc_id=fact.doc_id, section_id=fact.section_id,
            answer=fact.value))
        qid += 1

    absent = _service_names(np.random.default_rng(seed + 1), 30)
    absent = [a for a in absent if a not in set(names)][:15]
    for _ in range(n_unanswerable):
        if rng.random() < 0.5 and absent:
            name = str(rng.choice(absent))       # service does not exist
            attribute = str(rng.choice(list(QUESTION_TEMPLATES)))
            template = str(rng.choice(QUESTION_TEMPLATES[attribute]))
        else:
            name = str(rng.choice(target_names))  # attribute never documented
            template = str(rng.choice(UNANSWERABLE_ATTRS))
        questions.append(Question(
            question_id=f"q{qid:04d}", text=template.format(name=name),
            answerable=False, doc_id=None, section_id=None, answer=None))
        qid += 1

    logger.info("corpus: %d docs (%d distractor), %d facts, %d questions",
                len(documents), len(distractor_names), len(all_facts), len(questions))
    return documents, all_facts, questions


def write_corpus(documents: list[dict], facts: list[Fact], questions: list[Question],
                 output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "documents.json").write_text(json.dumps(documents, indent=1))
    (out / "facts.json").write_text(json.dumps([asdict(f) for f in facts], indent=1))
    (out / "questions.json").write_text(json.dumps([asdict(q) for q in questions], indent=1))
    return out
