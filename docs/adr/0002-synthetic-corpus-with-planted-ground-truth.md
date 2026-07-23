# ADR 0002: Synthetic handbook corpus with planted ground truth

## Status
Accepted

## Context
Retrieval evaluation needs to know, for every question, exactly which
document and section contains the answer. Public QA datasets provide
that for their own domains and licenses; none provides it for the
enterprise-wiki failure mode this service targets: near-duplicate
pages that differ only in entities and values.

## Decision
Generate the corpus: one handbook page per internal service, an
identical section skeleton on every page, facts that differ only in
entities and values, shared boilerplate sentences, and value-free
cross references that inject other service names into every page.
Questions are generated from the facts with paraphrase templates, and
a held-out set asks about absent services and undocumented attributes
so abstention is measurable. Everything is seeded.

## Consequences
Positive: exact ground truth at section granularity, controllable
difficulty, measurable false-answer rate, zero licensing questions.
Negative: template-generated text is easier than real prose, so the
absolute numbers flatter every retriever; the grid's comparative
findings are the meaningful output. The first corpus version scored
recall@1 of 1.000 in six of nine cells and was fixed for difficulty
(commit history has the diff), which is itself evidence the harness
can catch a benchmark that has gone soft.
