# ADR 0001: From-scratch BM25 and LSA instead of downloaded embedding models

## Status
Accepted

## Context
The build and CI environments download no pretrained models. A RAG
repo whose dense retriever silently pulls a sentence-transformer from
a hub cannot run its own evaluation in CI, and its numbers cannot be
reproduced by a reviewer without network access to that hub.

## Decision
The lexical leg is Okapi BM25 implemented from scratch (about sixty
lines, textbook k1 and b), so every scoring decision traces to
arithmetic in this repo. The dense leg is TF-IDF plus truncated SVD
(latent semantic analysis) learned from the corpus itself. Fusion is
reciprocal rank fusion, which needs no score calibration between legs.
The retriever interface (`fit`, `top_k`) is where a real embedding
model plugs in unchanged; the harness measures whatever sits behind it.

## Consequences
Positive: the entire evaluation grid runs offline, in CI, in seconds,
and every README number is reproducible from a clean clone.
Negative: LSA is not a modern embedding model and the measured
lexical-vs-dense comparison must be read as "BM25 vs a classical dense
baseline", which the README says plainly. On this entity-anchored
corpus the lexical leg dominates and hybrid fusion does not beat BM25;
that finding is reported, not hidden.
