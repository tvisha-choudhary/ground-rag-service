# ADR 0003: Boring choices, recorded

- Extractive answering as the default provider: it cannot hallucinate
  by construction, which calibrates the faithfulness gate, and it runs
  offline. LLM providers plug in behind the same protocol and must
  clear the same gate.
- FastAPI over heavier frameworks: one POST endpoint and a healthcheck
  do not need more, and the pydantic request model gives input
  validation (length bounds return 422) for free.
- Lexical support checking for faithfulness instead of an NLI model:
  runs in microseconds on every response with no model downloads. Its
  known blind spots are documented in the README's honest boundaries.
- JSON structured logs from day one, same as a production service.
