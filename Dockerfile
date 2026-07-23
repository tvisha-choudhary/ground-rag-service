FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config

RUN pip install --no-cache-dir -e .

# Bake the corpus and evaluation artifacts at build time so the
# container serves immediately; the build is deterministic (seeded).
RUN python -m ragsvc.pipeline

EXPOSE 8000
CMD ["uvicorn", "ragsvc.service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
