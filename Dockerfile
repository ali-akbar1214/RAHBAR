FROM python:3.12-slim

WORKDIR /app

# Install pinned Python runtime dependencies
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn==0.30.6 \
    pydantic==2.9.2 \
    numpy==1.26.4 \
    pyyaml==6.0.2 \
    groq==0.15.0 \
    httpx==0.27.2 \
    requests==2.32.3

# Copy application source and seed corpus (never bake data/ or secrets)
COPY src/ /app/src/
COPY corpus/ /app/seed_corpus/
COPY eval/ /app/eval/
COPY drift/ /app/drift/
COPY scripts/ /app/scripts/
COPY traces/trace_tool.py /app/traces/trace_tool.py

# Create non-root user for Kubernetes securityContext (F-10)
RUN useradd -m -u 1000 ragops \
    && mkdir -p /data/lake/incoming /data/lake/raw /data/lake/quarantine /data/lake/archive /data/index /results /traces \
    && chown -R ragops:ragops /app /data /results /traces

USER 1000

ENV PYTHONPATH=/app
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV LAKE_DIR=/data/lake
ENV INDEX_DIR=/data/index
ENV RESULTS_DIR=/results
ENV TRACES_DIR=/traces
ENV COLLECTION_NAME=fast-policies-v1
ENV EMBEDDING_VERSION=lexical-proj-v1

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
