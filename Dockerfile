FROM python:3.11-slim

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml /app/
COPY README.md /app/
COPY data/ /app/data/
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY run_pipeline.py /app/
COPY query.py /app/
COPY server.py /app/
COPY main.py /app/
COPY ui/ /app/ui/
COPY scripts/ /app/scripts/

# Install python dependencies exactly
RUN pip install --no-cache-dir -e .

# Pre-cache neural embedder so container runs self-contained without runtime downloads
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')"

# Command to execute end-to-end pipeline
CMD ["python", "run_pipeline.py"]
