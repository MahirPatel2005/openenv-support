FROM python:3.11-slim

WORKDIR /app

# Install system dependencies + uv
RUN apt-get update && apt-get install -y \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"

# Copy project files
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv (falls back to pip if uv.lock missing)
RUN uv pip install --system -r pyproject.toml 2>/dev/null || \
    pip install --no-cache-dir \
        openenv-core \
        fastapi \
        "uvicorn[standard]" \
        pydantic \
        httpx \
        openai \
        python-multipart \
        pyyaml \
        datasets \
        huggingface-hub \
        sentence-transformers

# Copy application code
COPY . .

# Create __init__ files
RUN touch app/__init__.py tasks/__init__.py graders/__init__.py data/__init__.py server/__init__.py

# Pre-download sentence-transformers model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

# HuggingFace Spaces runs as non-root user
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 7860

ENV PYTHONPATH=/app
ENV PORT=7860

# Use the server entry point as required by openenv validate
CMD ["python", "-m", "server.app"]