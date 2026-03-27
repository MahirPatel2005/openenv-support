FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary __init__ files
RUN touch app/__init__.py tasks/__init__.py graders/__init__.py data/__init__.py

# HuggingFace Spaces runs as non-root user
RUN useradd -m -u 1000 user
USER user

EXPOSE 7860

ENV PYTHONPATH=/app
ENV PORT=7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
