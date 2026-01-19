# Use an official lightweight Python image.
FROM python:3.10-slim

# Set environment variables to ensure Python output is sent straight to terminal
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=5000

# Set working directory
WORKDIR /app

# Install system dependencies required for PDF parsing and C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download SpaCy model to avoid runtime overhead
RUN python -m spacy download en_core_web_sm

# Pre-download SentenceTransformer model (Small version for Cloud Run efficiency)
# This ensures the model is baked into the image.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy the backend source code
# Note: We exclude frontend/ via .dockerignore
COPY . .

# Expose the port (Cloud Run will inject its own PORT env var, but 5000 is our default)
EXPOSE 5000

# Start the application using Gunicorn for production-grade performance
# We use 1 worker and 8 threads as a baseline for 512MB-1GB RAM instances
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
