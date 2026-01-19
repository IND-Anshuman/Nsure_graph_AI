# Use an official lightweight Python image.
FROM python:3.10-slim

# Set environment variables to ensure Python output is sent straight to terminal
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=5000

# Set working directory
WORKDIR /app

# Install system dependencies required for C-extensions and NLP libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    gcc \
    g++ \
    zlib1g-dev \
    libxml2-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

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

# Default environment variables for Cloud Run optimization
ENV KG_EMBEDDING_PROVIDER=gemini
ENV GEMINI_EMBEDDING_MODEL=text-embedding-004
ENV KG_EMBEDDING_DIM=768

# Run the web service on container startup.
# Reduced threads from 8 to 4 to save memory.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 0 main:app
