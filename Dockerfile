# --- Stage 1: Build Frontend ---
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Final Image ---
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=5000

# Set working directory
WORKDIR /app

# Install system dependencies
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

# Copy backend requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download models
RUN python -m spacy download en_core_web_sm
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy backend code
COPY . .

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port
EXPOSE 5000

# Default environment variables
ENV KG_EXTRACTION_STRATEGY=hybrid
ENV KG_DOC_WORKERS=1
ENV KG_COMMUNITY_SUMMARY_WORKERS=1
ENV KG_RELATION_WORKERS=1
ENV KG_EMBEDDING_PROVIDER=gemini
ENV GEMINI_EMBEDDING_MODEL=text-embedding-004
ENV KG_EMBEDDING_DIM=768

# Startup command
CMD exec gunicorn --bind :$PORT --workers 1 --threads 2 --timeout 0 main:app
