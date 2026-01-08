# Nsure AI - Complete Integration Guide

## Overview
Nsure AI is a production-grade GraphRAG (Graph Retrieval-Augmented Generation) system with a modern React frontend and Flask backend. It enables intelligent document analysis and question-answering using knowledge graphs.

## Architecture

### Backend (Flask + Python)
- **Framework**: Flask with CORS support
- **Graph Processing**: NetworkX, Louvain communities
- **LLM Integration**: OpenAI, Google Gemini
- **Document Processing**: PDFPlumber, BeautifulSoup
- **Embeddings**: Sentence Transformers

### Frontend (React + TypeScript)
- **Framework**: React 18 + TypeScript + Vite
- **UI**: Tailwind CSS + shadcn/ui
- **Animations**: Framer Motion
- **State**: TanStack React Query
- **Routing**: React Router v6

## Prerequisites

- Python 3.10+ 
- Node.js 18+ and npm
- OpenAI API key or Google Gemini API key

## Installation

### 1. Backend Setup

```bash
# Navigate to root directory
cd Nsure_graph_AI

# Install Python dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm

# Create .env file
cp .env.example .env

# Edit .env and add your API keys:
# OPENAI_API_KEY=your_key_here
# OR
# GOOGLE_API_KEY=your_key_here
```

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create .env file (already configured for localhost)
# VITE_API_URL=http://localhost:5000
```

## Running the Application

### Start Backend Server

```bash
# From root directory
python main.py

# Server will start on http://localhost:5000
# Default settings:
# - Host: 127.0.0.1
# - Port: 5000
# - Debug: false

# Custom settings via environment variables:
# KG_API_HOST=0.0.0.0 KG_API_PORT=8000 python main.py
```

### Start Frontend Development Server

```bash
# From frontend directory
npm run dev

# Server will start on http://localhost:3000
# Hot Module Replacement (HMR) enabled
```

### Access the Application

Open your browser and navigate to: **http://localhost:3000**

## Usage Guide

### 1. Upload Documents
- Click or drag-and-drop PDF files into the upload zone
- Multiple files supported
- Supported formats: PDF, URLs (in API)

### 2. Enter Query
- Type your question in the text area
- Press Enter (without Shift) or click "Submit Query"
- System will:
  - Build knowledge graph from documents
  - Extract entities and relationships
  - Perform semantic search
  - Generate comprehensive answer

### 3. View Results
- **Stats Card**: Shows graph size (nodes/edges) and timing
- **Answer Card**: Displays AI-generated answer
- **Key Facts**: Extracted important facts with evidence
- **Confidence Score**: Answer reliability metric

## API Endpoints

### Health Check
```bash
GET /health
```

### Build Knowledge Graph
```bash
POST /build
Content-Type: multipart/form-data

Fields:
- pdf: File
- options: JSON string (optional)
```

### Query (Build + Answer)
```bash
POST /query
Content-Type: multipart/form-data

Fields:
- pdf: File
- questions: JSON array string
- options: JSON string (optional)
```

### Query Existing Graph
```bash
POST /query_graphml
Content-Type: application/json

{
  "graphml_path": "path/to/graph.graphml",
  "questions": ["Your question?"],
  "options": {}
}
```

## Configuration Options

### Build Options
```json
{
  "build": {
    "doc_workers": 4,
    "skip_neo4j": true,
    "fast": true
  },
  "cache": {
    "enabled": true,
    "use_graph_cache": true
  },
  "verbose": false
}
```

### Query Options
```json
{
  "qa": {
    "top_n_semantic": 20,
    "top_k_final": 40,
    "rerank_top_k": 12,
    "max_workers": 2
  },
  "index": {
    "community_summary_weight": 1.5,
    "document_weight": 1.0
  }
}
```

## Environment Variables

### Backend (.env)
```bash
# API Keys (choose one or both)
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Optional Settings
KG_API_HOST=127.0.0.1
KG_API_PORT=5000
KG_API_DEBUG=0
KG_API_CACHE_DIR=outputs/api_cache
KG_API_GRAPHS_DIR=outputs/api_graphs
KG_API_UPLOAD_DIR=outputs/api_uploads
KG_DOC_WORKERS=4
KG_QA_MAX_WORKERS=2
KG_ALLOW_LOCAL_PATH=0
```

### Frontend (.env)
```bash
VITE_API_URL=http://localhost:5000
```

## Project Structure

```
Nsure_graph_AI/
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── components/      # UI components
│   │   │   ├── ui/          # shadcn/ui components
│   │   │   ├── BackgroundParticles.tsx
│   │   │   ├── FileUpload.tsx
│   │   │   └── VibrantEffects.tsx
│   │   ├── lib/
│   │   │   ├── api.ts       # API service layer
│   │   │   └── utils.ts     # Utility functions
│   │   ├── pages/
│   │   │   ├── LandingPage.tsx
│   │   │   └── AgentPage.tsx
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
├── main.py                  # Flask API server
├── graph_maker.py           # KG construction
├── engine.py                # Query engine
├── ner.py                   # Entity extraction
├── Community_processing.py  # Community detection
├── phase8_retrieval_enhanced.py  # Indexing
├── hybrid_search.py         # Search algorithm
├── llm_rerank.py            # LLM reranking
├── llm_synthesis.py         # Answer synthesis
├── requirements.txt
└── README.md
```

## Features

### Frontend Features
- 🎨 Modern glassmorphism UI with dark theme
- ✨ Smooth animations with Framer Motion
- 📁 Drag-and-drop file upload
- 💬 Real-time query processing
- 📊 Comprehensive result visualization
- 📱 Responsive design
- ⚡ Fast HMR development

### Backend Features
- 🧠 Advanced GraphRAG pipeline
- 🔍 Multi-level community detection
- 📈 Semantic search with embeddings
- 🤖 LLM-powered reranking and synthesis
- 💾 Intelligent caching system
- 🔄 Parallel processing support
- 📦 GraphML persistence
- 🌐 Neo4j export support

## Performance Optimization

### Backend
- **Caching**: Graph and index caching enabled by default
- **Fast Mode**: Skip community summaries for 3-5x speedup
- **Parallel Processing**: Configurable worker threads
- **Lazy Loading**: Index built on first query

### Frontend
- **Code Splitting**: Automatic route-based splitting
- **Lazy Loading**: Dynamic component imports
- **Optimized Builds**: Vite production optimization
- **Asset Optimization**: Image and font optimization

## Troubleshooting

### Backend Issues

**CORS Errors**
```bash
# Make sure flask-cors is installed
pip install flask-cors

# Check CORS configuration in main.py
```

**Missing API Keys**
```bash
# Verify .env file exists and contains keys
cat .env | grep API_KEY
```

**Port Already in Use**
```bash
# Change port in .env
KG_API_PORT=8000
```

### Frontend Issues

**API Connection Failed**
```bash
# Verify backend is running
curl http://localhost:5000/health

# Check VITE_API_URL in frontend/.env
```

**Module Not Found**
```bash
cd frontend
npm install
```

**Build Errors**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

## Development Tips

### Hot Reloading
- Frontend: Vite HMR works automatically
- Backend: Set `KG_API_DEBUG=1` for Flask debug mode

### Testing API
```bash
# Test health endpoint
curl http://localhost:5000/health

# Test query with file
curl -X POST http://localhost:5000/query \
  -F "pdf=@test.pdf" \
  -F 'questions=["What is this document about?"]'
```

### Debugging
- Backend logs: Check terminal running `python main.py`
- Frontend: Open browser DevTools (F12) → Console/Network tabs
- Enable verbose mode: Set `verbose: true` in options

## Production Deployment

### Backend
```bash
# Use production WSGI server (e.g., Gunicorn)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 main:app

# Or use uWSGI
pip install uwsgi
uwsgi --http :5000 --wsgi-file main.py --callable app
```

### Frontend
```bash
# Build for production
cd frontend
npm run build

# Serve with nginx or any static server
# Build output in: frontend/dist/
```

## License

See LICENSE file for details.

## Support

For issues, questions, or contributions, please open an issue on the repository.

## Credits

- Built with React, TypeScript, Flask, and Python
- UI components from shadcn/ui
- Animations powered by Framer Motion
- GraphRAG architecture inspired by Microsoft's GraphRAG
