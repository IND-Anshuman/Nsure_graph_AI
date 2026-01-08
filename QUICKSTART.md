# 🚀 Quick Start Guide - Nsure AI GraphRAG

## Fastest Way to Get Started (Windows)

### 1. Install flask-cors (if not already installed)
```powershell
pip install flask-cors
```

### 2. Start Backend Server
```powershell
python main.py
```
Backend will run on: **http://localhost:5000**

### 3. Start Frontend (New Terminal)
```powershell
cd frontend
npm run dev
```
Frontend will run on: **http://localhost:3000**

### 4. Access the Application
Open your browser: **http://localhost:3000**

---

## Or Use the Automated Script

### Windows
```powershell
.\start.bat
```

### Linux/Mac
```bash
chmod +x start.sh
./start.sh
```

---

## First Time Setup

### 1. Backend Dependencies
```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. API Keys
Create `.env` file in root directory:
```
OPENAI_API_KEY=sk-your-key-here
# OR
GOOGLE_API_KEY=your-gemini-key-here
```

### 3. Frontend Dependencies
```powershell
cd frontend
npm install
```

---

## How to Use

1. **Upload Document**: Click or drag PDF files into the upload zone
2. **Enter Question**: Type your question in the text area
3. **Submit**: Press Enter or click "Submit Query"
4. **View Results**: See the AI-generated answer with supporting facts

---

## Troubleshooting

### Backend won't start
- Check if `.env` file exists with API keys
- Verify Python 3.10+ is installed: `python --version`
- Check if port 5000 is available

### Frontend won't start  
- Verify Node.js 18+ is installed: `node --version`
- Delete `node_modules` and run `npm install` again
- Check if port 3000 is available

### CORS errors
- Ensure `flask-cors` is installed: `pip install flask-cors`
- Verify backend is running on port 5000
- Check frontend `.env` has `VITE_API_URL=http://localhost:5000`

### API connection fails
- Make sure both servers are running
- Test backend: `curl http://localhost:5000/health`
- Check browser console for detailed errors

---

## Configuration

### Fast Mode (Recommended for Testing)
The frontend is configured to use fast mode by default:
- Skips Neo4j export
- Skips community summaries
- 3-5x faster processing
- Still provides accurate answers

### Caching
Enabled by default for:
- Graph construction (reuses built graphs)
- Index building (reuses embeddings)
- Significantly faster for repeated queries

---

## What Gets Created

### Backend Outputs
```
outputs/
  ├── api_cache/        # Cached graphs and indexes
  ├── api_graphs/       # Generated GraphML files
  └── api_uploads/      # Uploaded PDF files
```

### File Handling
- Uploaded files are temporarily stored
- Graphs are cached and reused
- GraphML files persist for later querying

---

## Next Steps

1. Try uploading a PDF document
2. Ask questions about the document
3. Explore the knowledge graph visualization
4. Check the query stats (nodes, edges, timing)
5. Review extracted facts and confidence scores

---

## Need Help?

- Check [INTEGRATION_README.md](INTEGRATION_README.md) for detailed documentation
- Review backend logs in the terminal running `python main.py`
- Check frontend console (F12 in browser)
- Enable verbose mode in API calls for detailed logging
