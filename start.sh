#!/bin/bash

echo "🚀 Starting Nsure AI GraphRAG System..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

echo "✅ Python and Node.js found"
echo ""

# Backend setup
echo "📦 Setting up backend..."
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Please create one with your API keys."
    echo "   Example: OPENAI_API_KEY=sk-..."
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -q flask-cors 2>/dev/null || pip3 install -q flask-cors

# Start backend server
echo "🔧 Starting Flask backend on http://localhost:5000..."
python3 main.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start. Check logs above."
    exit 1
fi

echo "✅ Backend started successfully (PID: $BACKEND_PID)"
echo ""

# Frontend setup
echo "📦 Setting up frontend..."
cd frontend

# Install frontend dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start frontend server
echo "🎨 Starting React frontend on http://localhost:3000..."
npm run dev &
FRONTEND_PID=$!

cd ..

echo ""
echo "✅ Nsure AI GraphRAG System is running!"
echo ""
echo "📍 Frontend: http://localhost:3000"
echo "📍 Backend:  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop all servers..."

# Handle Ctrl+C
trap "echo ''; echo '🛑 Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
