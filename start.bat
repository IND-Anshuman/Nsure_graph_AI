@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   Nsure AI GraphRAG System Startup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10+ and add it to PATH
    pause
    exit /b 1
)
echo [OK] Python found

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo Please install Node.js 18+ and add it to PATH
    pause
    exit /b 1
)
echo [OK] Node.js found
echo.

REM Check .env file
if not exist ".env" (
    echo [WARNING] .env file not found
    echo Please create .env file with your API keys
    echo Example: OPENAI_API_KEY=sk-...
    pause
    exit /b 1
)
echo [OK] .env file found
echo.

REM Install flask-cors if needed
echo [SETUP] Installing flask-cors...
pip install -q flask-cors >nul 2>&1
echo [OK] Dependencies ready
echo.

REM Start Backend
echo ========================================
echo   Starting Backend Server
echo ========================================
start "Nsure AI Backend" cmd /k "python main.py"
echo [OK] Backend starting on http://localhost:5000
echo.

REM Wait for backend to initialize
echo [WAIT] Waiting for backend to initialize...
timeout /t 5 /nobreak >nul
echo.

REM Start Frontend
echo ========================================
echo   Starting Frontend Server
echo ========================================
cd frontend

REM Check if node_modules exists
if not exist "node_modules" (
    echo [SETUP] Installing frontend dependencies...
    call npm install
    echo.
)

start "Nsure AI Frontend" cmd /k "npm run dev"
cd ..

echo [OK] Frontend starting on http://localhost:3000
echo.

echo ========================================
echo   Nsure AI is Running!
echo ========================================
echo.
echo Frontend: http://localhost:3000
echo Backend:  http://localhost:5000
echo.
echo Two terminal windows have been opened:
echo  - Backend (Flask server)
echo  - Frontend (Vite dev server)
echo.
echo Close the terminal windows to stop the servers.
echo.
pause
