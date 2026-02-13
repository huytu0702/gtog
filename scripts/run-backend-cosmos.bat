@echo off
REM Windows batch script to run GraphRAG backend with Cosmos DB emulator

echo ========================================
echo GraphRAG Backend with Cosmos DB Emulator
echo ========================================
echo.

REM Check if .env.cosmos-local exists
if not exist .env.cosmos-local (
    echo Error: .env.cosmos-local not found!
    echo Please create it first.
    exit /b 1
)

REM Check if virtual environment exists
if not exist .venv_new\Scripts\activate.bat (
    echo Error: Virtual environment .venv_new not found!
    echo Please create it first: python -m venv .venv_new
    exit /b 1
)

REM Activate virtual environment
call .venv_new\Scripts\activate.bat

REM Change to backend directory
cd backend

REM Run uvicorn with cosmos env
echo Starting backend server...
echo API will be available at: http://localhost:8000
echo Health check: http://localhost:8000/health
echo.

.\.venv_new\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000 --env-file ..\.env.cosmos-local --reload
