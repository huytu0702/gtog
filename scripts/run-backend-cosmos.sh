#!/bin/bash
# Linux/Mac script to run GraphRAG backend with Cosmos DB emulator

echo "========================================"
echo "GraphRAG Backend with Cosmos DB Emulator"
echo "========================================"
echo ""

# Check if .env.cosmos-local exists
if [ ! -f .env.cosmos-local ]; then
    echo "Error: .env.cosmos-local not found!"
    echo "Please create it first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d .venv_new ]; then
    echo "Error: Virtual environment .venv_new not found!"
    echo "Please create it first: python -m venv .venv_new"
    exit 1
fi

# Activate virtual environment
source .venv_new/bin/activate

# Change to backend directory
cd backend

# Run uvicorn with cosmos env
echo "Starting backend server..."
echo "API will be available at: http://localhost:8000"
echo "Health check: http://localhost:8000/health"
echo ""

exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --env-file ../.env.cosmos-local --reload
