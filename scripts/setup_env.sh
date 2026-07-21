#!/usr/bin/env bash
# Sanctum — environment setup script
set -euo pipefail

echo "=== Sanctum Environment Setup ==="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

# Copy .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env from template..."
    cp .env.example .env
fi

# Set up frontend
echo "Setting up frontend..."
cd frontend
npm install
cd ..

# Set up pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install || echo "Warning: pre-commit install failed (optional)"

echo ""
echo "=== Setup complete! ==="
echo "  Activate venv:  source .venv/bin/activate"
echo "  Run server:     uvicorn src.server.main:app --reload"
echo "  Run frontend:   cd frontend && npm run dev"
echo "  Run tests:      pytest"
echo "  Run demo:       ./scripts/run_demo.sh"
