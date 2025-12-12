#!/bin/bash
#===============================================================================
# RyzenAI-LocalLab - Run Script
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if venv exists
if [[ ! -f "${SCRIPT_DIR}/venv/bin/activate" ]]; then
    echo "Error: Virtual environment not found. Run install.sh first."
    exit 1
fi

source "${SCRIPT_DIR}/venv/bin/activate"

# Load environment
if [[ -f "${SCRIPT_DIR}/.env" ]]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
fi

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                  RyzenAI-LocalLab                             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${GREEN}Starting server...${NC}"
echo ""
echo "  🌐 Web UI:  http://localhost:${API_PORT:-8000}"
echo "  📚 API Docs: http://localhost:${API_PORT:-8000}/docs"
echo "  🔌 OpenAI API: http://localhost:${API_PORT:-8000}/v1"
echo ""

# Run uvicorn
uvicorn backend.main:app \
    --host ${API_HOST:-0.0.0.0} \
    --port ${API_PORT:-8000} \
    --reload
