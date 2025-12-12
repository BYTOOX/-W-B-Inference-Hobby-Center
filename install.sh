#!/bin/bash
#===============================================================================
# RyzenAI-LocalLab - Installation Script
# Optimized for AMD Ryzen AI MAX+ with Radeon GPU (ROCm)
#===============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           RyzenAI-LocalLab Installation Script                ║"
echo "║         Optimized for AMD Ryzen AI MAX+ (ROCm)                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
MODELS_DIR="/srv/models"
DATA_DIR="${SCRIPT_DIR}/data"
PYTHON_MIN_VERSION="3.11"
ROCM_VERSION="6.2"

#-------------------------------------------------------------------------------
# Helper Functions
#-------------------------------------------------------------------------------
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        return 1
    fi
    return 0
}

#-------------------------------------------------------------------------------
# System Checks
#-------------------------------------------------------------------------------
echo ""
log_info "Checking system requirements..."

# Check OS
if [[ ! -f /etc/debian_version ]]; then
    log_warn "This script is optimized for Debian/Ubuntu. Proceeding anyway..."
fi

# Check Python version
if check_command python3; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    log_info "Python version: ${PYTHON_VERSION}"
    
    if [[ "$(echo -e "${PYTHON_MIN_VERSION}\n${PYTHON_VERSION}" | sort -V | head -n1)" != "${PYTHON_MIN_VERSION}" ]]; then
        log_error "Python ${PYTHON_MIN_VERSION}+ is required. Found: ${PYTHON_VERSION}"
        exit 1
    fi
else
    log_error "Python3 not found. Please install Python ${PYTHON_MIN_VERSION}+"
    exit 1
fi

# Check for ROCm
log_info "Checking ROCm installation..."
if check_command rocm-smi; then
    log_info "ROCm detected!"
    rocm-smi --showproductname 2>/dev/null || true
else
    log_warn "ROCm not detected. See docs/INSTALL_ROCM.md for installation guide."
    log_warn "Continuing with CPU-only setup..."
fi

# Check GPU with PyTorch (if installed)
if python3 -c "import torch" 2>/dev/null; then
    GPU_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
    if [[ "$GPU_AVAILABLE" == "True" ]]; then
        log_info "PyTorch GPU (ROCm) support: ✓"
    else
        log_warn "PyTorch GPU support not available. Will run on CPU."
    fi
fi

#-------------------------------------------------------------------------------
# Create Directories
#-------------------------------------------------------------------------------
echo ""
log_info "Creating directories..."

# Models directory
if [[ ! -d "$MODELS_DIR" ]]; then
    log_info "Creating models directory: ${MODELS_DIR}"
    sudo mkdir -p "$MODELS_DIR"
    sudo chown $USER:$USER "$MODELS_DIR"
    log_info "Models directory created."
else
    log_info "Models directory exists: ${MODELS_DIR}"
fi

# Data directory
mkdir -p "$DATA_DIR"
log_info "Data directory: ${DATA_DIR}"

#-------------------------------------------------------------------------------
# Python Virtual Environment
#-------------------------------------------------------------------------------
echo ""
log_info "Setting up Python virtual environment..."

if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    log_info "Virtual environment created at: ${VENV_DIR}"
else
    log_info "Virtual environment already exists."
fi

# Activate venv
source "${VENV_DIR}/bin/activate"

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel

#-------------------------------------------------------------------------------
# Install PyTorch with ROCm
#-------------------------------------------------------------------------------
echo ""
log_info "Installing PyTorch with ROCm support..."

# Check if ROCm is available
if check_command rocm-smi; then
    log_info "Installing PyTorch for ROCm ${ROCM_VERSION}..."
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/rocm${ROCM_VERSION}"
else
    log_warn "ROCm not detected. Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Verify PyTorch installation
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
"

#-------------------------------------------------------------------------------
# Install Project Dependencies
#-------------------------------------------------------------------------------
echo ""
log_info "Installing project dependencies..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

#-------------------------------------------------------------------------------
# Environment File
#-------------------------------------------------------------------------------
echo ""
log_info "Setting up environment configuration..."

ENV_FILE="${SCRIPT_DIR}/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    log_info "Creating default .env file..."
    
    # Generate a random secret key
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    
    cat > "$ENV_FILE" << EOF
# RyzenAI-LocalLab Configuration
# Generated on $(date)

# Paths
MODELS_PATH=${MODELS_DIR}
DATA_PATH=${DATA_DIR}

# Server
API_HOST=0.0.0.0
API_PORT=8000
UI_PORT=8501

# Security
SECRET_KEY=${SECRET_KEY}
FIRST_ADMIN_USERNAME=admin
FIRST_ADMIN_PASSWORD=changeme

# Hardware
# Set to "cpu" to force CPU-only mode
# Set to "cuda" or "auto" for GPU (ROCm)
DEVICE=auto

# Logging
LOG_LEVEL=INFO
EOF
    
    log_info ".env file created. Please review and update settings."
else
    log_info ".env file already exists."
fi

#-------------------------------------------------------------------------------
# Create Run Script
#-------------------------------------------------------------------------------
echo ""
log_info "Creating run script..."

cat > "${SCRIPT_DIR}/run.sh" << 'EOF'
#!/bin/bash
#===============================================================================
# RyzenAI-LocalLab - Run Script
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/venv/bin/activate"

# Load environment
set -a
source "${SCRIPT_DIR}/.env"
set +a

# Colors
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                  RyzenAI-LocalLab                             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check what to run
case "${1:-all}" in
    api)
        echo "Starting API server..."
        uvicorn backend.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000} --reload
        ;;
    ui)
        echo "Starting Streamlit UI..."
        streamlit run ui/app.py --server.port ${UI_PORT:-8501} --server.address ${API_HOST:-0.0.0.0}
        ;;
    all|*)
        echo "Starting API server and Streamlit UI..."
        echo "API: http://localhost:${API_PORT:-8000}"
        echo "UI:  http://localhost:${UI_PORT:-8501}"
        echo ""
        
        # Start API in background
        uvicorn backend.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000} &
        API_PID=$!
        
        # Wait a bit for API to start
        sleep 2
        
        # Start Streamlit
        streamlit run ui/app.py --server.port ${UI_PORT:-8501} --server.address ${API_HOST:-0.0.0.0} &
        UI_PID=$!
        
        # Handle shutdown
        trap "kill $API_PID $UI_PID 2>/dev/null; exit" SIGINT SIGTERM
        
        wait
        ;;
esac
EOF

chmod +x "${SCRIPT_DIR}/run.sh"
log_info "Run script created: ./run.sh"

#-------------------------------------------------------------------------------
# Summary
#-------------------------------------------------------------------------------
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Installation Complete! 🎉                        ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo "  1. Review configuration in .env"
echo "  2. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo "  3. Start the application:"
echo "     ./run.sh"
echo ""
echo "Access points:"
echo "  - API: http://localhost:8000"
echo "  - UI:  http://localhost:8501"
echo "  - API Docs: http://localhost:8000/docs"
echo ""

# Deactivate venv
deactivate
