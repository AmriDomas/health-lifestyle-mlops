#!/bin/bash

# build.sh - Build Docker images for Health ML Pipeline

set -e

echo "🚀 Building Health ML Pipeline Docker images..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build images
print_status "Building backend image..."
docker build -t health-ml-backend:latest -f Dockerfile .

print_status "Building frontend image..."
docker build -t health-ml-frontend:latest -f Dockerfile.streamlit .

print_status "Building training image..."
docker build -t health-ml-training:latest -f Dockerfile.training .

print_status "Building CI image..."
docker build -t health-ml-ci:latest -f Dockerfile.ci .

print_status "Building monitoring image..."
docker build -t health-ml-monitoring:latest -f Dockerfile.monitoring .

# List built images
print_status "Built images:"
docker images | grep health-ml

print_status "Build completed successfully! 🎉"
echo ""
echo "Next steps:"
echo "1. Run development environment: docker-compose up -d"
echo "2. Run production environment: docker-compose -f docker-compose.prod.yml up -d"
echo "3. View logs: docker-compose logs -f"