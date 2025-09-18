#!/bin/bash

# Docker Run Script for NetMind Workflow
# Usage: ./scripts/docker-run.sh [command] [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
DEV_COMPOSE_FILE="$PROJECT_ROOT/docker-compose.dev.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Build images
build() {
    local target=${1:-runtime}
    log_info "Building Docker images (target: $target)..."

    if [ "$target" = "dev" ]; then
        docker-compose -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" build
    else
        docker-compose -f "$COMPOSE_FILE" build
    fi

    log_success "Docker images built successfully"
}

# Start services
start() {
    local target=${1:-runtime}
    log_info "Starting NetMind Workflow services (target: $target)..."

    if [ "$target" = "dev" ]; then
        docker-compose -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" up -d
    else
        docker-compose -f "$COMPOSE_FILE" up -d
    fi

    log_success "Services started successfully"
    log_info "Dagster UI: http://localhost:3000"
    log_info "Jupyter Notebook: http://localhost:8888 (if enabled)"
}

# Stop services
stop() {
    log_info "Stopping NetMind Workflow services..."
    docker-compose -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" down
    log_success "Services stopped successfully"
}

# Restart services
restart() {
    local target=${1:-runtime}
    log_info "Restarting NetMind Workflow services..."
    stop
    sleep 2
    start "$target"
}

# View logs
logs() {
    local service=${1:-netmind-app}
    log_info "Showing logs for service: $service"
    docker-compose -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" logs -f "$service"
}

# Execute command in container
exec_cmd() {
    local service=${1:-netmind-app}
    local command=${2:-bash}
    log_info "Executing command in $service container: $command"
    docker-compose -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" exec "$service" $command
}

# Run data processing
run_processing() {
    local mode=${1:-parallel}
    log_info "Running data processing in $mode mode..."

    case $mode in
        parallel)
            docker-compose -f "$COMPOSE_FILE" exec netmind-app \
                python -c "
import sys
sys.path.append('src')
from dagster import execute_job
from src.dagster.jobs import parallel_data_processing_job
result = execute_job(parallel_data_processing_job)
print('Job completed with status:', result.success)
"
            ;;
        sequential)
            docker-compose -f "$COMPOSE_FILE" exec netmind-app \
                python -c "
import sys
sys.path.append('src')
from dagster import execute_job
from src.dagster.jobs import sequential_data_processing_job
result = execute_job(sequential_data_processing_job)
print('Job completed with status:', result.success)
"
            ;;
        *)
            log_error "Invalid mode. Use 'parallel' or 'sequential'"
            exit 1
            ;;
    esac
}

# Run tests
run_tests() {
    log_info "Running tests..."
    docker-compose -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" exec netmind-app \
        python -m pytest tests/ -v --cov=src --cov-report=html
}

# Clean up
cleanup() {
    log_info "Cleaning up Docker resources..."
    docker-compose -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" down -v
    docker system prune -f
    log_success "Cleanup completed"
}

# Show status
status() {
    log_info "NetMind Workflow Status:"
    echo ""
    docker-compose -f "$COMPOSE_FILE" -f "$DEV_COMPOSE_FILE" ps
    echo ""
    log_info "Disk usage:"
    docker system df
}

# Show help
show_help() {
    cat << EOF
NetMind Workflow Docker Management Script

USAGE:
    $0 [COMMAND] [OPTIONS]

COMMANDS:
    build [target]          Build Docker images (default: runtime, or 'dev')
    start [target]          Start all services (default: runtime, or 'dev')
    stop                    Stop all services
    restart [target]        Restart all services
    logs [service]          Show logs for service (default: netmind-app)
    exec [service] [cmd]    Execute command in container (default: bash)
    process [mode]          Run data processing (parallel/sequential)
    test                    Run test suite
    status                  Show status of services
    cleanup                 Clean up Docker resources
    help                    Show this help message

SERVICES:
    netmind-app            Main application
    dagster-webserver      Dagster UI
    dagster-daemon         Dagster job scheduler
    postgres-db            PostgreSQL database
    redis                  Redis cache
    jupyter                Jupyter notebook (dev only)

EXAMPLES:
    $0 build dev           # Build development images
    $0 start dev           # Start development environment
    $0 process parallel    # Run parallel data processing
    $0 logs dagster-webserver  # Show Dagster logs
    $0 exec netmind-app python main.py  # Run main script

EOF
}

# Main script logic
main() {
    check_docker

    case ${1:-help} in
        build)
            build "${2:-runtime}"
            ;;
        start)
            start "${2:-runtime}"
            ;;
        stop)
            stop
            ;;
        restart)
            restart "${2:-runtime}"
            ;;
        logs)
            logs "${2:-netmind-app}"
            ;;
        exec)
            exec_cmd "${2:-netmind-app}" "${3:-bash}"
            ;;
        process)
            run_processing "${2:-parallel}"
            ;;
        test)
            run_tests
            ;;
        status)
            status
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
