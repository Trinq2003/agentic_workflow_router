# Multi-stage Dockerfile for NetMind Workflow
# Stage 1: Build stage
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.10-slim as runtime

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    # Required for Excel processing
    libxml2 \
    libxslt1.1 \
    # Required for Dagster
    graphviz \
    # Utilities
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r netmind && useradd -r -g netmind netmind

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY dagster_workspace.yaml .
COPY main.py .

# Copy data directory (if needed for containerized execution)
COPY data/ ./data/

# Change ownership to non-root user
RUN chown -R netmind:netmind /app
USER netmind

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.append('src'); from data.data_loader import create_default_dataloader; loader = create_default_dataloader(); print('Health check passed')" || exit 1

# Default command
CMD ["python", "main.py"]

# Expose port for Dagster webserver (if running webserver)
EXPOSE 3000
