# Multi-stage Dockerfile for NetMind Workflow
# Stage 1: Build stage
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/opt/venv/nltk_data

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

# Pre-download NLTK resources (tokenizers, taggers, sentiment, corpora, chunkers)
RUN python -m nltk.downloader -d "$NLTK_DATA" \
    punkt \
    averaged_perceptron_tagger \
    vader_lexicon \
    stopwords \
    wordnet \
    omw-1.4 \
    maxent_ne_chunker \
    words

# Pre-download Underthesea models
RUN underthesea download-model TC_BANK_V131 && \
    underthesea download-model TC_GENERAL_V131 && \
    underthesea download-model SA_GENERAL && \
    underthesea download-model SA_GENERAL_V131 && \
    underthesea download-model SA_BANK && \
    underthesea download-model SA_BANK_V131 && \
    underthesea download-model VIET_TTS_V0_4_1 && \
    underthesea download-model LANG_DETECT_FAST_TEXT

# Stage 2: Runtime stage
FROM python:3.10-slim as runtime

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    # Required for Excel processing
    libxml2 \
    libxslt1.1 \
    # For fastText runtime used by Underthesea language detection
    libgomp1 \
    # Utilities
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV NLTK_DATA=/opt/venv/nltk_data

# Copy Underthesea cache (downloaded models)
COPY --from=builder /root/.underthesea /root/.underthesea

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY main.py .

# Copy data directory (if needed for containerized execution)
COPY data/ ./data/

# Run as root to allow runtime downloads of NLTK/Underthesea resources
# Default user in this image is root; no USER directive needed

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.append('src'); from data.data_loader import create_default_dataloader; loader = create_default_dataloader(); print('Health check passed')" || exit 1

# Expose API port
EXPOSE 8000

# Default command runs API server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
