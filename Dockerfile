# Use python-slim for a smaller footprint (approx 150MB base)
FROM python:3.11-slim

# Set environment variables for non-interactive installs and app behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install minimal system dependencies for building packages if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies - using --no-cache-dir to keep image small
# We use the CPU-only version of torch defined in requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformer model to avoid timeout on startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Copy only necessary application files
COPY 04_rag_search_only.py .
COPY embeddings.pkl .
# Add other necessary files if any (e.g., settings.json if used)
COPY settings.json .

# Railway automatically provides the PORT environment variable.
# Streamlit MUST bind to 0.0.0.0 and the assigned port to fix 502 errors.
# We use sh -c to ensure the environment variable is correctly expanded.
CMD ["sh", "-c", "streamlit run 04_rag_search_only.py --server.port=${PORT} --server.address=0.0.0.0"]
