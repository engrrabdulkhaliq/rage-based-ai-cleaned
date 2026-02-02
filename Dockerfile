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
COPY main.py .
COPY embeddings.pkl .
COPY settings.json .
COPY ragebase-ui/ ./ragebase-ui/

# Railway automatically provides the PORT environment variable.
# We use uvicorn to serve the FastAPI app.
CMD ["sh", "-c", "python main.py"]
