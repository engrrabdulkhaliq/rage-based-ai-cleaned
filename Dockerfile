# Use python-slim for a smaller footprint
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# We use the CPU-only version of torch to keep the image size well under 4GB
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Railway automatically provides the PORT environment variable
# Streamlit must bind to 0.0.0.0 and the assigned port to fix 502 errors
CMD ["sh", "-c", "streamlit run 04_rag_search_only.py --server.port=${PORT} --server.address=0.0.0.0"]
