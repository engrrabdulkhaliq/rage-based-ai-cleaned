FROM python:3.11-slim

WORKDIR /app

# Install gcc for building dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# EXPOSE optional, Railway ignores it
EXPOSE 8080

# Streamlit: bind to Railway PORT environment variable
CMD ["sh", "-c", "streamlit run 04_rag_search_only.py --server.address=0.0.0.0 --server.port=${PORT}"]
