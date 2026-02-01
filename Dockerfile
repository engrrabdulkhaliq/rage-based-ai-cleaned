FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

# 🔥 Install CPU-only torch
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "04_rag_search_only.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
