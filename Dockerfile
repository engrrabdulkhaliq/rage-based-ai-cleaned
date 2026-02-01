
FROM python:3.11-slim

WORKDIR /app

# System dependencies (agar zarurat ho)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Requirements copy aur install karo
COPY ragebase-ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code copy karo
COPY ragebase-ui/ .

# Port expose karo
EXPOSE 8501

# Streamlit run karo with correct file
CMD ["streamlit", "run", "04_rag_search_only.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

---

## **.dockerignore (Root directory mein):**
```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.env
.git
*.log
.vscode/
.DS_Store
*.swp
