# 🎓 Sigma RAG-Based AI Teaching Assistant

> Ask questions from the "Sigma Web Development Course" using Retrieval-Augmented Generation (RAG).

## 📌 Overview
This project is an intelligent **RAG AI Teaching Assistant** designed for students of the Sigma Web Development Course. It allows users to ask questions and receive answers based on actual course video content, complete with exact video numbers and timestamps.

## 🌟 Key Features
- **Professional Chat UI**: Modern, responsive interface with glassmorphism and smooth animations.
- **FastAPI Backend**: High-performance asynchronous backend for handling chat requests.
- **Groq Integration**: Lightning-fast inference using Groq's API.
- **Timestamp Precision**: Automatically finds and references exact timestamps (MM:SS) from course videos.
- **Multi-language Support**: Handles both Hindi and English content naturally.

## 🚀 Local Setup

### 1. Prerequisites
- Python 3.9+
- [Groq API Key](https://console.groq.com/)

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd Rage_based-ai

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file or export the following variable:
```bash
export GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run Locally
```bash
python main.py
```
The app will be available at `http://localhost:8080`.

## 🚢 Deployment (Railway)

1. **Push to GitHub**: Push your code to a GitHub repository.
2. **Connect to Railway**: Create a new project on Railway and link your repository.
3. **Variables**: Add `GROQ_API_KEY` to your Railway project variables.
4. **Deploy**: Railway will automatically use the `Dockerfile` to build and deploy your app.

## 📂 Project Structure
- `main.py`: The FastAPI backend serving the API and the frontend.
- `ragebase-ui/index.html`: The professional chat interface.
- `embeddings.pkl`: Pre-computed vector embeddings of the course content.
- `Dockerfile`: Configuration for containerized deployment.
- `04_rag_search_only.py`: (Legacy) Original Streamlit version of the assistant.

## 🎯 How It Works
1. **Frontend**: The user enters a question in the `index.html` interface.
2. **API Call**: JavaScript sends a POST request to the `/chat` endpoint.
3. **Retrieval**: The backend uses `SentenceTransformers` to find the most relevant content in `embeddings.pkl`.
4. **Augmentation**: Relevant chunks are injected into a prompt for the LLM.
5. **Generation**: Groq generates a teacher-like response with specific video references.
6. **Delivery**: The response is displayed in a polished chat bubble on the frontend.
