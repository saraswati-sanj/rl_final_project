# ─────────────────────────────────────────────────────────────────────────────
# FashionVerse — Dockerfile for Hugging Face Spaces
# Builds the React frontend, then runs FastAPI serving both API + static assets
# Port: 7860 (required by HF Spaces)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System deps: Node.js 20 for frontend build ──────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python: install CPU-only PyTorch first (much smaller ~200MB vs ~2GB) ────
RUN pip install --no-cache-dir \
        torch==2.4.0 \
        --index-url https://download.pytorch.org/whl/cpu

# ── Python: install remaining dependencies ───────────────────────────────────
COPY testrl102/FashionVerse/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Frontend: build React/Vite app ───────────────────────────────────────────
COPY testrl102/FashionVerse/frontend/ ./frontend/
WORKDIR /app/frontend
# VITE_API_URL="" → relative URLs so frontend calls same-origin FastAPI
RUN npm install && VITE_API_URL="" npm run build

# ── Copy the full FashionVerse project ───────────────────────────────────────
WORKDIR /app
COPY testrl102/FashionVerse/ .

# ── Runtime environment ───────────────────────────────────────────────────────
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# HF Spaces requires exactly port 7860
EXPOSE 7860

# ── Start FastAPI ─────────────────────────────────────────────────────────────
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
