# ─────────────────────────────────────────────────────────────────────────────
# FashionVerse — Dockerfile
# Deploys to: Render (free tier), HF Spaces PRO, or any Docker host
#
# Strategy: uses requirements-deploy.txt (no PyTorch/SB3) so the image stays
# under 400 MB RAM. The RL agent automatically falls back to rule-based
# OutfitGenerator — all other features (Gemini chat, analytics) work fully.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System deps + Node.js 20 (for frontend build) ────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python: install lightweight deps (no PyTorch) ────────────────────────────
COPY testrl102/FashionVerse/requirements-deploy.txt ./requirements-deploy.txt
RUN pip install --no-cache-dir -r requirements-deploy.txt

# ── Frontend: build React/Vite app ───────────────────────────────────────────
COPY testrl102/FashionVerse/frontend/ ./frontend/
WORKDIR /app/frontend
# VITE_API_URL="" → relative same-origin calls to FastAPI
RUN npm install && VITE_API_URL="" npm run build

# ── Copy the full FashionVerse project ───────────────────────────────────────
WORKDIR /app
COPY testrl102/FashionVerse/ .

# ── Runtime ──────────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
