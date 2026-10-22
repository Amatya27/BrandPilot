# LoRA Brand Assistant: web app (FastAPI + RAG) in a container.
# The fine-tuned Gemma model runs outside the container through an
# OpenAI-compatible server (for example `mlx_lm.server` on the Mac, which
# needs Apple's GPU and so cannot run inside Docker). The app reaches it
# through BRAND_ASSISTANT_API_BASE.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so code changes do not invalidate this layer.
COPY requirements-app.txt requirements-local.txt ./
RUN pip install -r requirements-app.txt -r requirements-local.txt

COPY *.py ./
COPY webapp ./webapp

# Run as a non-root user; cache and outputs live in writable folders.
RUN useradd --create-home appuser \
 && mkdir -p .cache/brand_assistant outputs/app \
 && chown -R appuser:appuser /app
USER appuser

ENV BRAND_ASSISTANT_BACKEND=openai-compatible \
    BRAND_ASSISTANT_API_BASE=http://host.docker.internal:8080/v1

EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/api/health', timeout=4)"

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8010"]
