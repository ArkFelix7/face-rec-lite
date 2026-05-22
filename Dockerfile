FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

ENV INSIGHTFACE_MODEL_DIR=/app/models
RUN mkdir -p /app/models && python -c "\
from insightface.app import FaceAnalysis; \
app = FaceAnalysis(name='buffalo_l', root='/app/models', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=-1)" || echo "Model pre-download skipped (no network)"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
