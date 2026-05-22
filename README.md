# Face Recognition API

A production-ready face recognition REST API built with FastAPI, InsightFace (ArcFace 512-dim embeddings), PostgreSQL + pgvector, and Redis.

## Features

- **Enroll** faces per user (up to 5 per user by default)
- **Verify** a query face against all enrolled faces (1:1 cosine similarity)
- **Quality gating** — blur, brightness, pose, and size checks before enrollment
- **Deduplication** — SHA-256 hash + embedding cosine similarity (≥0.95 threshold)
- **Bearer token auth** with bcrypt-hashed API keys
- **Fixed-window rate limiting** (per API key, per minute) via Redis
- **Prometheus metrics** at `GET /v1/metrics`
- **GDPR-friendly** — only 512-dim embeddings stored, no raw images

## Quickstart (Docker)

```bash
# Clone and enter the project
git clone <repo-url> && cd face-rec-lite

# Start all services (API + PostgreSQL + Redis)
docker compose up --build

# Run database migrations (in a separate terminal)
docker compose exec api alembic upgrade head

# Create your first API key
docker compose exec api python scripts/create_api_key.py --name "my-app"
# → prints: sk_live_<48 hex chars>  — save this, shown once

# Health check
curl http://localhost:8000/v1/health
```

## API Reference

All endpoints (except health/metrics) require `Authorization: Bearer <key>`.

All responses use the envelope: `{ "success": bool, "data": ..., "error": ..., "request_id": str }`.

### Users

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/users` | Register a new user |
| `GET` | `/v1/users/{user_id}` | Get user + face count |
| `DELETE` | `/v1/users/{user_id}` | Delete user + all faces |

**Create user:**
```bash
curl -X POST http://localhost:8000/v1/users \
  -H "Authorization: Bearer sk_live_..." \
  -H "Content-Type: application/json" \
  -d '{"external_id": "alice", "display_name": "Alice"}'
```

### Faces

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/users/{user_id}/faces` | Enroll a face |
| `GET` | `/v1/users/{user_id}/faces` | List enrolled faces |
| `DELETE` | `/v1/users/{user_id}/faces/{face_id}` | Delete a face |

**Enroll a face** (base64-encoded JPEG/PNG, with or without data URI prefix):
```bash
IMAGE=$(base64 -i photo.jpg)
curl -X POST http://localhost:8000/v1/users/alice/faces \
  -H "Authorization: Bearer sk_live_..." \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE\"}"
```

### Verification

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/users/{user_id}/verify` | Verify a query face |

**Verify:**
```bash
IMAGE=$(base64 -i query.jpg)
curl -X POST http://localhost:8000/v1/users/alice/verify \
  -H "Authorization: Bearer sk_live_..." \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$IMAGE\", \"threshold\": 0.60}"
```

Response:
```json
{
  "success": true,
  "data": {
    "match": true,
    "confidence": 0.87,
    "threshold_used": 0.60,
    "best_matching_face_id": "uuid...",
    "processing_time_ms": 42.3,
    "all_scores": [{"face_id": "uuid...", "similarity": 0.87}]
  }
}
```

## Local Development (without Docker)

Requirements: Python 3.11+, PostgreSQL 15 with pgvector, Redis 7.

```bash
# Install dependencies
pip install -e ".[dev]"

# Set environment variables
export DATABASE_URL="postgresql+asyncpg://faceapi:faceapi@localhost:5432/facedb"
export REDIS_URL="redis://localhost:6379/0"

# Run migrations
alembic upgrade head

# Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Configuration

All settings can be overridden via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://faceapi:faceapi@localhost:5432/facedb` | PostgreSQL DSN |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis DSN |
| `ML_DEVICE` | `cpu` | `cpu` or `cuda` |
| `MIN_QUALITY_SCORE` | `0.5` | Minimum overall quality for enrollment |
| `MIN_FACE_SIZE_PX` | `80` | Minimum face dimension in pixels |
| `MAX_YAW_DEG` | `35.0` | Maximum head yaw for enrollment |
| `DEFAULT_VERIFICATION_THRESHOLD` | `0.60` | Cosine similarity threshold |
| `MAX_FACES_PER_USER` | `5` | Face cap per user |
| `DEDUP_THRESHOLD` | `0.95` | Cosine similarity above which = duplicate |
| `DEFAULT_RATE_LIMIT_RPM` | `100` | Requests per minute per API key |
| `APP_ENV` | `development` | `development` or `production` |

## Running Tests

```bash
# Unit tests only (no DB/Redis required)
pytest tests/unit/ -v

# Integration tests (requires PostgreSQL + Redis)
pytest tests/integration/ -v

# All tests
pytest -v
```

## Architecture

```
Client → AuthMiddleware → RateLimitMiddleware → Router → DatabaseService / FaceMLService
                                                     ↕               ↕
                                                PostgreSQL+pgvector  Redis
```

- **ML model**: InsightFace `buffalo_l` pack — RetinaFace detection + ArcFace 512-dim embeddings
- **Similarity**: Cosine similarity on L2-normalized embeddings (equivalent to dot product)
- **Auth**: 8-char prefix lookup (chars 8–16 of `sk_live_<random>`) then bcrypt verify
- **Storage**: Embeddings only — raw images are never persisted
