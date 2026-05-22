# Face Recognition API — Product Requirements Document

**Version:** 1.0.0  
**Status:** Final — Ready for Implementation  
**Use Case:** User Authentication via Face (1:1 Verification + Enrollment)  
**Target Scale:** < 10,000 users, < 50 req/sec  
**Compute:** Cloud (CPU-first, GPU-ready)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Goals & Non-Goals](#2-goals--non-goals)
3. [System Architecture](#3-system-architecture)
4. [Tech Stack (Exact Versions)](#4-tech-stack-exact-versions)
5. [Project Structure](#5-project-structure)
6. [Data Models & Database Schema](#6-data-models--database-schema)
7. [Environment Configuration](#7-environment-configuration)
8. [API Specification](#8-api-specification)
9. [Business Logic](#9-business-logic)
10. [Error Handling](#10-error-handling)
11. [Middleware & Cross-Cutting Concerns](#11-middleware--cross-cutting-concerns)
12. [Docker & Local Development](#12-docker--local-development)
13. [Test Plan](#13-test-plan)
14. [Acceptance Criteria](#14-acceptance-criteria)
15. [Implementation Order](#15-implementation-order)

---

## 1. Project Overview

### What We Are Building

A REST API service that enables applications to:

1. **Enroll** — Store a user's face (generate a 512-dimensional ArcFace embedding and persist it in PostgreSQL).
2. **Verify** — Confirm that a live face image matches a previously enrolled user's face (1:1 cosine similarity comparison).
3. **Manage** — CRUD operations for users and their enrolled faces.

This is a **face authentication backend**, not a surveillance or 1:N identification system. Every verification request must supply a `user_id` — the system compares the submitted face against that specific user's stored embeddings only.

### What It Is Not

- Not a 1:N face search across all users (no nearest-neighbor search at query time).
- Not a video stream processor.
- Not an anti-spoofing system (liveness detection is out of scope for v1).
- Not a face detection-only service.

### Core Flow

```
ENROLLMENT:
  Client → POST /v1/users/{user_id}/faces
         → API validates image
         → Detect face (RetinaFace)
         → Quality check (blur, brightness, pose, size)
         → Generate 512-dim ArcFace embedding
         → Store embedding in PostgreSQL
         → Return face_id + quality metrics

VERIFICATION:
  Client → POST /v1/users/{user_id}/verify
         → API validates image
         → Detect face (RetinaFace)
         → Generate embedding for query image
         → Load user's stored embeddings from DB
         → Compute cosine similarity for each
         → Take max similarity score
         → Return match: true/false + confidence score
```

---

## 2. Goals & Non-Goals

### Goals

- [x] Enroll one or more face images per user
- [x] Verify a live face against a user's enrolled faces
- [x] Quality gate on enrollment (reject blurry, dark, extreme-angle, tiny faces)
- [x] Deduplication on enrollment (skip near-identical embeddings)
- [x] User management (create, read, delete users)
- [x] Face management (list, delete individual enrolled faces)
- [x] API key authentication
- [x] Per-key rate limiting
- [x] Structured JSON error responses with error codes
- [x] Health and readiness endpoints
- [x] Full test coverage (unit + integration + E2E)
- [x] Docker Compose for local development
- [x] GDPR-ready: cascade delete removes all embeddings when user is deleted
- [x] Configurable via environment variables only (no hardcoded values)

### Non-Goals (v1)

- [ ] Anti-spoofing / liveness detection
- [ ] 1:N face search (identify unknown person across all users)
- [ ] Batch enrollment endpoint
- [ ] Webhook callbacks
- [ ] Multi-tenancy / collections / namespaces
- [ ] Face image storage (we store embeddings only, never raw images)
- [ ] Admin UI / dashboard
- [ ] GPU-specific optimizations (Triton, TensorRT) — CPU-first design

---

## 3. System Architecture

### Component Diagram

```
┌──────────────────────────────────────────────────────────┐
│                        CLIENT                            │
│  (mobile app, web app, any HTTP client)                  │
└─────────────────────┬────────────────────────────────────┘
                      │ HTTPS (JSON)
                      ▼
┌──────────────────────────────────────────────────────────┐
│                  FastAPI Application                     │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  Auth       │  │  Rate Limit  │  │  Request Log   │  │
│  │  Middleware │  │  Middleware  │  │  Middleware     │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                   Routers                           │ │
│  │  /v1/users     /v1/users/{id}/faces    /v1/health   │ │
│  │  /v1/users/{id}/verify                /v1/metrics   │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌──────────────────────┐  ┌──────────────────────────┐  │
│  │   Face ML Service    │  │   Database Service       │  │
│  │                      │  │                          │  │
│  │  - Image decoder     │  │  - User CRUD             │  │
│  │  - Face detector     │  │  - Face embedding CRUD   │  │
│  │  - Quality scorer    │  │  - Similarity query      │  │
│  │  - ArcFace embedder  │  │  - pgvector extension    │  │
│  │  - Cosine similarity │  │                          │  │
│  └──────────────────────┘  └──────────────────────────┘  │
└────────────────────────────┬─────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌─────────────────────┐       ┌─────────────────────────┐
│   PostgreSQL 15     │       │   Redis 7               │
│   + pgvector ext    │       │   (rate limit counters) │
│                     │       │                         │
│   Tables:           │       │   Keys:                 │
│   - api_keys        │       │   - ratelimit:{key}     │
│   - users           │       │                         │
│   - faces           │       │                         │
└─────────────────────┘       └─────────────────────────┘
```

### Request Lifecycle

```
1. Request arrives at FastAPI
2. Auth middleware: extract Bearer token → lookup api_key in DB → attach org context
3. Rate limit middleware: check Redis counter for this api_key → 429 if exceeded
4. Router dispatches to endpoint handler
5. Pydantic model validates request body → 422 if invalid
6. Handler calls FaceMLService.process_image():
   a. Decode image (base64 or multipart)
   b. Detect faces (RetinaFace ONNX)
   c. Validate exactly 1 face found
   d. Score quality → reject if below threshold
   e. Align face (5-point affine warp)
   f. Generate embedding (ArcFace ONNX, 512-dim float32)
7. Handler calls DatabaseService for storage or retrieval
8. Return JSON response with standard envelope
```

---

## 4. Tech Stack (Exact Versions)

### Runtime

| Component | Package | Version |
|-----------|---------|---------|
| Python | — | 3.11.x |
| FastAPI | `fastapi` | 0.111.0 |
| ASGI Server | `uvicorn[standard]` | 0.29.0 |
| Validation | `pydantic` | 2.7.1 |
| Settings | `pydantic-settings` | 2.2.1 |

### ML / Vision

| Component | Package | Version |
|-----------|---------|---------|
| Face recognition | `insightface` | 0.7.3 |
| ONNX runtime | `onnxruntime` | 1.18.0 |
| OpenCV | `opencv-python-headless` | 4.9.0.80 |
| NumPy | `numpy` | 1.26.4 |
| Pillow | `Pillow` | 10.3.0 |
| SciPy | `scipy` | 1.13.0 |

### Database

| Component | Package | Version |
|-----------|---------|---------|
| PostgreSQL | — | 15.x |
| pgvector extension | — | 0.7.x |
| Async ORM | `sqlalchemy[asyncio]` | 2.0.30 |
| Async PG driver | `asyncpg` | 0.29.0 |
| pgvector Python | `pgvector` | 0.3.2 |
| Migrations | `alembic` | 1.13.1 |

### Cache / Rate Limiting

| Component | Package | Version |
|-----------|---------|---------|
| Redis | — | 7.x |
| Redis client | `redis[asyncio]` | 5.0.4 |

### Auth / Security

| Component | Package | Version |
|-----------|---------|---------|
| Password hashing | `passlib[bcrypt]` | 1.7.4 |
| Crypto | `python-multipart` | 0.0.9 |

### Observability

| Component | Package | Version |
|-----------|---------|---------|
| Structured logging | `structlog` | 24.1.0 |
| Prometheus metrics | `prometheus-client` | 0.20.0 |

### Testing

| Component | Package | Version |
|-----------|---------|---------|
| Test runner | `pytest` | 8.2.0 |
| Async test support | `pytest-asyncio` | 0.23.6 |
| HTTP test client | `httpx` | 0.27.0 |
| Test coverage | `pytest-cov` | 5.0.0 |
| Fixtures / mocking | `pytest-mock` | 3.14.0 |
| Factory fixtures | `factory-boy` | 3.3.0 |

### Dev Tools

| Component | Package | Version |
|-----------|---------|---------|
| Linter | `ruff` | 0.4.4 |
| Formatter | `black` | 24.4.2 |
| Type checker | `mypy` | 1.10.0 |

---

## 5. Project Structure

```
face-rec-lite/
├── PRD.md
├── README.md
├── pyproject.toml               # All dependencies + tool config
├── .env.example                 # Template for environment variables
├── .env                         # Local dev values (gitignored)
├── .gitignore
│
├── docker-compose.yml           # Full local dev stack
├── Dockerfile                   # Production-ready image
│
├── alembic.ini                  # Alembic config
├── migrations/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       ├── 0001_create_api_keys.py
│       ├── 0002_create_users.py
│       └── 0003_create_faces.py
│
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app factory, lifespan, routers
│   │
│   ├── config.py                # Settings (pydantic-settings, reads .env)
│   │
│   ├── dependencies.py          # FastAPI Depends() factories
│   │
│   ├── models/                  # SQLAlchemy ORM models
│   │   ├── __init__.py
│   │   ├── api_key.py
│   │   ├── user.py
│   │   └── face.py
│   │
│   ├── schemas/                 # Pydantic request/response schemas
│   │   ├── __init__.py
│   │   ├── common.py            # ErrorResponse, PaginatedResponse
│   │   ├── api_key.py
│   │   ├── user.py
│   │   └── face.py
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── health.py            # GET /v1/health, GET /v1/ready
│   │   ├── users.py             # User CRUD
│   │   ├── faces.py             # Face enrollment + management
│   │   └── verify.py            # Face verification
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── face_ml.py           # All ML operations (detect, embed, score)
│   │   ├── database.py          # All DB queries
│   │   └── rate_limiter.py      # Redis-backed rate limiting
│   │
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── auth.py              # API key extraction + validation
│   │   ├── rate_limit.py        # Rate limiting middleware
│   │   └── request_logger.py    # Structured request logging
│   │
│   └── utils/
│       ├── __init__.py
│       ├── image.py             # Image decoding, validation, resizing
│       └── metrics.py           # Prometheus counters/histograms
│
└── tests/
    ├── __init__.py
    ├── conftest.py              # Shared fixtures (app, db, client, ML mock)
    ├── fixtures/
    │   ├── faces/               # Sample face images for tests
    │   │   ├── person_a_1.jpg
    │   │   ├── person_a_2.jpg   # Different angle, same person
    │   │   ├── person_b_1.jpg   # Different person
    │   │   ├── no_face.jpg
    │   │   ├── multiple_faces.jpg
    │   │   └── blurry.jpg
    │   └── api_keys.py          # Test API key factory
    │
    ├── unit/
    │   ├── __init__.py
    │   ├── test_image_utils.py
    │   ├── test_face_ml.py
    │   └── test_rate_limiter.py
    │
    ├── integration/
    │   ├── __init__.py
    │   ├── test_users_api.py
    │   ├── test_faces_api.py
    │   └── test_verify_api.py
    │
    └── e2e/
        ├── __init__.py
        └── test_auth_flow.py    # Full enroll → verify → delete flow
```

---

## 6. Data Models & Database Schema

### 6.1 PostgreSQL Schema

Run in order via Alembic migrations.

```sql
-- Enable pgvector extension (requires pgvector installed)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─────────────────────────────────────────────────────────
-- TABLE: api_keys
-- Stores API keys for authentication.
-- Keys are hashed (bcrypt) before storage; the raw key
-- is returned once at creation and never stored in plaintext.
-- ─────────────────────────────────────────────────────────
CREATE TABLE api_keys (
    id          UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_hash    TEXT        NOT NULL UNIQUE,   -- bcrypt hash of the raw key
    key_prefix  VARCHAR(8)  NOT NULL,          -- first 8 chars for lookup display
    name        TEXT        NOT NULL,          -- human label e.g. "prod-app-1"
    is_active   BOOLEAN     NOT NULL DEFAULT TRUE,
    rate_limit  INTEGER     NOT NULL DEFAULT 100,  -- max requests per minute
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ
);

-- ─────────────────────────────────────────────────────────
-- TABLE: users
-- Each user can have multiple enrolled face embeddings.
-- external_id is set by the API caller (their own user ID).
-- ─────────────────────────────────────────────────────────
CREATE TABLE users (
    id           UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id  TEXT        NOT NULL UNIQUE,  -- caller-supplied user identifier
    display_name TEXT,                         -- optional human-readable name
    metadata     JSONB       NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_external_id ON users (external_id);

-- ─────────────────────────────────────────────────────────
-- TABLE: faces
-- Stores ArcFace embeddings (512-dim float32 vectors).
-- Raw face images are NEVER stored.
-- One user can have up to MAX_FACES_PER_USER embeddings.
-- ─────────────────────────────────────────────────────────
CREATE TABLE faces (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    embedding       vector(512) NOT NULL,          -- ArcFace 512-dim embedding
    image_hash      TEXT        NOT NULL,           -- SHA256 of decoded image bytes (dedup)
    quality_score   FLOAT       NOT NULL,           -- overall quality 0.0–1.0
    blur_score      FLOAT       NOT NULL,           -- Laplacian variance, normalized
    brightness      FLOAT       NOT NULL,           -- mean pixel value, normalized 0–1
    face_confidence FLOAT       NOT NULL,           -- RetinaFace detection confidence
    face_width_px   INTEGER     NOT NULL,           -- width of face bounding box
    face_height_px  INTEGER     NOT NULL,           -- height of face bounding box
    pitch_deg       FLOAT       NOT NULL,           -- head pose: up/down
    yaw_deg         FLOAT       NOT NULL,           -- head pose: left/right
    roll_deg        FLOAT       NOT NULL,           -- head pose: tilt
    label           TEXT,                           -- optional caller label e.g. "front"
    enrolled_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_faces_user_id ON faces (user_id);
CREATE INDEX idx_faces_image_hash ON faces (image_hash);

-- Cosine similarity index (for future 1:N if needed, low cost to add now)
CREATE INDEX idx_faces_embedding ON faces
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
```

### 6.2 SQLAlchemy ORM Models

**`app/models/api_key.py`**
```python
import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, Integer, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base

class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    key_hash: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(8), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    rate_limit: Mapped[int] = mapped_column(Integer, default=100, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

**`app/models/user.py`**
```python
import uuid
from datetime import datetime
from sqlalchemy import Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.models.base import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    external_id: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    faces: Mapped[list["Face"]] = relationship("Face", back_populates="user", cascade="all, delete-orphan")
```

**`app/models/face.py`**
```python
import uuid
from datetime import datetime
from sqlalchemy import Text, Float, Integer, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from app.models.base import Base

class Face(Base):
    __tablename__ = "faces"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(512), nullable=False)
    image_hash: Mapped[str] = mapped_column(Text, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    blur_score: Mapped[float] = mapped_column(Float, nullable=False)
    brightness: Mapped[float] = mapped_column(Float, nullable=False)
    face_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    face_width_px: Mapped[int] = mapped_column(Integer, nullable=False)
    face_height_px: Mapped[int] = mapped_column(Integer, nullable=False)
    pitch_deg: Mapped[float] = mapped_column(Float, nullable=False)
    yaw_deg: Mapped[float] = mapped_column(Float, nullable=False)
    roll_deg: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[str | None] = mapped_column(Text, nullable=True)
    enrolled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    user: Mapped["User"] = relationship("User", back_populates="faces")
```

---

## 7. Environment Configuration

### 7.1 `.env.example`

```dotenv
# ── Application ──────────────────────────────────────────
APP_ENV=development           # development | production
APP_HOST=0.0.0.0
APP_PORT=8000
APP_LOG_LEVEL=info            # debug | info | warning | error

# ── Database ─────────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://faceapi:faceapi@localhost:5432/facedb
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# ── Redis ────────────────────────────────────────────────
REDIS_URL=redis://localhost:6379/0

# ── ML Models ────────────────────────────────────────────
# Models are downloaded automatically by insightface on first run.
# Set to a writable directory. Models are ~200MB.
INSIGHTFACE_MODEL_DIR=./models
# Which InsightFace model pack to use
INSIGHTFACE_MODEL_NAME=buffalo_l   # buffalo_l (best) | buffalo_s (faster, smaller)
# Device: cpu | cuda (set cuda only if NVIDIA GPU + CUDA runtime available)
ML_DEVICE=cpu

# ── Face Quality Thresholds ──────────────────────────────
# Minimum quality score (0.0–1.0) to accept face at enrollment
MIN_QUALITY_SCORE=0.5
# Minimum face dimension (width AND height must exceed this in pixels)
MIN_FACE_SIZE_PX=80
# Maximum head pose deviation from frontal (degrees)
MAX_PITCH_DEG=30.0
MAX_YAW_DEG=35.0
MAX_ROLL_DEG=25.0

# ── Verification ─────────────────────────────────────────
# Cosine similarity threshold. Faces with score >= this are a match.
# Range: 0.0–1.0. Recommended: 0.50 (lenient) – 0.70 (strict)
DEFAULT_VERIFICATION_THRESHOLD=0.60

# ── Enrollment ───────────────────────────────────────────
# Max number of face embeddings per user
MAX_FACES_PER_USER=5
# Cosine similarity above which two embeddings are considered duplicates
DEDUP_THRESHOLD=0.95

# ── Rate Limiting ────────────────────────────────────────
# Default requests per minute per API key (overridable per key in DB)
DEFAULT_RATE_LIMIT_RPM=100

# ── Security ─────────────────────────────────────────────
# Salt rounds for bcrypt hashing of API keys
BCRYPT_ROUNDS=12
```

### 7.2 `app/config.py`

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Application
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "info"

    # Database
    database_url: str
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # ML
    insightface_model_dir: str = "./models"
    insightface_model_name: str = "buffalo_l"
    ml_device: str = "cpu"

    # Quality Thresholds
    min_quality_score: float = 0.5
    min_face_size_px: int = 80
    max_pitch_deg: float = 30.0
    max_yaw_deg: float = 35.0
    max_roll_deg: float = 25.0

    # Verification
    default_verification_threshold: float = 0.60

    # Enrollment
    max_faces_per_user: int = 5
    dedup_threshold: float = 0.95

    # Rate Limiting
    default_rate_limit_rpm: int = 100

    # Security
    bcrypt_rounds: int = 12

settings = Settings()
```

---

## 8. API Specification

### 8.1 Common Conventions

**Base URL:** `http://localhost:8000`

**Authentication:** All endpoints (except `/v1/health` and `/v1/ready`) require:
```
Authorization: Bearer <api_key>
```

**Request Content-Type:** `application/json` unless multipart (file upload endpoints).

**Response Envelope:** All responses follow this structure:
```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "request_id": "req_a1b2c3d4e5f6"
}
```
On error:
```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable description",
    "details": { ... }
  },
  "request_id": "req_a1b2c3d4e5f6"
}
```

**Image Input Format:** All image fields accept base64-encoded images with optional data URI prefix:
```
"data:image/jpeg;base64,/9j/4AAQSkZJRg..."
or just:
"/9j/4AAQSkZJRg..."
```
Supported formats: JPEG, PNG, WebP, BMP.  
Maximum raw image size: **10 MB**.  
Images are decoded server-side; original bytes are never stored.

---

### 8.2 Endpoint: Health & Readiness

#### `GET /v1/health`

No authentication required. Returns 200 if the server process is alive.

**Response 200:**
```json
{
  "success": true,
  "data": {
    "status": "ok",
    "version": "1.0.0"
  },
  "error": null,
  "request_id": "req_..."
}
```

---

#### `GET /v1/ready`

No authentication required. Returns 200 only if database and Redis connections are healthy. Returns 503 otherwise. Use this as the Kubernetes readiness probe.

**Response 200:**
```json
{
  "success": true,
  "data": {
    "status": "ready",
    "checks": {
      "database": "ok",
      "redis": "ok",
      "ml_model": "ok"
    }
  },
  "error": null,
  "request_id": "req_..."
}
```

**Response 503:**
```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "SERVICE_UNAVAILABLE",
    "message": "One or more dependencies are unhealthy",
    "details": {
      "database": "error: connection refused",
      "redis": "ok",
      "ml_model": "ok"
    }
  },
  "request_id": "req_..."
}
```

---

#### `GET /v1/metrics`

Returns Prometheus text-format metrics. No authentication required. Used by monitoring scrapers.

**Response 200:** Prometheus text format (Content-Type: `text/plain`)

Exposed metrics:
- `face_api_requests_total{method, endpoint, status_code}` — counter
- `face_api_request_duration_seconds{method, endpoint}` — histogram (buckets: 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
- `face_api_enrollments_total{result}` — counter (result: success, quality_rejected, duplicate_skipped, error)
- `face_api_verifications_total{result}` — counter (result: match, no_match, error)
- `face_api_ml_inference_duration_seconds{operation}` — histogram (operation: detect, embed)

---

### 8.3 Endpoint: User Management

#### `POST /v1/users`

Create a new user record. The `user_id` in the path of all subsequent calls must match `external_id` here.

**Request Body:**
```json
{
  "external_id": "user_abc123",
  "display_name": "Alice Smith",
  "metadata": {
    "department": "engineering",
    "employee_number": "E001"
  }
}
```

| Field | Type | Required | Constraints |
|-------|------|----------|------------|
| `external_id` | string | yes | 1–255 chars, must be unique across all users |
| `display_name` | string | no | 1–255 chars |
| `metadata` | object | no | Any JSON object, max depth 3, max 20 keys |

**Response 201:**
```json
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "external_id": "user_abc123",
    "display_name": "Alice Smith",
    "metadata": { "department": "engineering" },
    "face_count": 0,
    "created_at": "2025-05-22T10:30:00Z"
  },
  "error": null,
  "request_id": "req_..."
}
```

**Errors:**
- `409 CONFLICT` — `USER_ALREADY_EXISTS` if `external_id` is taken

---

#### `GET /v1/users/{user_id}`

Retrieve a user's profile and enrollment summary. `user_id` is the `external_id`.

**Response 200:**
```json
{
  "success": true,
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "external_id": "user_abc123",
    "display_name": "Alice Smith",
    "metadata": {},
    "face_count": 2,
    "created_at": "2025-05-22T10:30:00Z",
    "updated_at": "2025-05-22T10:30:00Z"
  },
  "error": null,
  "request_id": "req_..."
}
```

**Errors:**
- `404 NOT_FOUND` — `USER_NOT_FOUND`

---

#### `DELETE /v1/users/{user_id}`

Delete user and all associated face embeddings (cascade). This is the GDPR right-to-be-forgotten operation.

**Response 200:**
```json
{
  "success": true,
  "data": {
    "deleted_user_id": "user_abc123",
    "faces_deleted": 2,
    "deleted_at": "2025-05-22T10:30:00Z"
  },
  "error": null,
  "request_id": "req_..."
}
```

**Errors:**
- `404 NOT_FOUND` — `USER_NOT_FOUND`

---

### 8.4 Endpoint: Face Enrollment

#### `POST /v1/users/{user_id}/faces`

Enroll a new face image for the specified user. Runs the full quality + deduplication pipeline.

**Path Parameter:** `user_id` — the `external_id` of the user.

**Request Body (JSON):**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "label": "front",
  "quality_threshold": 0.5
}
```

| Field | Type | Required | Constraints |
|-------|------|----------|------------|
| `image` | string | yes | base64-encoded image, max 10MB decoded |
| `label` | string | no | 1–64 chars, e.g. "front", "left_profile" |
| `quality_threshold` | float | no | 0.0–1.0, overrides server default for this request |

**Response 201 (success):**
```json
{
  "success": true,
  "data": {
    "face_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "user_id": "user_abc123",
    "label": "front",
    "quality_metrics": {
      "overall_score": 0.87,
      "blur_score": 0.92,
      "brightness": 0.74,
      "face_confidence": 0.99,
      "face_size": { "width_px": 220, "height_px": 260 },
      "head_pose": { "pitch_deg": 3.1, "yaw_deg": -5.2, "roll_deg": 1.8 }
    },
    "bounding_box": { "x": 100, "y": 80, "width": 220, "height": 260 },
    "enrolled_at": "2025-05-22T10:30:00Z",
    "total_faces_for_user": 1
  },
  "error": null,
  "request_id": "req_..."
}
```

**Response 200 (duplicate skipped):**

When `image_hash` or embedding similarity >= `DEDUP_THRESHOLD` against this user's existing faces:
```json
{
  "success": true,
  "data": {
    "face_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "user_id": "user_abc123",
    "duplicate": true,
    "duplicate_of_face_id": "existing-face-uuid",
    "similarity": 0.98,
    "message": "This face image is too similar to an already enrolled face. Enrollment skipped.",
    "total_faces_for_user": 1
  },
  "error": null,
  "request_id": "req_..."
}
```

**Errors:**
- `404 NOT_FOUND` — `USER_NOT_FOUND`
- `400 BAD_REQUEST` — `INVALID_IMAGE` (corrupt, wrong format, too large)
- `400 BAD_REQUEST` — `NO_FACE_DETECTED` (0 faces found)
- `400 BAD_REQUEST` — `MULTIPLE_FACES` (> 1 face found, include count in details)
- `400 BAD_REQUEST` — `LOW_QUALITY` (overall quality below threshold, include failing metrics)
- `409 CONFLICT` — `MAX_FACES_REACHED` (user already has `MAX_FACES_PER_USER` enrolled)

**`LOW_QUALITY` Error Detail Example:**
```json
{
  "code": "LOW_QUALITY",
  "message": "Face quality score 0.31 is below the required threshold 0.50",
  "details": {
    "overall_score": 0.31,
    "threshold": 0.50,
    "failing_checks": [
      { "check": "blur_score", "value": 0.21, "minimum": 0.40, "reason": "Image is too blurry" },
      { "check": "face_size", "value": 60, "minimum": 80, "reason": "Face bounding box too small" }
    ]
  }
}
```

---

#### `GET /v1/users/{user_id}/faces`

List all enrolled faces for a user.

**Response 200:**
```json
{
  "success": true,
  "data": {
    "user_id": "user_abc123",
    "faces": [
      {
        "face_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        "label": "front",
        "quality_score": 0.87,
        "face_size": { "width_px": 220, "height_px": 260 },
        "enrolled_at": "2025-05-22T10:30:00Z"
      }
    ],
    "total": 1,
    "max_allowed": 5
  },
  "error": null,
  "request_id": "req_..."
}
```

---

#### `DELETE /v1/users/{user_id}/faces/{face_id}`

Delete a single enrolled face embedding.

**Response 200:**
```json
{
  "success": true,
  "data": {
    "deleted_face_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "user_id": "user_abc123",
    "remaining_faces": 0,
    "deleted_at": "2025-05-22T10:30:00Z"
  },
  "error": null,
  "request_id": "req_..."
}
```

**Errors:**
- `404 NOT_FOUND` — `USER_NOT_FOUND` or `FACE_NOT_FOUND`

---

### 8.5 Endpoint: Face Verification

#### `POST /v1/users/{user_id}/verify`

Compare a submitted face image against all enrolled faces for the given user. Returns a match decision plus the best similarity score found.

**Path Parameter:** `user_id` — the `external_id` of the user to verify against.

**Request Body:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "threshold": 0.60
}
```

| Field | Type | Required | Constraints |
|-------|------|----------|------------|
| `image` | string | yes | base64-encoded image, max 10MB |
| `threshold` | float | no | 0.0–1.0, overrides `DEFAULT_VERIFICATION_THRESHOLD` |

**Response 200 — Match:**
```json
{
  "success": true,
  "data": {
    "match": true,
    "confidence": 0.89,
    "threshold_used": 0.60,
    "best_matching_face_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "user_id": "user_abc123",
    "query_face_quality": {
      "overall_score": 0.81,
      "blur_score": 0.88,
      "brightness": 0.72,
      "face_confidence": 0.97
    },
    "all_scores": [
      { "face_id": "f47ac10b-...", "similarity": 0.89 },
      { "face_id": "a1b2c3d4-...", "similarity": 0.76 }
    ],
    "processing_time_ms": 215
  },
  "error": null,
  "request_id": "req_..."
}
```

**Response 200 — No Match:**
```json
{
  "success": true,
  "data": {
    "match": false,
    "confidence": 0.31,
    "threshold_used": 0.60,
    "best_matching_face_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "user_id": "user_abc123",
    "query_face_quality": { ... },
    "all_scores": [
      { "face_id": "f47ac10b-...", "similarity": 0.31 }
    ],
    "processing_time_ms": 198
  },
  "error": null,
  "request_id": "req_..."
}
```

**Errors:**
- `404 NOT_FOUND` — `USER_NOT_FOUND`
- `409 CONFLICT` — `USER_HAS_NO_FACES` (user exists but has 0 enrolled faces; code 409 chosen to distinguish from 404)
- `400 BAD_REQUEST` — `INVALID_IMAGE`
- `400 BAD_REQUEST` — `NO_FACE_DETECTED`
- `400 BAD_REQUEST` — `MULTIPLE_FACES`

> **Note:** Verification does NOT apply a quality gate by default. Even a lower-quality query image is processed to give the best possible match. The quality metrics are returned for the caller to decide if they want to ask the user to retake.

---

## 9. Business Logic

### 9.1 Image Decoding (`app/utils/image.py`)

```python
def decode_image(image_input: str | bytes) -> np.ndarray:
    """
    Accept:
      - base64 string (with or without data URI prefix)
      - raw bytes
    Return: BGR numpy array (OpenCV convention)
    Raise: InvalidImageError on failure
    """
    steps:
    1. If string: strip "data:image/...;base64," prefix if present
    2. base64.b64decode() → bytes
    3. Validate size <= 10MB
    4. np.frombuffer() → np.uint8 array
    5. cv2.imdecode(buf, cv2.IMREAD_COLOR) → BGR ndarray
    6. If result is None: raise InvalidImageError("corrupt or unsupported format")
    7. EXIF orientation correction (use PIL to read EXIF, rotate accordingly)
    8. Return BGR ndarray
```

### 9.2 Face Detection & Alignment (`app/services/face_ml.py`)

```python
class FaceMLService:
    """
    Wraps InsightFace FaceAnalysis model.
    Initialized once at app startup via lifespan context.
    Thread-safe for concurrent async use.
    """

    def __init__(self, settings: Settings):
        self.app = FaceAnalysis(
            name=settings.insightface_model_name,
            root=settings.insightface_model_dir,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                      if settings.ml_device == "cuda"
                      else ["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0 if settings.ml_device == "cuda" else -1, det_size=(640, 640))

    def detect_and_embed(self, image_bgr: np.ndarray) -> FaceMLResult:
        """
        Returns: FaceMLResult with faces list.
        Each face has: bbox, kps (landmarks), det_score, embedding (512-dim).
        Raises: nothing — caller checks len(faces).
        """
        faces = self.app.get(image_bgr)
        return FaceMLResult(faces=faces, image_shape=image_bgr.shape)
```

**InsightFace `buffalo_l` model pack includes:**
- Detection: RetinaFace (det_10g.onnx)
- Recognition/Embedding: ArcFace (w600k_r50.onnx) → 512-dim
- These are downloaded automatically by InsightFace on first use

### 9.3 Quality Scoring (`app/services/face_ml.py`)

Quality is computed as a weighted average of 5 components. All components are normalized to [0.0, 1.0].

```python
QUALITY_WEIGHTS = {
    "blur":            0.30,  # Most important — blurry faces can't be recognized
    "brightness":      0.20,  # Dark/overexposed significantly hurts accuracy
    "face_confidence": 0.25,  # Detector's own confidence in the detection
    "face_size":       0.15,  # Tiny faces → low resolution → poor embedding
    "head_pose":       0.10,  # Extreme angles reduce accuracy
}

def compute_quality(image_bgr: np.ndarray, face: Face, settings: Settings) -> QualityMetrics:

    # ── Blur (Laplacian variance) ──────────────────────────
    face_crop = crop_face(image_bgr, face.bbox, margin=0.1)
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize: 0 = extremely blurry, 1 = sharp
    # Empirically: < 50 = blurry, 100–500 = good, > 500 = very sharp
    blur_score = min(lap_var / 500.0, 1.0)

    # ── Brightness ────────────────────────────────────────
    # Mean pixel value in [0, 255], normalize to [0, 1]
    mean_brightness = face_crop.mean() / 255.0
    # Ideal range: 0.3–0.8. Penalize extremes.
    if 0.3 <= mean_brightness <= 0.8:
        brightness_score = 1.0
    elif mean_brightness < 0.3:
        brightness_score = mean_brightness / 0.3
    else:
        brightness_score = (1.0 - mean_brightness) / 0.2

    # ── Face Confidence ───────────────────────────────────
    confidence_score = float(face.det_score)  # Already 0–1 from RetinaFace

    # ── Face Size ─────────────────────────────────────────
    x1, y1, x2, y2 = face.bbox
    face_width = x2 - x1
    face_height = y2 - y1
    min_dim = min(face_width, face_height)
    # Below min_face_size_px = 0, above 400px = 1.0
    size_score = min(min_dim / 400.0, 1.0)

    # ── Head Pose ─────────────────────────────────────────
    # InsightFace buffalo_l provides pose via face.pose attribute [pitch, yaw, roll]
    pitch, yaw, roll = face.pose
    pitch_ok = 1.0 - min(abs(pitch) / settings.max_pitch_deg, 1.0)
    yaw_ok   = 1.0 - min(abs(yaw)   / settings.max_yaw_deg,   1.0)
    roll_ok  = 1.0 - min(abs(roll)  / settings.max_roll_deg,  1.0)
    pose_score = (pitch_ok + yaw_ok + roll_ok) / 3.0

    # ── Overall Score ─────────────────────────────────────
    overall = (
        QUALITY_WEIGHTS["blur"]            * blur_score +
        QUALITY_WEIGHTS["brightness"]      * brightness_score +
        QUALITY_WEIGHTS["face_confidence"] * confidence_score +
        QUALITY_WEIGHTS["face_size"]       * size_score +
        QUALITY_WEIGHTS["head_pose"]       * pose_score
    )

    return QualityMetrics(
        overall_score=overall,
        blur_score=blur_score,
        brightness=mean_brightness,
        face_confidence=confidence_score,
        face_width_px=int(face_width),
        face_height_px=int(face_height),
        pitch_deg=float(pitch),
        yaw_deg=float(yaw),
        roll_deg=float(roll),
    )
```

**Quality Gate Rules (enrollment only):**

The face is rejected if ANY of the following hard rules fail, regardless of overall score:
1. `face_width_px < MIN_FACE_SIZE_PX` OR `face_height_px < MIN_FACE_SIZE_PX`
2. `abs(yaw_deg) > MAX_YAW_DEG`
3. `abs(pitch_deg) > MAX_PITCH_DEG`
4. `abs(roll_deg) > MAX_ROLL_DEG`

Then, if all hard rules pass, reject if `overall_score < MIN_QUALITY_SCORE`.

Return `LOW_QUALITY` with which specific checks failed.

### 9.4 Embedding Generation

```python
def get_embedding(face: Face) -> np.ndarray:
    """
    Returns normalized 512-dim float32 ArcFace embedding.
    Normalization: L2-normalize so cosine similarity == dot product.
    InsightFace buffalo_l returns the embedding directly from face.embedding.
    """
    emb = face.embedding.astype(np.float32)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb
```

### 9.5 Deduplication Check (Enrollment)

```python
async def check_duplicate(
    new_embedding: np.ndarray,
    user_id: str,
    db: DatabaseService,
    threshold: float,
) -> DuplicateCheckResult:
    """
    Load all stored embeddings for this user.
    Compute cosine similarity between new_embedding and each stored embedding.
    If max similarity >= threshold: return duplicate result.
    """
    stored_faces = await db.get_user_faces_with_embeddings(user_id)
    if not stored_faces:
        return DuplicateCheckResult(is_duplicate=False)

    similarities = [
        cosine_similarity(new_embedding, face.embedding)
        for face in stored_faces
    ]
    max_sim = max(similarities)
    if max_sim >= threshold:
        best_match_idx = similarities.index(max_sim)
        return DuplicateCheckResult(
            is_duplicate=True,
            existing_face_id=stored_faces[best_match_idx].id,
            similarity=max_sim,
        )
    return DuplicateCheckResult(is_duplicate=False)
```

### 9.6 Image Hash Deduplication

Before running ML at all, compute `hashlib.sha256(raw_image_bytes).hexdigest()` and check if this exact hash exists in the `faces` table for this user. If it does, return duplicate response immediately without running inference. This prevents reprocessing exact duplicate uploads.

### 9.7 Cosine Similarity

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Both embeddings must be L2-normalized (unit vectors).
    For L2-normalized vectors: cosine_similarity = dot(a, b).
    Result: -1.0 to 1.0, where 1.0 = identical.
    """
    return float(np.dot(a, b))
```

### 9.8 Verification Logic

```python
async def verify_face(
    query_embedding: np.ndarray,
    user_id: str,
    threshold: float,
    db: DatabaseService,
) -> VerificationResult:
    """
    1. Load all stored embeddings for user_id.
    2. Compute cosine similarity for each.
    3. Take the maximum (best match).
    4. If max >= threshold: match=True.
    5. Return match, confidence, best_face_id, all_scores.
    """
    stored_faces = await db.get_user_faces_with_embeddings(user_id)

    all_scores = [
        {"face_id": str(face.id), "similarity": cosine_similarity(query_embedding, face.embedding)}
        for face in stored_faces
    ]
    all_scores.sort(key=lambda x: x["similarity"], reverse=True)

    best = all_scores[0]
    return VerificationResult(
        match=best["similarity"] >= threshold,
        confidence=best["similarity"],
        best_matching_face_id=best["face_id"],
        threshold_used=threshold,
        all_scores=all_scores,
    )
```

---

## 10. Error Handling

### 10.1 Error Code Registry

| HTTP | Code | Trigger |
|------|------|---------|
| 400 | `INVALID_IMAGE` | Image cannot be decoded (corrupt, wrong format) |
| 400 | `IMAGE_TOO_LARGE` | Decoded image > 10MB |
| 400 | `NO_FACE_DETECTED` | RetinaFace finds 0 faces |
| 400 | `MULTIPLE_FACES` | RetinaFace finds > 1 face |
| 400 | `LOW_QUALITY` | Quality gate fails (include failing checks in details) |
| 400 | `INVALID_THRESHOLD` | Threshold value outside 0.0–1.0 |
| 401 | `UNAUTHORIZED` | Missing or malformed Authorization header |
| 401 | `INVALID_API_KEY` | API key not found or is_active=false |
| 404 | `USER_NOT_FOUND` | No user with given external_id |
| 404 | `FACE_NOT_FOUND` | No face with given face_id for this user |
| 409 | `USER_ALREADY_EXISTS` | external_id already registered |
| 409 | `MAX_FACES_REACHED` | User already has MAX_FACES_PER_USER enrolled |
| 409 | `USER_HAS_NO_FACES` | Verification called but user has 0 enrolled faces |
| 422 | `VALIDATION_ERROR` | Pydantic validation failure (malformed JSON, wrong types) |
| 429 | `RATE_LIMIT_EXCEEDED` | API key exceeded rate limit |
| 500 | `INTERNAL_ERROR` | Unexpected exception |
| 503 | `SERVICE_UNAVAILABLE` | DB or Redis unreachable |

### 10.2 Global Exception Handler

Register a FastAPI exception handler that catches all unhandled exceptions and returns a structured `INTERNAL_ERROR` response with a `request_id`. Never leak stack traces in production. Always log the full exception with the `request_id` for traceability.

### 10.3 Request ID

Every request gets a unique `request_id` generated at middleware entry (`req_` + 12-char hex). It is:
- Added to response body JSON
- Added to response header `X-Request-ID`
- Included in every log line for that request

---

## 11. Middleware & Cross-Cutting Concerns

### 11.1 Auth Middleware (`app/middleware/auth.py`)

```
Endpoints excluded from auth: /v1/health, /v1/ready, /v1/metrics

For all other endpoints:
1. Extract "Authorization" header
2. Must be "Bearer <token>" format → 401 UNAUTHORIZED if missing/malformed
3. Hash the token with bcrypt and compare against api_keys table
4. IMPORTANT: Lookup by key_prefix (first 8 chars) first to reduce bcrypt comparisons
   - SELECT * FROM api_keys WHERE key_prefix = :prefix AND is_active = true
   - Then bcrypt.verify(raw_key, row.key_hash)
5. If not found or inactive → 401 INVALID_API_KEY
6. UPDATE api_keys SET last_used_at = NOW() WHERE id = :id
7. Attach api_key record to request.state.api_key
```

### 11.2 Rate Limit Middleware (`app/middleware/rate_limit.py`)

Uses Redis with sliding window counter (fixed window per minute is acceptable for v1).

```
Algorithm (fixed window, per minute):
1. key = f"ratelimit:{api_key.id}:{current_minute_unix}"
2. count = await redis.incr(key)
3. If count == 1: await redis.expire(key, 60)
4. If count > api_key.rate_limit:
   → Return 429 RATE_LIMIT_EXCEEDED
   → Header: Retry-After: (60 - seconds_into_current_minute)
   → Header: X-RateLimit-Limit: {api_key.rate_limit}
   → Header: X-RateLimit-Remaining: 0
5. Else: proceed, add headers:
   → X-RateLimit-Limit: {api_key.rate_limit}
   → X-RateLimit-Remaining: {api_key.rate_limit - count}
```

### 11.3 Request Logger Middleware

Log every request at completion with structlog:
```json
{
  "event": "request_completed",
  "request_id": "req_a1b2c3d4e5f6",
  "method": "POST",
  "path": "/v1/users/user_abc123/verify",
  "status_code": 200,
  "duration_ms": 215,
  "api_key_prefix": "sk_live_",
  "user_agent": "..."
}
```

Log at `WARNING` level for 4xx, `ERROR` for 5xx, `INFO` for 2xx.

---

## 12. Docker & Local Development

### 12.1 `docker-compose.yml`

```yaml
version: "3.9"

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://faceapi:faceapi@postgres:5432/facedb
      - REDIS_URL=redis://redis:6379/0
      - ML_DEVICE=cpu
      - INSIGHTFACE_MODEL_DIR=/app/models
      - APP_ENV=development
      - APP_LOG_LEVEL=debug
    volumes:
      - ./models:/app/models        # Persist downloaded ML models
      - .:/app                      # Hot reload in dev
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_USER: faceapi
      POSTGRES_PASSWORD: faceapi
      POSTGRES_DB: facedb
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U faceapi -d facedb"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  postgres_data:
```

### 12.2 `Dockerfile`

```dockerfile
FROM python:3.11-slim

# System deps for OpenCV headless + InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

# Pre-download InsightFace models so container starts fast
# buffalo_l is ~200MB. On first build this will download.
# On subsequent builds it's cached in the Docker layer.
ENV INSIGHTFACE_MODEL_DIR=/app/models
RUN python -c "from insightface.app import FaceAnalysis; \
    app = FaceAnalysis(name='buffalo_l', root='/app/models', \
    providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1)"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### 12.3 `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "face-rec-lite"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi==0.111.0",
    "uvicorn[standard]==0.29.0",
    "pydantic==2.7.1",
    "pydantic-settings==2.2.1",
    "insightface==0.7.3",
    "onnxruntime==1.18.0",
    "opencv-python-headless==4.9.0.80",
    "numpy==1.26.4",
    "Pillow==10.3.0",
    "scipy==1.13.0",
    "sqlalchemy[asyncio]==2.0.30",
    "asyncpg==0.29.0",
    "pgvector==0.3.2",
    "alembic==1.13.1",
    "redis[asyncio]==5.0.4",
    "passlib[bcrypt]==1.7.4",
    "python-multipart==0.0.9",
    "structlog==24.1.0",
    "prometheus-client==0.20.0",
]

[project.optional-dependencies]
dev = [
    "pytest==8.2.0",
    "pytest-asyncio==0.23.6",
    "httpx==0.27.0",
    "pytest-cov==5.0.0",
    "pytest-mock==3.14.0",
    "factory-boy==3.3.0",
    "ruff==0.4.4",
    "black==24.4.2",
    "mypy==1.10.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "--cov=app --cov-report=term-missing --cov-fail-under=80"

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "UP", "ANN"]

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
```

### 12.4 Local Setup Commands

```bash
# 1. Clone and set up environment
git clone <repo>
cd face-rec-lite
cp .env.example .env

# 2. Start infrastructure (PostgreSQL + Redis)
docker compose up postgres redis -d

# 3. Install Python deps
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 4. Run migrations
alembic upgrade head

# 5. Create a test API key (run this script once)
python scripts/create_api_key.py --name "dev-key"
# Outputs: <your-api-key-shown-once>
# Save this — it's shown once.

# 6. Start API server
uvicorn app.main:app --reload --port 8000

# 7. Verify it's running
curl http://localhost:8000/v1/health

# OR: Start everything with Docker Compose
docker compose up --build
```

### 12.5 `scripts/create_api_key.py`

This script must exist and do the following:
1. Accept `--name <label>` CLI argument
2. Generate a random 32-byte key: `secrets.token_hex(32)` → `sk_live_<hex>`
3. Hash with bcrypt
4. Insert into `api_keys` table
5. Print the raw key **once** to stdout — never stored in DB
6. Print confirmation with `key_prefix`

---

## 13. Test Plan

### 13.1 Test Fixtures Required

**`tests/fixtures/faces/`** — Provide real JPEG test images:

| Filename | Contents | Purpose |
|----------|----------|---------|
| `person_a_frontal.jpg` | Clear frontal face, 640×480 | Happy-path enrollment + verify |
| `person_a_slight_angle.jpg` | Same person, slight left turn | Verify across pose change |
| `person_a_different_lighting.jpg` | Same person, different lighting | Verify across conditions |
| `person_b_frontal.jpg` | Different person, frontal | Should NOT match person_a |
| `no_face.jpg` | Landscape photo, no humans | `NO_FACE_DETECTED` tests |
| `multiple_faces.jpg` | Image with 2+ people | `MULTIPLE_FACES` tests |
| `blurry_face.jpg` | Intentionally blurry face crop | `LOW_QUALITY` tests |
| `tiny_face.jpg` | Face occupying < 50px | `LOW_QUALITY` tests |
| `extreme_yaw.jpg` | Face turned > 35° sideways | Fails pose check |

> **Implementation note:** If real photos are unavailable, tests that require actual ML inference must be marked `@pytest.mark.integration` and can generate synthetic embeddings for unit/schema tests. But at least one E2E test must use real images to validate the actual ML pipeline.

### 13.2 `tests/conftest.py` — Key Fixtures

```python
@pytest.fixture(scope="session")
def settings() -> Settings:
    """Override settings for test environment."""
    return Settings(
        database_url="postgresql+asyncpg://faceapi:faceapi@localhost:5432/facedb_test",
        redis_url="redis://localhost:6379/1",  # DB 1 for tests
        min_quality_score=0.3,  # Lower threshold for test images
        ml_device="cpu",
    )

@pytest.fixture(scope="session")
async def db_engine(settings):
    """Create test DB, run migrations, yield, drop."""
    engine = create_async_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest.fixture
async def db_session(db_engine):
    """Provide a rolled-back DB session per test."""
    async with AsyncSession(db_engine) as session:
        async with session.begin():
            yield session
            await session.rollback()

@pytest.fixture(scope="session")
def face_ml_service(settings) -> FaceMLService:
    """Real ML service — loaded once for entire test session."""
    return FaceMLService(settings)

@pytest.fixture
def mock_face_ml(mocker) -> MagicMock:
    """Mocked ML service for unit tests that don't need real inference."""
    mock = mocker.MagicMock(spec=FaceMLService)
    mock.detect_and_embed.return_value = FakeMLResult(
        embedding=np.random.randn(512).astype(np.float32),
        quality=QualityMetrics(overall_score=0.85, blur_score=0.9, ...)
    )
    return mock

@pytest.fixture
async def test_client(settings, db_session, mock_face_ml):
    """AsyncClient for unit/integration tests with mocked ML."""
    app = create_app(settings=settings, face_ml=mock_face_ml)
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
async def api_key(db_session) -> tuple[str, ApiKey]:
    """Create a test API key, return (raw_key, db_record)."""
    raw_key = "sk_live_" + secrets.token_hex(16)
    key_hash = bcrypt.hash(raw_key)
    record = ApiKey(key_hash=key_hash, key_prefix=raw_key[:8], name="test-key")
    db_session.add(record)
    await db_session.flush()
    return raw_key, record

@pytest.fixture
async def auth_headers(api_key) -> dict:
    raw_key, _ = api_key
    return {"Authorization": f"Bearer {raw_key}"}

@pytest.fixture
def image_base64(request) -> str:
    """Load a test image fixture by filename and return base64 string."""
    path = Path(__file__).parent / "fixtures" / "faces" / request.param
    return base64.b64encode(path.read_bytes()).decode()
```

### 13.3 Unit Tests

**`tests/unit/test_image_utils.py`**

| Test | Description |
|------|-------------|
| `test_decode_base64_with_prefix` | Data URI prefix stripped correctly |
| `test_decode_base64_without_prefix` | Plain base64 also works |
| `test_decode_too_large` | Raises `InvalidImageError` when > 10MB |
| `test_decode_corrupt_bytes` | Raises `InvalidImageError` |
| `test_decode_unsupported_format` | GIF raises `InvalidImageError` |
| `test_exif_orientation_corrected` | Portrait JPEG with EXIF rotation is corrected |
| `test_compute_image_hash_deterministic` | Same bytes → same SHA256 |

**`tests/unit/test_face_ml.py`**

| Test | Description |
|------|-------------|
| `test_quality_blur_component` | Sharp image gets blur_score > 0.7 |
| `test_quality_blur_blurry_image` | Blurry image gets blur_score < 0.4 |
| `test_quality_brightness_dark` | Dark image gets brightness_score < 0.5 |
| `test_quality_brightness_bright` | Overexposed image gets penalized |
| `test_quality_tiny_face` | Face < 80px gets size_score < 0.3 |
| `test_quality_overall_weighted` | Check weighted sum calculation is correct |
| `test_cosine_similarity_identical` | Identical vectors → 1.0 |
| `test_cosine_similarity_orthogonal` | Orthogonal vectors → 0.0 |
| `test_cosine_similarity_opposite` | Opposite vectors → -1.0 |
| `test_embedding_normalized` | Output of `get_embedding()` has L2 norm ≈ 1.0 |

**`tests/unit/test_rate_limiter.py`**

| Test | Description |
|------|-------------|
| `test_first_request_allowed` | First request within limit passes |
| `test_at_limit_allowed` | Request exactly at limit passes |
| `test_over_limit_blocked` | Request over limit returns 429 |
| `test_new_minute_resets_counter` | Counter resets at next minute window |
| `test_rate_limit_headers_present` | Response includes X-RateLimit-* headers |

### 13.4 Integration Tests

All integration tests use a real test database (PostgreSQL) and real Redis, but mock the ML service.

**`tests/integration/test_users_api.py`**

| Test | Description |
|------|-------------|
| `test_create_user_success` | 201, correct response shape |
| `test_create_user_duplicate_external_id` | 409 USER_ALREADY_EXISTS |
| `test_create_user_no_auth` | 401 UNAUTHORIZED |
| `test_create_user_invalid_api_key` | 401 INVALID_API_KEY |
| `test_get_user_success` | 200, matches created user |
| `test_get_user_not_found` | 404 USER_NOT_FOUND |
| `test_delete_user_success` | 200, faces_deleted count correct |
| `test_delete_user_not_found` | 404 USER_NOT_FOUND |
| `test_delete_user_cascades_faces` | After delete, faces are gone |

**`tests/integration/test_faces_api.py`**

| Test | Description |
|------|-------------|
| `test_enroll_success` | 201, returns face_id + quality_metrics |
| `test_enroll_no_face` | 400 NO_FACE_DETECTED (ML mock returns 0 faces) |
| `test_enroll_multiple_faces` | 400 MULTIPLE_FACES (ML mock returns 2 faces) |
| `test_enroll_low_quality` | 400 LOW_QUALITY with failing checks in details |
| `test_enroll_duplicate_by_hash` | 200 with duplicate=true (same image twice) |
| `test_enroll_duplicate_by_embedding` | 200 with duplicate=true (similar embeddings) |
| `test_enroll_max_faces_reached` | 409 MAX_FACES_REACHED after N enrollments |
| `test_enroll_user_not_found` | 404 USER_NOT_FOUND |
| `test_list_faces_empty` | 200, empty list |
| `test_list_faces_populated` | 200, correct face records |
| `test_delete_face_success` | 200, face removed |
| `test_delete_face_not_found` | 404 FACE_NOT_FOUND |
| `test_enroll_invalid_base64` | 400 INVALID_IMAGE |
| `test_enroll_corrupt_image` | 400 INVALID_IMAGE |
| `test_enroll_with_label` | Label is stored and returned |

**`tests/integration/test_verify_api.py`**

| Test | Description |
|------|-------------|
| `test_verify_match` | ML returns embedding similar to enrolled → match=true |
| `test_verify_no_match` | ML returns dissimilar embedding → match=false |
| `test_verify_user_no_faces` | 409 USER_HAS_NO_FACES |
| `test_verify_user_not_found` | 404 USER_NOT_FOUND |
| `test_verify_no_face_in_image` | 400 NO_FACE_DETECTED |
| `test_verify_multiple_faces` | 400 MULTIPLE_FACES |
| `test_verify_custom_threshold_strict` | threshold=0.99 → no_match even for same embedding |
| `test_verify_custom_threshold_lenient` | threshold=0.01 → match even for dissimilar |
| `test_verify_returns_all_scores` | all_scores array contains one entry per enrolled face |
| `test_verify_best_score_selected` | Returns highest score when multiple enrolled faces |
| `test_verify_processing_time_present` | processing_time_ms in response |

### 13.5 End-to-End Tests

**`tests/e2e/test_auth_flow.py`**

Uses **real ML inference** (no mocking). Requires test face images in `tests/fixtures/faces/`.

| Test | Description | Pass Criterion |
|------|-------------|---------------|
| `test_full_enrollment_and_verify_match` | Enroll person_a_frontal.jpg, verify with person_a_slight_angle.jpg | match=true, confidence >= 0.60 |
| `test_verify_different_person_no_match` | Enroll person_a, verify with person_b | match=false |
| `test_multiple_enrollments_verify_picks_best` | Enroll 2 images of person_a, verify → picks higher score | match=true |
| `test_delete_user_then_verify_fails` | Enroll, delete user, verify → 404 | 404 USER_NOT_FOUND |
| `test_delete_face_then_verify_with_remaining` | Enroll 2 faces, delete 1, verify still works with remaining | match=true |
| `test_enroll_blurry_rejected` | Enroll blurry.jpg → rejected | 400 LOW_QUALITY |
| `test_enroll_no_face_rejected` | Enroll no_face.jpg → rejected | 400 NO_FACE_DETECTED |
| `test_rate_limiting` | Exceed rate limit → eventually get 429 | 429 RATE_LIMIT_EXCEEDED |

### 13.6 Running Tests

```bash
# All tests (requires running Postgres + Redis)
pytest

# Unit tests only (no external services)
pytest tests/unit/

# Integration tests
pytest tests/integration/

# E2E tests (real ML, needs images in fixtures/)
pytest tests/e2e/

# With coverage report
pytest --cov=app --cov-report=html

# Single test
pytest tests/integration/test_verify_api.py::test_verify_match -v

# Lint and type check
ruff check .
black --check .
mypy app/
```

---

## 14. Acceptance Criteria

The implementation is complete when ALL of the following pass:

### Functional

- [ ] `POST /v1/users` creates a user; duplicate `external_id` returns 409
- [ ] `POST /v1/users/{id}/faces` with a valid frontal face returns 201 with face_id and quality metrics
- [ ] `POST /v1/users/{id}/faces` with a blurry image returns 400 LOW_QUALITY with failing checks listed
- [ ] `POST /v1/users/{id}/faces` with no face returns 400 NO_FACE_DETECTED
- [ ] `POST /v1/users/{id}/faces` with same image twice returns 200 with duplicate=true
- [ ] `POST /v1/users/{id}/faces` rejects when user already has MAX_FACES_PER_USER faces
- [ ] `POST /v1/users/{id}/verify` returns match=true when same person's face is submitted
- [ ] `POST /v1/users/{id}/verify` returns match=false when a different person's face is submitted
- [ ] `POST /v1/users/{id}/verify` returns 409 USER_HAS_NO_FACES when user has no enrolled faces
- [ ] `DELETE /v1/users/{id}` removes user and all face embeddings from DB
- [ ] `GET /v1/ready` returns 503 when PostgreSQL is unreachable
- [ ] API key not in DB returns 401 INVALID_API_KEY
- [ ] Requests exceeding rate limit return 429 with Retry-After header
- [ ] All responses include `request_id` field

### Non-Functional

- [ ] `pytest` passes with ≥ 80% code coverage
- [ ] Verify response time ≤ 500ms on CPU (warm model) for a single-face image
- [ ] `docker compose up` starts all services and `/v1/ready` returns 200 within 60 seconds
- [ ] Alembic migrations run cleanly on fresh DB: `alembic upgrade head`
- [ ] `mypy app/` exits 0 (no type errors)
- [ ] `ruff check .` exits 0 (no lint errors)
- [ ] No raw face images are written to disk or database at any point

### Security

- [ ] API keys are stored as bcrypt hashes — raw key is never in DB
- [ ] SQL is parameterized (no f-string query construction)
- [ ] Stack traces are not included in error responses when `APP_ENV=production`

---

## 15. Implementation Order

Follow this order strictly. Each step is independently testable before moving to the next.

```
Step 1 — Project scaffold
  - pyproject.toml, directory structure, .env.example
  - app/config.py with Settings
  - app/main.py with lifespan (empty, no routes yet)
  - docker-compose.yml + Dockerfile
  - alembic.ini + migrations/env.py
  - Verify: docker compose up && curl /v1/health returns 200

Step 2 — Database layer
  - All SQLAlchemy models (api_key, user, face)
  - All Alembic migrations (3 migrations)
  - app/services/database.py with all DB query methods
  - Verify: alembic upgrade head succeeds on fresh DB

Step 3 — ML pipeline
  - app/utils/image.py (decode, hash, EXIF fix)
  - app/services/face_ml.py (FaceMLService, quality scoring, embedding)
  - Unit tests: tests/unit/test_image_utils.py, tests/unit/test_face_ml.py
  - Verify: unit tests pass

Step 4 — Auth + Rate Limiting
  - app/middleware/auth.py
  - app/services/rate_limiter.py
  - app/middleware/rate_limit.py
  - scripts/create_api_key.py
  - Unit tests: tests/unit/test_rate_limiter.py
  - Verify: unauthenticated request to any protected endpoint returns 401

Step 5 — User management API
  - app/schemas/user.py (request + response Pydantic models)
  - app/routers/users.py
  - Integration tests: tests/integration/test_users_api.py
  - Verify: all user API integration tests pass

Step 6 — Face enrollment API
  - app/schemas/face.py
  - app/routers/faces.py
  - Integration tests: tests/integration/test_faces_api.py
  - Verify: all face enrollment integration tests pass

Step 7 — Verification API
  - app/routers/verify.py
  - Integration tests: tests/integration/test_verify_api.py
  - Verify: all verify integration tests pass

Step 8 — Observability
  - app/utils/metrics.py (Prometheus counters/histograms)
  - app/middleware/request_logger.py
  - GET /v1/metrics endpoint
  - Verify: /v1/metrics returns Prometheus text format

Step 9 — E2E tests + final polish
  - tests/e2e/test_auth_flow.py (requires real face images)
  - Final mypy + ruff pass
  - Coverage report ≥ 80%
  - README with quickstart instructions
  - Verify: all acceptance criteria checked
```

---

## Appendix A: Threshold Tuning Guide

When deploying, calibrate the verification threshold against your actual user photos:

| Threshold | False Accept Rate | False Reject Rate | Use Case |
|-----------|-----------------|-----------------|---------|
| 0.45 | ~3% | 0.2% | Low-security, convenience-first |
| 0.55 | ~1% | 0.5% | Typical consumer app |
| **0.60** | **~0.5%** | **1%** | **Default — balanced** |
| 0.65 | ~0.3% | 2% | Moderate security |
| 0.70 | ~0.1% | 4% | High security |
| 0.75 | ~0.05% | 8% | Very high security |

## Appendix B: Troubleshooting Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `No module named 'cv2'` | `opencv-python` installed instead of headless | `pip install opencv-python-headless` |
| InsightFace download fails | No internet access in container | Pre-download models in Dockerfile |
| `pgvector` not found | Extension not enabled | Run `CREATE EXTENSION vector;` in DB |
| Very slow first request | Cold model load (~10s) | Pre-warm in lifespan startup hook |
| `CUDA out of memory` | GPU batch too large | Reduce `det_size` or set `ML_DEVICE=cpu` |
| Low recall (many false rejects) | Threshold too high | Lower threshold or add more enrolled angles |
