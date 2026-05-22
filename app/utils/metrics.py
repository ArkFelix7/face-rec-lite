"""Prometheus metrics for the face recognition API.

All metric objects are module-level singletons. Import them wherever you need
to record an observation:

    from app.utils.metrics import ENROLLMENT_COUNTER, ML_INFERENCE_DURATION
    ENROLLMENT_COUNTER.labels(result="success").inc()

The ``/v1/metrics`` endpoint (served by ``prometheus_client.make_asgi_app``)
exposes these metrics in the standard Prometheus text format.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# ---------------------------------------------------------------------------
# HTTP request metrics
# ---------------------------------------------------------------------------

REQUEST_COUNTER: Counter = Counter(
    "face_api_requests_total",
    "Total number of HTTP requests received by the API",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION: Histogram = Histogram(
    "face_api_request_duration_seconds",
    "End-to-end HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# ---------------------------------------------------------------------------
# Business-logic counters
# ---------------------------------------------------------------------------

ENROLLMENT_COUNTER: Counter = Counter(
    "face_api_enrollments_total",
    "Number of face enrollment attempts, labelled by outcome",
    ["result"],  # success | quality_rejected | duplicate_skipped | error
)

VERIFICATION_COUNTER: Counter = Counter(
    "face_api_verifications_total",
    "Number of face verification attempts, labelled by outcome",
    ["result"],  # match | no_match | error
)

# ---------------------------------------------------------------------------
# ML inference latency
# ---------------------------------------------------------------------------

ML_INFERENCE_DURATION: Histogram = Histogram(
    "face_api_ml_inference_duration_seconds",
    "Duration of ML inference operations in seconds",
    ["operation"],  # detect | embed
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
