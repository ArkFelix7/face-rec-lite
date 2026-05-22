from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "info"

    # Database
    database_url: str = "postgresql+asyncpg://faceapi:faceapi@localhost:5432/facedb"
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
