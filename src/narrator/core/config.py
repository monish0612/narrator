"""Central configuration (ARCHITECTURE.md section 18).

Every tunable lives here. Settings are loaded once at import and validated
fail-fast with explicit, one-line error messages (GROUND RULE 5 / 10). Pipeline
code must never read os.environ directly - it imports ``settings`` from here.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

StorageBackend = Literal["local", "gdrive"]
ProcessorName = Literal["explainer", "verbatim"]
NonEnglishPolicy = Literal["skip_warn", "read_anyway"]


class Settings(BaseSettings):
    """Fully validated runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Auth (required) -----------------------------------------------------
    api_keys: str = Field(..., alias="API_KEYS")

    # --- Infra ---------------------------------------------------------------
    redis_url: str = Field("redis://redis:6379/0", alias="REDIS_URL")
    data_dir: Path = Field(Path("/data"), alias="DATA_DIR")
    tts_base_url: str = Field("http://tts:8880/v1", alias="TTS_BASE_URL")

    # --- Synthesis defaults --------------------------------------------------
    tts_voice: str = Field("af_heart", alias="TTS_VOICE")
    tts_speed: float = Field(1.0, alias="TTS_SPEED", ge=0.5, le=2.0)
    synth_concurrency: int = Field(2, alias="SYNTH_CONCURRENCY", ge=1, le=8)
    onnx_intra_op: int = Field(2, alias="ONNX_INTRA_OP", ge=1, le=16)
    chunk_target_chars: int = Field(1200, alias="CHUNK_TARGET_CHARS", ge=200, le=100_000)
    chunk_hard_max: int = Field(2000, alias="CHUNK_HARD_MAX", ge=200, le=100_000)
    cache_max_gb: float = Field(5.0, alias="CACHE_MAX_GB", ge=0.0)
    non_english_policy: NonEnglishPolicy = Field("skip_warn", alias="NON_ENGLISH_POLICY")

    # --- Processing ----------------------------------------------------------
    processor_default: ProcessorName = Field("explainer", alias="PROCESSOR_DEFAULT")
    explainer_target_minutes: int = Field(30, alias="EXPLAINER_TARGET_MINUTES", ge=1, le=600)
    gemini_api_key: str | None = Field(None, alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-2.5-flash", alias="GEMINI_MODEL")

    # --- Scheduling / limits -------------------------------------------------
    fast_lane_max_minutes: float = Field(20.0, alias="FAST_LANE_MAX_MINUTES", gt=0)
    max_input_chars: int = Field(3_000_000, alias="MAX_INPUT_CHARS", ge=1)
    max_active_jobs: int = Field(2, alias="MAX_ACTIVE_JOBS", ge=1)
    max_failed_chunk_pct: float = Field(1.0, alias="MAX_FAILED_CHUNK_PCT", ge=0, le=100)
    retention_days: int = Field(7, alias="RETENTION_DAYS", ge=1)

    # --- Storage -------------------------------------------------------------
    storage_backend: StorageBackend = Field("local", alias="STORAGE_BACKEND")
    gdrive_root_folder_name: str = Field("Narrator", alias="GDRIVE_ROOT_FOLDER_NAME")
    gdrive_client_id: str | None = Field(None, alias="GDRIVE_CLIENT_ID")
    gdrive_client_secret: str | None = Field(None, alias="GDRIVE_CLIENT_SECRET")
    gdrive_refresh_token: str | None = Field(None, alias="GDRIVE_REFRESH_TOKEN")

    # --- Webhooks ------------------------------------------------------------
    webhook_secret: str | None = Field(None, alias="WEBHOOK_SECRET")

    # --- API -----------------------------------------------------------------
    rate_limit_per_min: int = Field(120, alias="RATE_LIMIT_PER_MIN", ge=1)

    # --- Derived -------------------------------------------------------------
    @cached_property
    def api_key_set(self) -> frozenset[str]:
        """Non-empty API keys, split from the comma-separated env value."""
        return frozenset(k.strip() for k in self.api_keys.split(",") if k.strip())

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache" / "tts"

    @property
    def silence_dir(self) -> Path:
        return self.data_dir / "silence"

    @property
    def ledger_path(self) -> Path:
        return self.data_dir / "ledger.db"

    # --- Validation ----------------------------------------------------------
    @field_validator("tts_base_url")
    @classmethod
    def _tts_url_shape(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("TTS_BASE_URL must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("redis_url")
    @classmethod
    def _redis_url_shape(cls, v: str) -> str:
        if not v.startswith(("redis://", "rediss://", "unix://")):
            raise ValueError("REDIS_URL must start with redis://, rediss:// or unix://")
        return v

    @model_validator(mode="after")
    def _cross_field(self) -> Settings:
        if not self.api_key_set:
            raise ValueError("API_KEYS must contain at least one non-empty key")
        if self.chunk_target_chars > self.chunk_hard_max:
            raise ValueError("CHUNK_TARGET_CHARS must be <= CHUNK_HARD_MAX")
        if self.storage_backend == "gdrive":
            missing = [
                name
                for name, val in (
                    ("GDRIVE_CLIENT_ID", self.gdrive_client_id),
                    ("GDRIVE_CLIENT_SECRET", self.gdrive_client_secret),
                    ("GDRIVE_REFRESH_TOKEN", self.gdrive_refresh_token),
                )
                if not val
            ]
            if missing:
                raise ValueError(
                    "STORAGE_BACKEND=gdrive requires " + ", ".join(missing)
                )
        return self

    def require_gemini(self) -> str:
        """Return the Gemini key or raise a clear error (used by the factory).

        The explainer processor is validated lazily so that a verbatim-only or
        config-print boot does not demand a Gemini key it will never use.
        """
        if not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is required for explainer mode "
                "(set it, or use PROCESSOR_DEFAULT=verbatim)"
            )
        return self.gemini_api_key


def _load() -> Settings:
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        parts: list[str] = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"]) or "config"
            parts.append(f"{loc}: {err['msg']}")
        message = "Narrator config error: " + "; ".join(parts)
        raise SystemExit(message) from exc


settings = _load()
