"""Fail-fast settings for the orchestrator."""

from __future__ import annotations

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    redis_url: str = Field(..., min_length=8, validation_alias=AliasChoices("REDIS_URL", "NARRATION_REDIS_URL"))
    redis_key_prefix: str = "narration:"
    # Must outlive the on-disk reaper (168h) so article→cache bindings
    # still resolve for replay / phone download until the opus is deleted.
    redis_ttl_seconds: int = 691200
    reaper_age_hours: int = Field(
        default=168,
        validation_alias=AliasChoices(
            "NARRATION_REAPER_AGE_HOURS",
            "REAPER_AGE_HOURS",
        ),
    )

    narration_api_key: str = ""
    data_dir: str = "/data"

    tts_base_url: str = "http://narration-tts:8880/v1"
    tts_voice: str = "am_onyx"
    tts_speed: float = 0.9
    tts_alt_voice: str = "am_michael"
    tts_blend: str = "am_onyx(2)+am_michael(1)"

    llm_base_url: str = "http://narration-llm:11434"
    llm_model: str = "gemini-2.5-flash-lite"
    gemini_api_key: str
    gemini_fallback_models: str = ""
    narration_daily_llm_cap: int = 150
    llm_num_ctx: int = 8192
    model_version: str = "qwen3.5-4b-q4km+kokoro-int8-v1.0"

    audio_format: str = "opus"
    audio_bitrate: int = 32000
    hd_bitrate: int = 48000
    max_duration_s: int = 600
    word_target_min: int = 500
    word_target_max: int = 700

    ram_floor_bytes: int = 1_073_741_824
    ram_alert_floor_bytes: int = 524_288_000
    breaker_threshold: int = 3
    breaker_cooldown_s: int = 900
    job_timeout_s: int = 2700

    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    gguf_path: str = "/models/Qwen3.5-4B-Q4_K_M.gguf"

    @field_validator("gemini_api_key")
    @classmethod
    def _gemini_required(cls, v: str) -> str:
        text = str(v or "").strip()
        if not text or text.startswith("${") or text.startswith("{{"):
            raise ValueError("GEMINI_API_KEY is missing. Set the Coolify team variable and redeploy.")
        return text

    @field_validator("redis_url")
    @classmethod
    def _redis_db(cls, v: str) -> str:
        # Dedicated DB index is a hard requirement — DB 0 is LiteLLM.
        if v.rstrip("/").endswith("/0"):
            raise ValueError("REDIS_URL must not use DB 0 (reserved for LiteLLM)")
        return v

    @field_validator("reaper_age_hours")
    @classmethod
    def _reaper_hours(cls, v: int) -> int:
        if not 1 <= v <= 720:
            raise ValueError("NARRATION_REAPER_AGE_HOURS must be between 1 and 720")
        return v

    @field_validator("tts_speed")
    @classmethod
    def _speed(cls, v: float) -> float:
        if not 0.5 <= v <= 2.0:
            raise ValueError("TTS_SPEED must be between 0.5 and 2.0")
        return v


def load_settings() -> Settings:
    return Settings()
