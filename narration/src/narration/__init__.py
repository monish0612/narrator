"""Article-to-audio narration orchestrator.

This package is the system of record for explainer generation + Kokoro TTS.
It must never call or modify the existing LiteLLM / Gemini news-summary path.
"""

__version__ = "1.0.0"
