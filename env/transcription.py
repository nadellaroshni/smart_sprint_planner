"""
Transcription module using OpenAI Whisper.

Uses `tiny` model to stay within CPU/memory constraints.
Audio files are cached so repeated runs are deterministic.
"""

from __future__ import annotations

import hashlib
import json
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Cache directory for transcription results (ensures reproducibility)
CACHE_DIR = Path(".cache/transcriptions")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Lazy-load model to avoid slow startup
_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            import whisper
            logger.info("Loading Whisper tiny model...")
            _model = whisper.load_model("tiny")
            logger.info("Whisper model loaded.")
        except ImportError:
            logger.warning("openai-whisper not installed. Transcription will use fallback.")
            _model = "unavailable"
    return _model


def _cache_key(audio_path: str) -> str:
    """Stable cache key from file content hash."""
    with open(audio_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def transcribe(audio_path: str) -> str:
    """
    Transcribe audio file to text.

    Returns cached result if available (deterministic across runs).
    Falls back to a hardcoded demo transcript if Whisper is unavailable.
    """
    cache_file = CACHE_DIR / f"{_cache_key(audio_path)}.json"

    if cache_file.exists():
        logger.info(f"Cache hit for {audio_path}")
        return json.loads(cache_file.read_text())["text"]

    model = _get_model()
    if model == "unavailable":
        logger.warning("Using fallback transcript (Whisper not available).")
        return _fallback_transcript()

    result = model.transcribe(audio_path)
    text = result["text"]

    cache_file.write_text(json.dumps({"text": text, "audio": audio_path}))
    logger.info(f"Transcription cached to {cache_file}")
    return text


def transcribe_from_text(text: str) -> str:
    """Pass-through for when transcript is already available (testing/demo)."""
    return text


def _fallback_transcript() -> str:
    """
    Realistic demo transcript used when no audio file is provided.
    Covers a mix of features, bugs, and infra tasks with varying urgency.
    """
    return (
        "Alright team, let's go through today's priorities. "
        "First — the login feature with OAuth2 and Google SSO needs to be implemented by end of sprint, "
        "it's blocking the client demo on day 5. "
        "Second — there's a critical payment processing bug; transactions are failing intermittently, "
        "this is a P0 and must be fixed within 2 days. "
        "Third — the dashboard analytics page needs charts for user engagement metrics, "
        "medium priority, deadline is day 7. "
        "Fourth — we need to migrate the user database to PostgreSQL, "
        "high priority, estimate 8 points, due day 6. "
        "Fifth — write unit tests for the auth module, low priority, due day 8. "
        "Sixth — set up the CI/CD pipeline on GitHub Actions, medium priority, due day 5. "
        "Also, fix the UI bug on the profile page where avatars aren't loading — quick fix, day 3."
    )