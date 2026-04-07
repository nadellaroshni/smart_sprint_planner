"""
Action Item Extraction from meeting transcripts using an LLM.

Features:
- Structured JSON extraction with Pydantic validation
- Retry with exponential backoff (tenacity)
- Deterministic output cache keyed by transcript hash
- Graceful fallback to rule-based extraction if LLM is unavailable
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .models import ExtractedItem, Priority

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/extractions")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM client (lazy init)
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from openai import OpenAI
            _client = OpenAI(
                base_url=os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
                api_key=os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY", "no-key")),
            )
        except Exception as e:
            logger.warning(f"Could not initialise OpenAI client: {e}")
            _client = "unavailable"
    return _client


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior Agile project manager.
Extract all actionable tasks from the meeting transcript.
Return ONLY a valid JSON array. No markdown, no explanation.

Each element must have:
  "task"     : short imperative title (string)
  "deadline" : sprint day number (int, 1-10)
  "priority" : 1=low 2=medium 3=high 4=critical (int)
  "tags"     : relevant tags like ["backend","bug","auth","infra","frontend","testing"] (list)
  "raw_text" : verbatim snippet from transcript (string)

Rules:
- If something is "urgent" or "P0" → priority 4
- "high priority" → 3
- default → 2
- Include ALL tasks, even minor ones
"""


# ---------------------------------------------------------------------------
# LLM extraction (with retry)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=False,
)
def _call_llm(transcript: str) -> List[dict]:
    client = _get_client()
    if client == "unavailable":
        raise RuntimeError("LLM client not available")

    model = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n{transcript}"},
        ],
        temperature=0.0,  # deterministic
        max_tokens=1024,
    )

    raw = response.choices[0].message.content.strip()
    # Strip possible markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_tasks(transcript: str) -> List[ExtractedItem]:
    """
    Extract structured task items from a meeting transcript.

    Tries LLM extraction first; falls back to rule-based if unavailable.
    Results are cached by transcript hash for reproducibility.
    """
    cache_key = hashlib.sha256(transcript.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        logger.info("Extraction cache hit.")
        raw_items = json.loads(cache_file.read_text())
        return [ExtractedItem(**i) for i in raw_items]

    try:
        raw_items = _call_llm(transcript)
        logger.info(f"LLM extracted {len(raw_items)} items.")
    except Exception as e:
        logger.warning(f"LLM extraction failed ({e}), using rule-based fallback.")
        raw_items = _rule_based_extract(transcript)

    # Validate and normalise
    items = []
    for r in raw_items:
        try:
            items.append(
                ExtractedItem(
                    task=r.get("task", "Unnamed task"),
                    deadline=max(1, min(int(r.get("deadline", 5)), 10)),
                    priority=max(1, min(int(r.get("priority", 2)), 4)),
                    tags=r.get("tags", []),
                    raw_text=r.get("raw_text", ""),
                )
            )
        except Exception as ve:
            logger.warning(f"Skipping malformed item {r}: {ve}")

    cache_file.write_text(json.dumps([i.model_dump() for i in items]))
    return items


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

_PATTERNS = [
    # (regex, priority, tags)
    (r"(fix|bug|crash|error|broken|failing|intermittent)", 4, ["bug"]),
    (r"(implement|build|create|develop|add)\s+\w+\s+(feature|module|page|api)", 3, ["feature"]),
    (r"(migrate|refactor|upgrade|update)\s+\w+", 3, ["infra"]),
    (r"(test|unit test|integration test|e2e)", 1, ["testing"]),
    (r"(setup|configure|ci.?cd|pipeline|docker)", 2, ["infra"]),
    (r"(dashboard|chart|ui|ux|frontend|profile)", 2, ["frontend"]),
    (r"(database|db|sql|postgres|mongo)", 3, ["backend", "infra"]),
]

_DEADLINE_HINTS = {
    "urgent": 2, "immediately": 1, "asap": 1,
    "end of sprint": 10, "this week": 5, "tomorrow": 2,
    "day 1": 1, "day 2": 2, "day 3": 3, "day 4": 4, "day 5": 5,
    "day 6": 6, "day 7": 7, "day 8": 8,
}


def _rule_based_extract(transcript: str) -> List[dict]:
    """Simple sentence-level heuristic extraction."""
    sentences = re.split(r"[.;]", transcript)
    items = []

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 10:
            continue

        priority = 2
        tags = []
        for pattern, prio, ptags in _PATTERNS:
            if re.search(pattern, sent, re.IGNORECASE):
                priority = max(priority, prio)
                tags.extend(ptags)

        deadline = 5
        for hint, day in _DEADLINE_HINTS.items():
            if hint in sent.lower():
                deadline = day
                break

        # Extract a short task title (first 8 words)
        words = sent.split()
        title = " ".join(words[:8]).capitalize()

        items.append({
            "task": title,
            "deadline": deadline,
            "priority": priority,
            "tags": list(set(tags)) or ["general"],
            "raw_text": sent,
        })

    logger.info(f"Rule-based extracted {len(items)} items.")
    return items