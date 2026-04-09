"""
Action item extraction from meeting transcripts.

Design goals:
  - Prefer structured LLM extraction when available
  - Preserve richer planning detail than just task title + deadline
  - Keep an offline heuristic fallback that is still useful
  - Cache normalized outputs for deterministic training
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .models import ExtractedItem

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/extractions")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_VERSION = "v2"

_client = None

ACTION_VERBS = (
    "fix",
    "implement",
    "build",
    "create",
    "develop",
    "add",
    "write",
    "migrate",
    "refactor",
    "upgrade",
    "update",
    "configure",
    "setup",
    "stabilize",
    "patch",
    "complete",
    "finish",
    "improve",
    "reduce",
    "optimize",
    "document",
)

TAG_KEYWORDS: Dict[str, List[str]] = {
    "bug": ["bug", "broken", "failing", "failure", "error", "crash", "hotfix", "patch"],
    "auth": ["auth", "oauth", "sso", "login", "token", "identity"],
    "payments": ["payment", "checkout", "invoice", "billing", "transaction"],
    "frontend": ["frontend", "ui", "ux", "layout", "page", "component", "mobile", "safari"],
    "backend": ["backend", "api", "service", "endpoint", "worker", "gateway"],
    "infra": ["infra", "ci", "cd", "pipeline", "deploy", "deployment", "docker", "kubernetes", "rollback"],
    "testing": ["test", "tests", "coverage", "regression", "integration", "qa", "e2e"],
    "database": ["database", "postgres", "mysql", "sql", "schema", "migration", "table"],
    "analytics": ["analytics", "dashboard", "chart", "reporting", "metrics"],
    "performance": ["slow", "latency", "performance", "optimize", "8 seconds", "load time"],
    "documentation": ["document", "docs", "documentation", "runbook", "guide"],
    "security": ["security", "privilege", "exploit", "vulnerability", "leak"],
}

OWNER_HINTS: Dict[str, List[str]] = {
    "frontend": ["frontend", "ui", "mobile", "safari", "layout"],
    "backend": ["backend", "api", "service", "database", "auth", "payment"],
    "devops": ["infra", "pipeline", "deploy", "kubernetes", "docker"],
    "qa": ["test", "coverage", "qa", "regression"],
}

CATEGORY_BY_TAG = {
    "bug": "bugfix",
    "testing": "quality",
    "infra": "infrastructure",
    "database": "data",
    "analytics": "feature",
    "documentation": "documentation",
    "security": "security",
}

SYSTEM_PROMPT = """You are a senior Agile delivery lead extracting sprint-ready work from messy meetings.
Return ONLY a valid JSON array. No markdown and no prose.

For every actionable task, output:
- task: short imperative title
- description: one-sentence planning summary
- deadline: sprint day number, integer 1-10
- priority: integer 1-4 where 4 is critical
- category: one of ["bugfix","feature","quality","infrastructure","documentation","security","data","general"]
- tags: relevant technical tags
- acceptance_criteria: 1-3 concise checklist items
- dependency_hints: plain-language task dependencies if implied
- owner_hint: best-fit team area if implied, else null
- urgency_reason: why it is urgent, else empty string
- raw_text: supporting quote/snippet from the transcript

Rules:
- Extract all actionable items, including implied engineering follow-up work.
- Preserve urgency and sequencing details.
- If a task blocks another, include that in dependency_hints.
- If a deadline is vague, infer a reasonable sprint day from the wording.
"""


def _get_client():
    global _client
    if _client is None:
        try:
            from openai import OpenAI

            # Read environment variables with fallback chain
            # Validator injects API_KEY and API_BASE_URL, but code must handle other sources too
            api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN")
            api_base_url = os.environ.get("API_BASE_URL") or "https://api.openai.com/v1"
            
            if not api_key:
                logger.info("No API key configured for LLM extraction; using fallback extractor.")
                _client = "unavailable"
                return _client

            _client = OpenAI(
                base_url=api_base_url,
                api_key=api_key,
            )
        except Exception as e:
            logger.warning(f"Could not initialise OpenAI client: {e}")
            _client = "unavailable"
    return _client


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

    model = os.getenv("MODEL_NAME", "gpt-4o-mini")

    if hasattr(client, "responses"):
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Transcript:\n{transcript}"},
            ],
        )
        raw = getattr(response, "output_text", "").strip()
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Transcript:\n{transcript}"},
            ],
            temperature=0.0,
            max_tokens=1600,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()

    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    parsed = json.loads(raw)
    if isinstance(parsed, dict) and "items" in parsed:
        parsed = parsed["items"]
    if not isinstance(parsed, list):
        raise ValueError("Expected extracted task list")
    return parsed


def extract_tasks(transcript: str) -> List[ExtractedItem]:
    """Extract normalized task items from a meeting transcript."""
    cache_key = hashlib.sha256(f"{CACHE_VERSION}::{transcript}".encode()).hexdigest()
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

    items = [_normalize_item(r) for r in raw_items]
    items = [item for item in items if item is not None]

    cache_file.write_text(json.dumps([i.model_dump() for i in items], indent=2))
    return items


def _normalize_item(raw: dict) -> Optional[ExtractedItem]:
    try:
        return ExtractedItem(
            task=_clean_title(str(raw.get("task", "Unnamed task"))),
            description=str(raw.get("description", "")).strip(),
            deadline=max(1, min(int(raw.get("deadline", 5)), 10)),
            priority=max(1, min(int(raw.get("priority", 2)), 4)),
            category=str(raw.get("category", "general")).strip().lower() or "general",
            tags=_normalize_tags(raw.get("tags", [])),
            acceptance_criteria=_normalize_list(raw.get("acceptance_criteria", []), limit=3),
            dependency_hints=_normalize_list(raw.get("dependency_hints", []), limit=3),
            owner_hint=_clean_optional(raw.get("owner_hint")),
            urgency_reason=str(raw.get("urgency_reason", "")).strip(),
            raw_text=str(raw.get("raw_text", "")).strip(),
        )
    except Exception as ve:
        logger.warning(f"Skipping malformed extraction item {raw}: {ve}")
        return None


def _rule_based_extract(transcript: str) -> List[dict]:
    clauses = _split_into_clauses(transcript)
    items: List[dict] = []

    for clause in clauses:
        if not _looks_actionable(clause):
            continue

        tags = _infer_tags(clause)
        category = _infer_category(tags)
        priority = _infer_priority(clause)
        deadline = _infer_deadline(clause)
        dependency_hints = _infer_dependency_hints(clause)
        owner_hint = _infer_owner_hint(clause, tags)
        urgency_reason = _infer_urgency_reason(clause)

        items.append(
            {
                "task": _task_from_clause(clause),
                "description": _describe_clause(clause),
                "deadline": deadline,
                "priority": priority,
                "category": category,
                "tags": tags,
                "acceptance_criteria": _acceptance_from_clause(clause, tags),
                "dependency_hints": dependency_hints,
                "owner_hint": owner_hint,
                "urgency_reason": urgency_reason,
                "raw_text": clause,
            }
        )

    logger.info(f"Rule-based extracted {len(items)} items.")
    return items


def _split_into_clauses(transcript: str) -> List[str]:
    text = re.sub(r"\s+", " ", transcript.strip())
    sentences = re.split(r"(?<=[.!?])\s+|;\s*", text)
    clauses: List[str] = []
    for sentence in sentences:
        parts = re.split(r"\b(?:also|and|plus|then|meanwhile|separately)\b", sentence, flags=re.IGNORECASE)
        for part in parts:
            cleaned = part.strip(" ,.-")
            if len(cleaned) >= 12:
                clauses.append(cleaned)
    return clauses


def _looks_actionable(text: str) -> bool:
    lower = text.lower()
    return any(re.search(rf"\b{verb}\b", lower) for verb in ACTION_VERBS) or any(
        keyword in lower for words in TAG_KEYWORDS.values() for keyword in words
    )


def _infer_tags(text: str) -> List[str]:
    lower = text.lower()
    tags = {
        tag
        for tag, keywords in TAG_KEYWORDS.items()
        for keyword in keywords
        if keyword in lower
    }
    if not tags:
        tags.add("general")
    return sorted(tags)


def _infer_category(tags: List[str]) -> str:
    for tag in tags:
        if tag in CATEGORY_BY_TAG:
            return CATEGORY_BY_TAG[tag]
    if "general" in tags:
        return "general"
    return "feature"


def _infer_priority(text: str) -> int:
    lower = text.lower()
    if any(token in lower for token in ["p0", "critical", "urgent", "immediately", "today", "security"]):
        return 4
    if any(token in lower for token in ["high priority", "blocker", "blocking", "must", "tomorrow"]):
        return 3
    if any(token in lower for token in ["low priority", "nice to have", "later"]):
        return 1
    return 2


def _infer_deadline(text: str) -> int:
    lower = text.lower()
    explicit = re.search(r"\bday\s*(\d{1,2})\b", lower)
    if explicit:
        return max(1, min(int(explicit.group(1)), 10))

    if any(token in lower for token in ["immediately", "today", "now"]):
        return 1
    if any(token in lower for token in ["tomorrow", "within 2 days", "next 2 days"]):
        return 2
    if "end of sprint" in lower:
        return 10
    if "this week" in lower:
        return 5
    if any(token in lower for token in ["client demo", "release", "before launch"]):
        return 4
    return 5


def _infer_dependency_hints(text: str) -> List[str]:
    lower = text.lower()
    hints: List[str] = []
    patterns = [
        r"blocks? ([^.,;]+)",
        r"blocked by ([^.,;]+)",
        r"depends on ([^.,;]+)",
        r"before ([^.,;]+)",
        r"after ([^.,;]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lower):
            hints.append(match.group(0).strip())
    return hints[:3]


def _infer_owner_hint(text: str, tags: List[str]) -> Optional[str]:
    lower = text.lower()
    for owner, keywords in OWNER_HINTS.items():
        if any(keyword in lower for keyword in keywords):
            return owner
    if "frontend" in tags:
        return "frontend"
    if {"backend", "database", "auth", "payments"} & set(tags):
        return "backend"
    if "infra" in tags:
        return "devops"
    if "testing" in tags:
        return "qa"
    return None


def _infer_urgency_reason(text: str) -> str:
    lower = text.lower()
    if "blocking" in lower or "blocks" in lower:
        return "Blocks downstream work"
    if any(token in lower for token in ["urgent", "critical", "p0", "security"]):
        return "Critical production urgency"
    if "client demo" in lower:
        return "Needed for client demo"
    if "release" in lower:
        return "Needed for release readiness"
    return ""


def _task_from_clause(text: str) -> str:
    cleaned = re.sub(r"^(we need to|someone needs to|need to|please|also)\s+", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" .")
    words = cleaned.split()
    if len(words) <= 10:
        return _clean_title(cleaned)
    return _clean_title(" ".join(words[:10]))


def _describe_clause(text: str) -> str:
    cleaned = text.strip()
    if cleaned.endswith("."):
        return cleaned
    return f"{cleaned}."


def _acceptance_from_clause(text: str, tags: List[str]) -> List[str]:
    criteria = ["Implementation merged and verified"]
    if "bug" in tags:
        criteria = ["Issue reproduced", "Fix validated", "Regression coverage added"]
    elif "testing" in tags:
        criteria = ["Coverage added for target flow", "Tests pass in CI"]
    elif "frontend" in tags:
        criteria = ["Works on desktop and mobile", "Visual regression checked"]
    elif "infra" in tags:
        criteria = ["Pipeline succeeds", "Rollback or recovery path documented"]
    elif "database" in tags:
        criteria = ["Migration runs safely", "Data integrity verified"]
    elif "documentation" in tags:
        criteria = ["Documentation reviewed", "Consumers can follow the new flow"]
    elif "security" in tags:
        criteria = ["Exploit path closed", "Security-sensitive behavior re-tested"]

    if "performance" in tags:
        criteria = criteria[:2] + ["Performance improved against baseline"]
    return criteria[:3]


def _normalize_tags(tags: object) -> List[str]:
    if not isinstance(tags, list):
        return []
    normalized = []
    for tag in tags:
        clean = str(tag).strip().lower().replace(" ", "_")
        if clean:
            normalized.append(clean)
    return sorted(set(normalized))


def _normalize_list(values: object, limit: int = 3) -> List[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    cleaned = []
    for value in values:
        text = str(value).strip()
        if text:
            cleaned.append(text)
    return cleaned[:limit]


def _clean_title(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" .")
    if not text:
        return "Unnamed task"
    return text[0].upper() + text[1:]


def _clean_optional(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
