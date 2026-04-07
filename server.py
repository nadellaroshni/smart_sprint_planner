"""
FastAPI server for the Smart Sprint Planner RL Environment.

Endpoints:
  POST /reset       — Start a new sprint episode
  POST /step        — Execute one agent action
  GET  /state       — Inspect current environment state
  GET  /render      — Human-readable sprint board
  GET  /grade       — Compute final grade for the current episode
  GET  /health      — Health check
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from env.environment import SprintEnv
from env.graders import grade
from env.models import (
    Action, Difficulty, GradeResponse, Observation,
    ResetRequest, StepRequest, StepResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

env: SprintEnv | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global env
    env = SprintEnv(difficulty=Difficulty.MEDIUM, max_steps=20, use_llm=False)
    logger.info("SprintEnv initialised.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Smart Sprint Planner RL Environment",
    description="End-to-end Agile sprint planning RL environment with NLP pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "env_ready": env is not None}


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest = ResetRequest()):
    global env
    if env is None:
        raise HTTPException(500, "Environment not initialised")
    obs = env.reset(
        difficulty=request.difficulty,
        audio_path=request.audio_path,
        transcript_override=request.transcript_override,
    )
    logger.info(f"Environment reset: difficulty={request.difficulty}")
    return obs


@app.post("/step", response_model=StepResult)
def step(request: StepRequest):
    if env is None:
        raise HTTPException(500, "Environment not initialised. Call /reset first.")

    action = Action(task_id=request.task_id, developer_id=request.developer_id, notes=request.notes)
    obs, reward, done, info = env.step(action)
    return StepResult(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def get_state():
    if env is None:
        raise HTTPException(500, "Environment not initialised.")
    return env.state()


@app.get("/render")
def render():
    if env is None:
        raise HTTPException(500, "Environment not initialised.")
    return {"board": env.render()}


@app.get("/grade", response_model=GradeResponse)
def get_grade():
    if env is None:
        raise HTTPException(500, "Environment not initialised.")
    result = grade(env)
    return GradeResponse(**result)