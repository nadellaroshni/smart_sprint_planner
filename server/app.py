"""
FastAPI server for the Smart Sprint Planner RL Environment.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from env.environment import SprintEnv
from env.graders import grade
from env.models import (
    Action,
    PlanRequest,
    PlanResponse,
    Difficulty,
    GradeResponse,
    Observation,
    ResetRequest,
    StepRequest,
    StepResult,
)
from planner import generate_plan

logger = logging.getLogger(__name__)

env: SprintEnv | None = None


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global env
    env = SprintEnv(difficulty=Difficulty.MEDIUM, max_steps=20, use_llm=False)
    logger.info("SprintEnv initialised.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Smart Sprint Planner RL Environment",
    description="Agile sprint planning and dynamic replanning environment.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "env_ready": env is not None}


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest = ResetRequest()):
    global env
    if env is None:
        raise HTTPException(500, "Environment not initialised")
    return env.reset(
        difficulty=request.difficulty,
        audio_path=request.audio_path,
        transcript_override=request.transcript_override,
    )


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
    return GradeResponse(**grade(env))


@app.post("/plan", response_model=PlanResponse)
def plan(request: PlanRequest):
    if not request.audio_path and not request.transcript:
        raise HTTPException(400, "Provide either audio_path or transcript")
    return generate_plan(
        difficulty=request.difficulty,
        audio_path=request.audio_path,
        transcript=request.transcript,
        strategy=request.strategy,
        checkpoint=request.checkpoint,
    )


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, workers=1)


if __name__ == "__main__":
    main()
