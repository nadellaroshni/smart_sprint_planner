---
title: Smart Sprint Planner
emoji: "🗂️"
colorFrom: blue
colorTo: green
# Force rebuild - ensures latest code is deployed
sdk: docker
app_port: 7860
---

# Smart Sprint Planner

Real-world OpenEnv environment for agile sprint planning and dynamic replanning.

Core pipeline:
`audio or transcript -> extraction -> JIRA-style tickets -> developer assignments -> dynamic disruptions -> reward and grading`

This repository is aligned to the Round 1 competition requirements captured in [context_scaler.txt](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/context_scaler.txt).

## Environment Summary

The environment simulates a real planning workflow an engineering manager, scrum lead, or delivery lead would actually perform:

- convert planning discussion into structured tasks
- assign work under team capacity constraints
- respond to urgent work, capacity loss, and dependency changes
- maximize completion, timeliness, workload balance, and adaptability

This is intended as a real-world planning and replanning environment, not a toy game.

## Tasks And Difficulty

There are 3 graded tasks:

- `easy`
  Static sprint planning. Fixed backlog, fixed capacity, fixed deadlines.
- `medium`
  One disruption event. Usually urgent work or a developer capacity loss.
- `hard`
  Multiple disruptions over time. New work, dependency shifts, and changing capacity.

Difficulty represents volatility, not just more tickets.

## Project Flow

1. [env/transcription.py](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/transcription.py)
   Handles audio-to-text or accepts provided transcript text.
2. [env/extraction.py](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/extraction.py)
   Extracts structured work items with an LLM or deterministic fallback logic.
3. [env/jira.py](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/jira.py)
   Converts extracted items into JIRA-style sprint tickets.
4. [env/environment.py](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/environment.py)
   Implements `reset()`, `step()`, and `state()` with dynamic event handling.
5. [env/graders.py](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/graders.py)
   Computes dense rewards and final deterministic grading in `[0.0, 1.0]`.
6. [planner.py](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/planner.py)
   Runs the full end-to-end pipeline and returns assignment recommendations.

## Observation And Action Space

Observation includes:

- meeting text
- extracted work items
- active JIRA tickets
- developer pool
- completed task ids
- sprint day
- metrics and event history
- pending and recent disruption signals

Action is one assignment:

```json
{
  "task_id": "T001",
  "developer_id": "D1"
}
```

## Extraction Schema

Each extracted item can include:

- `task`
- `description`
- `deadline`
- `priority`
- `category`
- `tags`
- `acceptance_criteria`
- `dependency_hints`
- `owner_hint`
- `urgency_reason`
- `raw_text`

LLM extraction uses the OpenAI client when credentials are available. Otherwise the system uses a deterministic rule-based fallback for offline reproducibility.

## Reward And Grading

Dense step rewards include:

- on-time completion
- specialization or skill-match reward
- priority-aware completion reward
- penalties for invalid, blocked, and over-capacity actions
- adaptation reward for disruption-created work
- future-feasibility shaping for preserving replanning options

Final grading combines:

- completion rate
- on-time rate
- extraction quality
- workload balance
- efficiency
- adaptability

All final scores are normalized to `[0.0, 1.0]`.

## Baseline Inference

The required root-level baseline script is [inference.py](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/inference.py).

It:

- uses the OpenAI client for LLM calls
- reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- also supports `OPENAI_API_KEY` locally
- falls back deterministically when no key is configured
- emits strict competition stdout lines:
  - `[START]`
  - `[STEP]`
  - `[END]`

Run one task:

```bash
python inference.py medium
```

Run all tasks:

```bash
python inference.py --all
```

Current local reproducible baseline from `python inference.py --all`:

- `easy`: `0.83`
- `medium`: `0.73`
- `hard`: `0.76`

These are the submission-safe heuristic fallback scores in the current environment.

## Learned Planner

The learned planner is trained separately and is not required for the baseline script.

- training entrypoint: [train.py](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/train.py)
- evaluation entrypoint: [eval.py](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/eval.py)
- strongest checkpoint: `checkpoints/best`

Current held-out dataset-eval comparison for the strongest checkpoint:

- Heuristic: `0.893`
- DDQN: `0.897`

On the richer held-out split, the learned DDQN now slightly outperforms the heuristic overall and on `hard`.

Train:

```bash
python train.py --episodes 400
```

Evaluate:

```bash
python eval.py --checkpoint checkpoints/best --scenario-source dataset-eval
```

## Full Pipeline

The full product-facing path is exposed through [planner.py](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/planner.py) and `POST /plan`.

Run locally from transcript:

```bash
python planner.py --transcript "Fix the checkout bug today, then finish analytics after auth." --difficulty medium --strategy auto
```

`auto` prefers the trained DDQN checkpoint when `checkpoints/best` exists, and falls back to heuristic otherwise.

## API Server

Start the server:

```bash
uvicorn server.app:app --reload --port 7860
```

Endpoints:

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /render`
- `GET /grade`
- `POST /plan`

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Validation And Tests

Run tests:

```bash
pytest tests -v
```

Run OpenEnv validation:

```powershell
.\whisper_env\Scripts\openenv.exe validate
```

Current local status:

- `25` tests passing
- `openenv validate` previously passing in the project environment
- baseline inference reproducing all 3 tasks

## Docker

Build:

```bash
docker build -t smart-sprint-planner .
```

Run:

```bash
docker run -p 7860:7860 smart-sprint-planner
```

## Metadata

Environment metadata lives in [openenv.yaml](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/openenv.yaml).

Before final submission, the remaining non-code checklist is:

1. verify Docker builds on the target machine
2. verify the Hugging Face Space responds with `200`
3. keep the root `inference.py` output unchanged
4. submit with `checkpoints/best` included if you want `auto` to use DDQN
