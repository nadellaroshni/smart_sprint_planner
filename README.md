# Smart Sprint Planner

Real-world OpenEnv environment for Agile sprint planning and re-planning.

Pipeline:
`meeting text/audio -> extraction -> JIRA-style tickets -> sprint assignments -> dynamic disruptions -> reward + grading`

This project is built for the OpenEnv Round 1 competition context in [context_scaler.txt](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/context_scaler.txt).

## What The Environment Simulates

The environment models a real planning workflow a delivery lead or engineering manager would actually do:

- read sprint planning input
- turn messy discussion into structured work items
- assign work across a team with capacity constraints
- react to changing conditions mid-sprint
- maximize completion, on-time delivery, balance, and adaptability

## Difficulty Design

Difficulty now means volatility, not just “more tasks”.

- `easy`: static sprint planning
  Fixed backlog, fixed team capacity, fixed deadlines.
- `medium`: one mid-sprint disruption
  A single event is introduced, such as urgent work being added.
- `hard`: multiple dynamic disruptions
  The environment can introduce new work, capacity changes, and dependency shifts over time.

## Project Flow

1. [env/transcription.py](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/transcription.py)
   Turns audio into text, or uses provided text / deterministic scenario text.
2. [env/extraction.py](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/extraction.py)
   Extracts structured sprint work items.
3. [env/jira.py](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/jira.py)
   Converts extracted items into JIRA-style tickets.
4. [env/environment.py](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/environment.py)
   Runs the sprint simulation using `reset()`, `step()`, and `state()`.
5. [env/graders.py](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/graders.py)
   Computes dense rewards and final episode grades.

## Observation And Action Spaces

Observation includes:

- meeting text
- extracted items
- active backlog tickets
- developer pool
- completed task ids
- sprint day
- sprint metrics
- recent disruption events

Action is a single assignment:

```json
{
  "task_id": "T001",
  "developer_id": "D1"
}
```

## Extraction Schema

Each extracted item can include richer planning context than just title + deadline:

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

LLM extraction is used when API credentials are configured. Otherwise the project uses a deterministic rule-based fallback for offline reproducibility.

## Reward And Grading

Step rewards include:

- on-time completion reward
- specialization match reward
- high-priority completion reward
- penalties for invalid ids, blocked tasks, and over-capacity assignment
- adaptation reward for handling disruption-created work

Final grading combines:

- completion rate
- on-time rate
- extraction quality
- workload balance
- efficiency
- adaptability

All final scores are normalized into `[0.0, 1.0]`.

## Competition Baseline Inference

The root-level [inference.py](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/inference.py) is structured for the competition baseline requirements:

- uses the OpenAI client when API credentials are configured
- falls back deterministically when no key is available locally
- emits strict stdout lines in this order:
  - `[START]`
  - `[STEP]`
  - `[END]`

Environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `OPENAI_API_KEY`

Run one task:

```bash
python inference.py medium
```

Run all tasks:

```bash
python inference.py --all
```

## Training

Training is separate from the competition baseline script.

[train.py](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/train.py) trains a DDQN agent over the environment. The training curriculum is intended to move from stable planning into dynamic re-planning.

```bash
python train.py --episodes 400
```

Evaluate saved checkpoints:

```bash
python eval.py --checkpoint checkpoints/best
```

## API Server

Start the environment server:

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

## Tests

```bash
pytest tests -v
```

Current local status:

- environment tests pass
- extraction tests pass

## Docker

Build:

```bash
docker build -t smart-sprint-planner .
```

Run:

```bash
docker run -p 7860:7860 smart-sprint-planner
```

## OpenEnv Metadata

The environment metadata currently lives in [openenv.yaml](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/openenv.yaml).

Before submission, make sure:

- OpenEnv metadata matches the final environment behavior
- `openenv validate` passes
- Docker build succeeds
- the Hugging Face Space responds correctly
- baseline inference is reproducible

## Current Priority

For the competition, the practical order is:

1. keep the environment and inference path validator-compliant
2. verify `openenv validate`, Docker, and HF Space behavior
3. improve extraction quality and trained-agent performance
