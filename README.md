---
title: Smart Sprint Planner
emoji: "🗂️"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Smart Sprint Planner

Smart Sprint Planner is an OpenEnv environment for a real software-delivery task: planning and replanning an engineering sprint from meeting context, backlog pressure, team capacity, and changing sprint conditions.

Core pipeline:
`audio or transcript -> extracted action items -> JIRA-style tickets -> developer assignments -> disruptions -> grading`

## Why This Environment

This environment simulates the kind of planning work an engineering manager, scrum lead, or delivery owner actually performs:

- convert planning discussion into structured work
- assign tickets under capacity and specialization constraints
- react to urgent work, lost capacity, and dependency changes
- preserve feasibility while maximizing delivery value

It is deliberately not a toy game. The task is operational planning under uncertainty.

## Tasks

There are 3 graded tasks:

| Task | Difficulty | Max Steps | Description |
|------|------------|-----------|-------------|
| `easy` | Easy | 10 | Static sprint planning with fixed backlog and no disruptions |
| `medium` | Medium | 15 | Replanning after one mid-sprint disruption |
| `hard` | Hard | 20 | Multi-disruption planning with shifting capacity and dependencies |

## Observation Space

Each `reset()` and `step()` returns a typed [`Observation`](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/models.py) containing:

- `meeting_text`
- `extracted_items`
- `jira_tickets`
- `developers`
- `completed_task_ids`
- `sprint_day`
- `metrics`
- `difficulty`
- `recent_events`
- `pending_events`

## Action Space

The action is a typed [`Action`](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/models.py) with one assignment:

```json
{
  "task_id": "T001",
  "developer_id": "D1"
}
```

## Reward Design

Dense reward signals are used throughout the trajectory:

- positive reward for on-time completion
- positive reward for skill-matched assignment
- positive reward for high-priority completion
- adaptation reward for disruption-created work
- penalties for invalid ids, blocked work, and over-capacity assignments
- episode bonus for full completion, balance, efficiency, and disruption handling

Final graders return normalized scores in `[0.0, 1.0]`.

## Project Structure

- [`env/models.py`](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/models.py): typed Pydantic contracts
- [`env/tasks.py`](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/tasks.py): scenario registry and fallback datasets
- [`env/extraction.py`](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/extraction.py): transcript-to-work extraction
- [`env/jira.py`](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/jira.py): JIRA-style ticket generation
- [`env/environment.py`](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/environment.py): `reset()`, `step()`, `state()`
- [`env/graders.py`](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/env/graders.py): dense reward and final task graders
- [`server/app.py`](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/server/app.py): FastAPI runtime
- [`inference.py`](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/inference.py): baseline OpenAI-client inference script

## API Endpoints

- `GET /health`
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /render`
- `GET /grade`
- `POST /plan`

## Baseline Inference

The required root-level baseline script is [`inference.py`](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/inference.py).

It:

- uses the OpenAI client for all LLM calls
- reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- also accepts `API_KEY` or `OPENAI_API_KEY` as fallbacks for local testing
- emits only the required `[START]`, `[STEP]`, and `[END]` log lines
- uses an LLM to choose among valid assignment candidates at every step

Example:

```bash
API_BASE_URL=https://router.huggingface.co/v1 MODEL_NAME=Qwen/Qwen2.5-72B-Instruct HF_TOKEN=... python inference.py
```

## Baseline Scores

Current deterministic baseline scores from the built-in heuristic policy on the default scenarios:

| Task | Score |
|------|-------|
| `easy` | `0.833` |
| `medium` | `0.734` |
| `hard` | `0.761` |
| **overall mean** | **`0.776`** |

These numbers are also reflected in [`openenv.yaml`](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/openenv.yaml).

## Local Usage

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Run Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run Planner From Transcript

```bash
python planner.py --difficulty medium --strategy heuristic --transcript "Fix checkout today, then finish analytics after auth."
```

### Run Tests

```bash
python -m pytest tests -v
```

### Run OpenEnv Validation

```bash
openenv validate
```

## Docker

Build:

```bash
docker build -t smart-sprint-planner .
```

Run:

```bash
docker run -p 7860:7860 smart-sprint-planner
```

## Submission Notes

- keep [`inference.py`](C:/Users/ASUS/Documents/GitHub/smart_sprint_planner/inference.py) at the repo root
- do not change the `[START]`, `[STEP]`, `[END]` stdout format
- ensure your Hugging Face Space responds to `POST /reset`
- ensure `openenv validate` passes before submission
