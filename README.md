# 🚀 Smart Sprint Planner — RL Environment

End-to-end Agile sprint planning RL environment.  
**Pipeline:** Meeting Audio → Whisper Transcription → LLM Action Extraction → JIRA Ticket Generation → RL Sprint Planning → Dense Reward Signal

---

## Architecture

```
smart_sprint_env/
├── env/
│   ├── __init__.py
│   ├── models.py          # Typed Pydantic models (Task, Developer, Observation, Action)
│   ├── environment.py     # Core SprintEnv RL environment
│   ├── transcription.py   # Whisper-based audio transcription (cached)
│   ├── extraction.py      # LLM action item extraction with fallback
│   ├── jira.py            # JIRA ticket generator (story points, deps, criteria)
│   ├── tasks.py           # Easy / Medium / Hard task sets + developer pools
│   └── graders.py         # Dense reward + hackathon grading (5 dimensions)
├── server.py              # FastAPI REST server
├── inference.py           # Heuristic agent runner
├── tests/
│   └── test_environment.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Setup
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run inference (no LLM needed)
```bash
# Single episode
python inference.py medium --render

# All difficulties
python inference.py --all
```

### 3. Start API server
```bash
uvicorn server:app --reload --port 7860
```

API docs: http://localhost:7860/docs

### 4. Run tests
```bash
pytest tests/ -v
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/reset` | Start new sprint episode |
| POST | `/step` | Execute assignment action |
| GET | `/state` | Inspect environment state |
| GET | `/render` | Human-readable sprint board |
| GET | `/grade` | Compute final score |

### Reset example
```json
POST /reset
{
  "difficulty": "medium",
  "transcript_override": "Fix auth bug urgently. Build dashboard by day 7."
}
```

### Step example
```json
POST /step
{
  "task_id": "T001",
  "developer_id": "D1"
}
```

---

## Reward Design (Dense)

| Signal | Value |
|--------|-------|
| Task completed on time | +0.5 |
| Developer skill matches task tags | +0.2 |
| High-priority task completed | +0.1 |
| Task completed late | -0.3 |
| Developer over capacity | -0.2 |
| Invalid task/dev ID | -0.1 |
| Blocked task assigned early | -0.4 |
| All tasks completed (episode bonus) | +1.0 |
| Balanced workload (Gini < 0.2) | +0.5 |
| No deadline violations | +0.3 |
| Efficiency bonus | +0.2 |

---

## Grading (Hackathon)

| Dimension | Weight |
|-----------|--------|
| Completion rate | 30% |
| On-time delivery | 25% |
| Extraction quality | 20% |
| Workload balance | 15% |
| Efficiency | 10% |

---

## Difficulty Levels

| Level | Tickets | Developers | Characteristics |
|-------|---------|------------|-----------------|
| Easy | 4 | 3 (full cap) | Clear tasks, generous deadlines |
| Medium | 7 | 3 (reduced cap) | Mixed priorities, some dependencies |
| Hard | 11 | 4 (bottlenecks) | Noisy transcript, tight deadlines, dep chains, skill mismatches |

---

## Deployment (Hugging Face Spaces)

1. Create a Space → Docker template
2. Push this repo
3. Add environment variables:
   - `API_BASE_URL` — LLM API base URL
   - `MODEL_NAME` — e.g. `gpt-3.5-turbo` or HuggingFace model
   - `HF_TOKEN` — your token

```bash
git init
git add .
git commit -m "init"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/smart-sprint-env
git push
```

Validate:
```bash
bash validate-submission.sh https://YOUR_SPACE.hf.space
```