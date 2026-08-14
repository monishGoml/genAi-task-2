# LLM Fine-Tuning, Distillation & LLM Ops Pipeline

A complete, end-to-end workflow for taking a base Large Language Model from adaptation to a production-style serving setup — covering **LoRA fine-tuning**, **knowledge distillation**, and **LLM Ops** (serving, containerization, and automated evaluation gating).

**Repo:** https://github.com/monishGoml/genAi-task-2.git

---

## Table of Contents

- [Overview](#overview)
- [Architecture / Workflow](#architecture--workflow)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Act 1 — Fine-Tuning](#act-1--fine-tuning)
- [Act 2 — Distillation](#act-2--distillation)
- [Act 3 — LLM Ops](#act-3--llm-ops)
- [Docker](#docker)
- [Evaluation Results](#evaluation-results)
- [Generated Artifacts](#generated-artifacts)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Deliberately Excluded Features](#deliberately-excluded-features)
- [Roadmap / Extending the Project](#roadmap--extending-the-project)
- [License](#license)

---

## Overview

This project simulates a realistic ML lifecycle for a small, task-specific LLM — using a "bakery customer support assistant" as the running example. It walks through three connected phases ("Acts"):

| Act | Phase | Goal |
|---|---|---|
| 1 | **Fine-Tuning** | Adapt a base instruct model (`Llama-3.2-1B-Instruct-bnb-4bit`) to the target domain using LoRA. |
| 2 | **Distillation** | Compress the fine-tuned "teacher" into a smaller, faster "student" model with minimal quality loss. |
| 3 | **LLM Ops** | Serve the resulting model behind a FastAPI endpoint, containerize it with Docker, and gate deployments with an automated evaluation script. |

Each act builds on the previous one's output, so the pipeline can be run start-to-finish or resumed at any stage using the checked-in artifacts.

## Architecture / Workflow

```
                 ┌─────────────────────┐
                 │   Base LLM (4-bit)   │
                 │ Llama-3.2-1B-Instruct│
                 └──────────┬───────────┘
                            │  LoRA fine-tuning
                            ▼
                 ┌─────────────────────┐
   Act 1         │   Teacher Model      │
                 │ (base + LoRA adapter)│
                 └──────────┬───────────┘
                            │  Knowledge distillation
                            ▼
                 ┌─────────────────────┐
   Act 2         │   Student Model      │
                 │  (smaller, faster)   │
                 └──────────┬───────────┘
                            │  Package & serve
                            ▼
                 ┌─────────────────────┐
   Act 3         │  FastAPI + Docker    │
                 │  + eval.py gate      │
                 └─────────────────────┘
```

## Project Structure

```
.
├── fine_tuning/
│   ├── train_lora.py
│   ├── inference_compare.py
│   ├── outputs/
│   │   ├── lora-adapter/
│   │   └── before_after_comparison.json
│   └── README.md
├── distillation/
│   ├── distill.py
│   ├── evaluate_student.py
│   ├── outputs/
│   │   ├── student-model/
│   │   └── student_vs_teacher.json
│   └── README.md
├── llmops/
│   ├── app.py
│   ├── eval.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── outputs/
│   │   └── eval_gate_result.json
│   └── README.md
├── data/
│   └── (sample dataset for training / evaluation)
├── requirements.txt
└── README.md   ← you are here
```

## Prerequisites

- Python 3.9+
- `pip`
- (Optional but recommended) A CUDA-capable GPU for fine-tuning and distillation — CPU-only execution will work but will be significantly slower.
- [Docker](https://www.docker.com/) — only required for Act 3's containerization step.
- ~10–15 GB free disk space for model weights and adapters.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/monishGoml/genAi-task-2.git
cd genAi-task-2
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install root dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the acts in order

```bash
# Act 1 — Fine-Tuning
cd fine_tuning
python train_lora.py
python inference_compare.py
cd ..

# Act 2 — Distillation
cd distillation
python distill.py
python evaluate_student.py
cd ..

# Act 3 — LLM Ops
cd llmops
pip install -r requirements.txt
export MODEL_PATH=../fine_tuning/outputs/lora-adapter
uvicorn app:app --reload --port 8000
```

Each act's sub-directory README has the full detail (flags, expected outputs, Docker commands, etc.) — see the links below.

## Act 1 — Fine-Tuning

Adapts the base model using **LoRA** so it responds appropriately to the target domain (bakery customer support), without the cost of full fine-tuning.

- Script: `fine_tuning/train_lora.py` → produces `fine_tuning/outputs/lora-adapter/`
- Comparison script: `fine_tuning/inference_compare.py` → produces `fine_tuning/outputs/before_after_comparison.json` (base vs. fine-tuned responses + ROUGE-L)

📄 Full instructions: [`fine_tuning/README.md`](./fine_tuning/README.md)

## Act 2 — Distillation

Trains a smaller **student model** to imitate the Act 1 **teacher model** using soft-label / KL-divergence-based distillation loss, trading a small amount of quality for a large gain in speed and size.

- Script: `distillation/distill.py` → produces `distillation/outputs/student-model/`
- Evaluation script: `distillation/evaluate_student.py` → produces `distillation/outputs/student_vs_teacher.json` (parameter reduction, ROUGE-L, latency speedup)

📄 Full instructions: [`distillation/README.md`](./distillation/README.md)

## Act 3 — LLM Ops

Wraps the resulting model in a servable, containerized API with an automated quality gate suitable for CI/CD.

- `llmops/app.py` — FastAPI app exposing a `/generate` endpoint
- `llmops/Dockerfile` — containerizes the API for reproducible deployment
- `llmops/eval.py` — runs held-out prompts against the model and checks ROUGE-L / latency thresholds, exiting `0` (pass) or `1` (fail)

```bash
python eval.py --threshold 0.20 --max-latency 15
```

📄 Full instructions: [`llmops/README.md`](./llmops/README.md)

## Docker

The `llmops/` service can be built and run as a standalone container so the API doesn't depend on the host's Python environment.

**Build the image:**

```bash
cd llmops
docker build -t bakery-llm-api .
```

**Run the container** (maps container port 8000 to the host):

```bash
docker run -p 8000:8000 \
  -e MODEL_PATH="/app/fine_tuning/outputs/lora-adapter" \
  bakery-llm-api
```

**Test the running container:**

```bash
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Do you have any gluten-free cakes?"}'
```

> `MODEL_PATH` inside the container must match the path where model artifacts are copied to in the Dockerfile (`/app/fine_tuning/outputs/lora-adapter`) — this is different from the relative path (`../fine_tuning/outputs/lora-adapter`) used when running locally outside Docker.

## Evaluation Results

The model is evaluated at each stage using **ROUGE-L** (quality) and **response latency** (speed) on a fixed set of customer-support prompts. Three separate evaluation runs exist in this repo, each over its own prompt set and comparison — the numbers below are pulled directly from the corresponding JSON artifacts, so they won't perfectly reconcile with each other stage-to-stage.

### 1. Fine-Tuning — Base vs. Fine-Tuned (`fine_tuning/outputs/before_after_comparison.json`)

| Stage | Avg ROUGE-L |
|---|---|
| Base model | 0.213 |
| Fine-tuned model | 0.233 |

The fine-tuned model produces noticeably shorter, more direct answers (e.g. "Our smallest cake size is 1/4 cup." vs. a long base-model tangent about cupcake sizes), which improves ROUGE-L overlap with the reference answers on most prompts, though not uniformly across all five.

### 2. Distillation — Teacher vs. Student (`distillation/outputs/student_vs_teacher.json`)

| Metric | Teacher | Student |
|---|---|---|
| Total parameters | 760,547,328 | 750,979,072 |
| Avg ROUGE-L | 0.209 | 0.254 |
| Avg latency (s) | 1.146 | 0.815 |

- **Parameter reduction:** 1.3%
- **Speedup:** 1.41x faster inference
- **Quality gap:** -0.045 (student ROUGE-L is *higher* on this eval set — the student is not strictly worse on quality here, while being meaningfully faster)

### 3. LLM Ops — Automated Evaluation Gate (`llmops/outputs/eval_gate_result.json`)

```json
{
  "model_path": "../fine_tuning/outputs/lora-adapter",
  "avg_rouge_l": 0.217,
  "avg_latency_s": 1.317,
  "threshold": 0.2,
  "max_latency": 15.0,
  "passed": true
}
```

| Check | Result | Threshold | Status |
|---|---|---|---|
| ROUGE-L | 0.217 | ≥ 0.20 | ✅ Pass |
| Latency | 1.317s | ≤ 15.0s | ✅ Pass |
| **Gate** | — | — | **✅ PASS** |

This is the deployment gate: `eval.py` exits `0` here, so a CI/CD pipeline would allow this model to be promoted.

**Takeaway:** Fine-tuning improves response quality and conciseness over the base model. Distillation then trades a small (in this case, favorable) change in ROUGE-L for a real ~1.4x latency improvement and a lighter model — a reasonable trade-off for deployment. The LLM Ops eval gate confirms the served model clears both the quality and latency bars before being considered deployment-ready.

## Generated Artifacts

Running all three acts end-to-end produces the following:

| Artifact | Path | Description |
|---|---|---|
| Fine-Tuned LoRA Adapter | `fine_tuning/outputs/lora-adapter/` | LoRA weights to be applied on top of the base model. |
| Base vs. Fine-Tuned Comparison | `fine_tuning/outputs/before_after_comparison.json` | Side-by-side responses + ROUGE-L scores. |
| Distilled Student Model | `distillation/outputs/student-model/` | Smaller, deployment-ready model. |
| Student vs. Teacher Comparison | `distillation/outputs/student_vs_teacher.json` | Efficiency gains and quality trade-offs. |
| LLM Ops Evaluation Gate Results | `llmops/outputs/eval_gate_result.json` | ROUGE-L, latency, and pass/fail status from the eval gate. |

## Configuration

Key environment variables used across the pipeline:

| Variable | Used In | Description |
|---|---|---|
| `MODEL_PATH` | `llmops/app.py` | Path to the model/adapter the API should load and serve. |

Command-line flags of note:

| Flag | Script | Description |
|---|---|---|
| `--threshold` | `llmops/eval.py` | Minimum acceptable ROUGE-L score (e.g. `0.20`). |
| `--max-latency` | `llmops/eval.py` | Maximum acceptable per-request latency in seconds (e.g. `15`). |

## Troubleshooting

- **Out of memory during fine-tuning/distillation**: reduce batch size in `train_lora.py` / `distill.py`, or run on a machine with a larger GPU / more system RAM.
- **`uvicorn` command not found**: make sure you've run `pip install -r requirements.txt` inside `llmops/` (it's not included in the root requirements).
- **API returns errors / model not found**: double-check `MODEL_PATH` points to a valid adapter/model directory — this differs between local runs (`../fine_tuning/outputs/lora-adapter`) and the Docker container (`/app/fine_tuning/outputs/lora-adapter`).
- **`eval.py` fails the gate**: inspect `llmops/outputs/eval_gate_result.json` for per-prompt ROUGE-L/latency values to see which threshold was missed.

## Deliberately Excluded Features

To keep this demo focused and easy to follow, the following advanced LLM Ops capabilities were intentionally left out:

- Semantic caching or cost optimization.
- Llama Guard-style moderation or human-in-the-loop approval gates.
- LangSmith/Arize-style tracing and observability.

## Roadmap / Extending the Project

Natural next steps for turning this into a more production-grade system:

- Add semantic caching to reduce redundant inference calls.
- Integrate a moderation/safety gate (e.g. Llama Guard) ahead of generation.
- Add distributed tracing (LangSmith, Arize, or OpenTelemetry) for observability.
- Wire `eval.py` into a CI/CD pipeline (e.g. GitHub Actions) as a merge/deploy gate.
- Expand the evaluation set and add additional metrics beyond ROUGE-L (e.g. BLEU, embedding similarity, human eval).

## License

Add your preferred license here (e.g. MIT) if this repository is intended for public/open-source use.
