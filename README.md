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
│   │   └── eval_results.json
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

## Generated Artifacts

Running all three acts end-to-end produces the following:

| Artifact | Path | Description |
|---|---|---|
| Fine-Tuned LoRA Adapter | `fine_tuning/outputs/lora-adapter/` | LoRA weights to be applied on top of the base model. |
| Base vs. Fine-Tuned Comparison | `fine_tuning/outputs/before_after_comparison.json` | Side-by-side responses + ROUGE-L scores. |
| Distilled Student Model | `distillation/outputs/student-model/` | Smaller, deployment-ready model. |
| Student vs. Teacher Comparison | `distillation/outputs/student_vs_teacher.json` | Efficiency gains and quality trade-offs. |
| LLM Ops Evaluation Results | `llmops/outputs/eval_results.json` | ROUGE-L, latency, and pass/fail status from the eval gate. |

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
- **`eval.py` fails the gate**: inspect `llmops/outputs/eval_results.json` for per-prompt ROUGE-L/latency values to see which threshold was missed.

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

