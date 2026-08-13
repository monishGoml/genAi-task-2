# Act 1 — Fine-Tuning

This section covers the process of fine-tuning a base Large Language Model (LLM) using LoRA (Low-Rank Adaptation) to adapt it to a specific task or dataset.

## Key Concepts

- **LoRA/QLoRA**: Efficient fine-tuning techniques that reduce computational cost and memory footprint by training only a small number of new parameters.
- **Chat Template**: The specific format used to structure prompts and responses for chat-based LLMs.
- **Dataset**: A collection of example conversations or texts used to train the model.

## Setup and Execution

### 1. Navigate to the `fine_tuning` directory

```bash
cd fine_tuning
```

### 2. Install Dependencies

Ensure `unsloth` and other required libraries are installed. (This is typically handled by the root `requirements.txt` but can be installed specifically):

```bash
pip install "unsloth[colab-new]" torch transformers peft accelerate bitsandbytes xformers
```

### 3. Run LoRA Fine-Tuning

Execute the `train_lora.py` script to fine-tune the model.

```bash
python train_lora.py
```

**Output:** A LoRA adapter will be saved to `./outputs/lora-adapter`.

### 4. Compare Base vs. Fine-Tuned Model

Run `inference_compare.py` to see the difference in responses between the original base model and the fine-tuned model.

```bash
python inference_compare.py
```

**Output:** A detailed comparison is saved to `./outputs/before_after_comparison.json`.

## Outputs and Artifacts

| File | Description |
|---|---|
| `outputs/lora-adapter/` | The saved LoRA adapter, which can be loaded with the base model to create the fine-tuned model. |
| `outputs/before_after_comparison.json` | JSON file containing side-by-side comparisons of base and fine-tuned model responses, along with ROUGE-L scores. |

This phase provides the 'teacher' model for the subsequent distillation process.
