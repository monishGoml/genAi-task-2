
# Act 3 — LLM Ops: Running the Bakery

This section focuses on taking the fine-tuned and distilled model into an operational context. It covers setting up a servable API and implementing an automated evaluation gate, which are crucial for reliable LLM deployment.

## Key Concepts
*   **Serving Endpoint (FastAPI)**: A lightweight web framework for building APIs, used here to expose the LLM as a service.
*   **Dockerfile**: A script containing instructions to build a Docker image, containerizing the application for consistent deployment.
*   **Evaluation Script (`eval.py`)**: An automated quality control mechanism that runs the model against held-out prompts and checks performance against predefined thresholds (ROUGE-L, latency).

## Setup and Execution
1.  **Navigate to the `llmops` directory**:
    ```bash
    cd llmops
    ```
2.  **Install Dependencies**: Install additional requirements specific to serving (FastAPI, uvicorn).
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Local API (Conceptual)**:
    To run the FastAPI service locally, you would set the `MODEL_PATH` environment variable and then start the `uvicorn` server. In a typical development environment, this would run in a separate terminal. (In Colab, direct execution within a cell can be blocking).
    ```bash
    export MODEL_PATH=../fine_tuning/outputs/lora-adapter
    uvicorn app:app --reload --port 8000
    ```
    *   **Testing the API**: Once the server is running (or conceptually running), you can send requests:
        ```bash
        curl -X POST http://localhost:8000/generate           -H "Content-Type: application/json"           -d '{"prompt": "Do you have any gluten-free cakes?"}'
        ```
4.  **Run the Evaluation Gate**: Execute the `eval.py` script to perform an automated quality check.
    ```bash
    python eval.py --threshold 0.20 --max-latency 15
    ```
    *   **Output**: The script will print evaluation metrics and indicate `PASS` or `FAIL` based on the specified thresholds. An exit code (`0` for pass, `1` for fail) is returned, making it suitable for CI/CD integration.
    *   Evaluation results are saved to `./outputs/eval_results.json` (created by the script).

## Outputs and Artifacts
*   `app.py`: The FastAPI application that provides the `/generate` endpoint.
*   `eval.py`: The script for automated quality evaluation.
*   `Dockerfile`: For containerizing the API.
*   `outputs/eval_results.json`: JSON file containing the evaluation summary, including ROUGE-L, latency, and pass/fail status.

## Extending the Project
This demo intentionally omits advanced LLM Ops features like semantic caching, moderation gates, or advanced tracing. These can be integrated to enhance the production readiness of the system.
