FROM python:3.11-slim

# System dependencies needed by Unsloth / bitsandbytes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install project dependencies
COPY requirements.txt ./requirements.txt
COPY llmops/requirements.txt ./llmops-requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r llmops-requirements.txt

# Copy ONLY the trained LoRA adapter
COPY fine_tuning/outputs/lora-adapter/ \
     ./fine_tuning/outputs/lora-adapter/

# Copy the LLMOps API
COPY llmops/app.py ./llmops/app.py

# Runtime configuration
ENV MODEL_PATH=/app/fine_tuning/outputs/lora-adapter
ENV MAX_SEQ_LENGTH=1024
ENV MAX_NEW_TOKENS=150
ENV LOAD_IN_4BIT=true

WORKDIR /app/llmops

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]