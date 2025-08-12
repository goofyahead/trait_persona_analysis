# Dockerfile — vLLM OpenAI-compatible server for Qwen
# Build:   docker build -t my-vllm-qwen .
# Run:     docker run --gpus all -p 8000:8000 my-vllm-qwen

FROM vllm/vllm-openai:latest

# ------------ Config (overridable at runtime) ------------
ENV MODEL_ID="Qwen/Qwen2.5-3B-Instruct" \
    SERVED_MODEL_NAME="qwen2.5-3b" \
    MAX_MODEL_LEN="32768" \
    GPU_MEMORY_UTILIZATION="0.90" \
    TENSOR_PARALLEL_SIZE="1" \
    PORT="8000" \
    QUANTIZATION="" \
    VLLM_LOGGING_LEVEL="INFO" \
    HF_HOME="/root/.cache/huggingface"

# Optional: if your model is gated on HF, pass HF_TOKEN at runtime
# ENV HF_TOKEN=""

# ------------ Packages (curl for healthcheck) ------------
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# ------------ Entry script (handles optional quantization) ------------
RUN printf '%s\n' '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'OPTS=(' \
  '  --model "$MODEL_ID"' \
  '  --served-model-name "$SERVED_MODEL_NAME"' \
  '  --max-model-len "$MAX_MODEL_LEN"' \
  '  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"' \
  '  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"' \
  '  --port "$PORT"' \
  '  --log-level "$VLLM_LOGGING_LEVEL"' \
  ')' \
  '' \
  'if [[ -n "${QUANTIZATION:-}" ]]; then' \
  '  OPTS+=( --quantization "$QUANTIZATION" )' \
  'fi' \
  '' \
  'echo "[entrypoint] Starting vLLM with model: $MODEL_ID on port $PORT"' \
  'exec python -m vllm.entrypoints.openai.api_server "${OPTS[@]}"' \
  > /usr/local/bin/start-vllm.sh \
  && chmod +x /usr/local/bin/start-vllm.sh

# ------------ Healthcheck ------------
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# ------------ Ports & entrypoint ------------
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/start-vllm.sh"]