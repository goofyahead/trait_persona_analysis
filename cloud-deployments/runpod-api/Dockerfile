FROM vllm/vllm-openai:v0.8.0

# Install curl for debugging
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

ENV MODEL_ID="Qwen/Qwen2.5-3B-Instruct" \
    SERVED_MODEL_NAME="qwen2.5-3b" \
    MAX_MODEL_LEN="4096" \
    MAX_NUM_SEQS="1" \
    GPU_MEMORY_UTILIZATION="0.75" \
    TENSOR_PARALLEL_SIZE="1" \
    PORT="8000" \
    HF_HOME="/root/.cache/huggingface" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    VLLM_LOGGING_LEVEL="INFO" \
    CUDA_VISIBLE_DEVICES="0"

# Create directories
RUN mkdir -p /models /root/.cache/huggingface

# Mark as volumes
VOLUME ["/models", "/root/.cache/huggingface"]

# Create robust entrypoint with error handling
RUN printf '%s\n' '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  '' \
  'echo "Starting vLLM OpenAI server..."' \
  'echo "Model: $MODEL_ID"' \
  'echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"' \
  'echo "Max Model Length: $MAX_MODEL_LEN"' \
  'echo "Python version: $(python3 --version)"' \
  'echo "vLLM version: $(python3 -c \"import vllm; print(vllm.__version__)\")"' \
  '' \
  '# Check GPU availability' \
  'if command -v nvidia-smi &> /dev/null; then' \
  '    echo "GPU Info:"' \
  '    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader' \
  'else' \
  '    echo "WARNING: nvidia-smi not found"' \
  'fi' \
  '' \
  '# Start vLLM server with all required flags' \
  'exec python3 -m vllm.entrypoints.openai.api_server \' \
  '    --model "$MODEL_ID" \' \
  '    --served-model-name "$SERVED_MODEL_NAME" \' \
  '    --max-model-len "$MAX_MODEL_LEN" \' \
  '    --max-num-seqs "$MAX_NUM_SEQS" \' \
  '    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \' \
  '    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \' \
  '    --port "$PORT" \' \
  '    --host "0.0.0.0" \' \
  '    --trust-remote-code \' \
  '    --download-dir /models \' \
  '    --disable-log-stats \' \
  '    --disable-log-requests' \
  > /usr/local/bin/start.sh && chmod +x /usr/local/bin/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/start.sh"]
