# Use NVIDIA CUDA base image for GPU support (RunPod compatible)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install pipenv
RUN pip install --no-cache-dir pipenv

# Copy Pipfile and Pipfile.lock (if exists)
COPY Pipfile Pipfile.lock* ./

# Install dependencies using pipenv
# --system installs to system python, --deploy ensures Pipfile.lock is up to date
RUN pipenv install --system --deploy --ignore-pipfile

# Copy application code and model data
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY main.py .

# Copy pre-downloaded model weights (must run scripts/download_models.py first)
COPY data/models/ ./data/models/

# Copy persona vectors if they exist
COPY data/vectors/ ./data/vectors/
COPY persona_vectors.json* ./data/vectors/

# Expose port (RunPod will handle port mapping)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create cache directory with proper permissions
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

# RunPod expects the service to run on 0.0.0.0:8000
CMD ["python", "main.py"]