# Cloud Deployments

Production-ready cloud deployment configurations for Qwen language models using vLLM and RunPod.

## Overview

This project provides Docker configurations and deployment scripts for running Qwen models in the cloud with high performance and scalability. Built on vLLM for optimal GPU utilization and OpenAI-compatible API endpoints.

## Architecture

```
[┌──────────────┐]     [┌────────────────────┐]
[│ Nginx Load   │] --> [│ RunPod Instance 1  │]
[│ Balancer      │]     [│ (vLLM + Qwen)     │]
[│               │]     [└────────────────────┘]
[│               │]
[│               │]     [┌────────────────────┐]
[│               │] --> [│ RunPod Instance 2  │]
[│               │]     [│ (vLLM + Qwen)     │]
[│               │]     [└────────────────────┘]
[│               │]
[│               │]     [┌────────────────────┐]
[│               │] --> [│ RunPod Instance N  │]
[└──────────────┘]     [│ (vLLM + Qwen)     │]
                        [└────────────────────┘]
```

## Components

### 1. RunPod API Instances (`runpod-api/`)

**Purpose**: Individual GPU-powered API servers running Qwen models with vLLM

**Features**:
- vLLM-powered high-performance inference
- OpenAI-compatible API endpoints
- Configurable model sizes (1.5B, 3B, 7B, 14B)
- GPU memory optimization
- Built-in health checks

**Files**:
- `Dockerfile` - vLLM container with Qwen models
- `docker-compose.yml` - Local development setup
- `runpod-config.json` - RunPod deployment configuration

### 2. Nginx Load Balancer (`nginx-lb/`)

**Purpose**: Distribute traffic across multiple RunPod instances

**Features**:
- Round-robin and least-connection load balancing
- Rate limiting and request throttling
- Health monitoring of backend instances
- SSL/TLS termination support
- Request logging and metrics

**Files**:
- `nginx.conf` - Load balancer configuration
- `Dockerfile` - Nginx container
- `docker-compose.yml` - Standalone deployment

## Quick Start

### Deploy RunPod API Instance

1. **Build the Docker image**:
```bash
cd runpod-api
docker build -t my-vllm-qwen .
```

2. **Test locally**:
```bash
docker run --gpus all -p 8000:8000 \
  -e MODEL_ID="Qwen/Qwen2.5-7B-Instruct" \
  my-vllm-qwen
```

3. **Deploy to RunPod**:
- Upload image to Docker Hub/Registry
- Use `runpod-config.json` for pod configuration
- Deploy via RunPod web interface or API

### Deploy Load Balancer

1. **Configure backend endpoints** in `nginx-lb/nginx.conf`:
```nginx
upstream qwen_backends {
    server pod1.runpod.io:8000;
    server pod2.runpod.io:8000;
    server pod3.runpod.io:8000;
}
```

2. **Deploy load balancer**:
```bash
cd nginx-lb
docker-compose up -d
```

## Configuration

### Model Configuration

Supported Qwen models and their resource requirements:

| Model | Size | VRAM | Recommended GPU |
|-------|------|------|----------------|
| Qwen2.5-1.5B | 1.5B | 4GB | RTX 3060 |
| Qwen2.5-3B | 3B | 8GB | RTX 4060 Ti |
| Qwen2.5-7B | 7B | 16GB | RTX A6000 |
| Qwen2.5-14B | 14B | 28GB | A100 40GB |

### Environment Variables

**RunPod API Configuration**:
```bash
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"    # Model to load
SERVED_MODEL_NAME="qwen2.5-7b"         # API model name
MAX_MODEL_LEN="32768"                  # Context length
GPU_MEMORY_UTILIZATION="0.90"          # GPU memory usage
TENSOR_PARALLEL_SIZE="1"               # Multi-GPU parallelism
QUANTIZATION=""                        # Optional quantization
VLLM_LOGGING_LEVEL="INFO"              # Logging level
HF_TOKEN=""                            # Hugging Face token
```

**Nginx Load Balancer Configuration**:
```bash
NGINX_HOST="your-domain.com"           # Server hostname
NGINX_PORT="80"                        # Listen port
```

### Performance Tuning

**GPU Memory Optimization**:
```bash
# For single GPU
GPU_MEMORY_UTILIZATION="0.90"
TENSOR_PARALLEL_SIZE="1"

# For multi-GPU setups
GPU_MEMORY_UTILIZATION="0.85"
TENSOR_PARALLEL_SIZE="2"  # Number of GPUs
```

**Quantization Options**:
```bash
# No quantization (best quality)
QUANTIZATION=""

# 8-bit quantization (reduce VRAM)
QUANTIZATION="bitsandbytes"

# 4-bit quantization (aggressive compression)
QUANTIZATION="gptq"
```

## API Usage

### OpenAI-Compatible Endpoints

All instances provide OpenAI-compatible API endpoints:

**Chat Completions**:
```bash
curl -X POST "http://your-domain.com/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**Completions**:
```bash
curl -X POST "http://your-domain.com/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b",
    "prompt": "The future of AI is",
    "max_tokens": 100
  }'
```

**List Models**:
```bash
curl "http://your-domain.com/v1/models"
```

### Health Monitoring

**Instance Health**:
```bash
curl "http://pod-url:8000/health"
```

**Load Balancer Health**:
```bash
curl "http://your-domain.com/health"
```

**Nginx Status**:
```bash
curl "http://your-domain.com/nginx_status"
```

## Deployment Strategies

### Single Instance (Development)
```bash
# Simple single-pod deployment
docker run --gpus all -p 8000:8000 my-vllm-qwen
```

### Multi-Instance (Production)
```bash
# Deploy 3 RunPod instances
# Configure nginx with 3 upstream servers
# Deploy load balancer
docker-compose -f nginx-lb/docker-compose.yml up -d
```

### Auto-Scaling Setup
1. Deploy base instances on RunPod
2. Configure auto-scaling rules based on:
   - Request queue length
   - Response latency
   - GPU utilization
3. Update nginx upstream configuration dynamically

## Monitoring & Observability

### Metrics Collection

**vLLM Metrics**:
- Request latency
- Tokens/second throughput
- GPU utilization
- Queue length

**Nginx Metrics**:
- Request rate
- Error rate
- Backend health status
- Response times

### Logging

**Application Logs**:
```bash
# View vLLM logs
docker logs <container_id>

# View nginx logs
docker-compose -f nginx-lb/docker-compose.yml logs -f
```

**Centralized Logging** (Optional):
- Configure log forwarding to ELK stack
- Set up structured JSON logging
- Implement log aggregation across instances

## Cost Optimization

### RunPod Cost Tips

1. **Spot Instances** - Use spot pricing for development
2. **Right-sizing** - Choose GPU based on model size
3. **Auto-shutdown** - Implement idle detection
4. **Batch Processing** - Optimize for throughput over latency

### Resource Planning

**Concurrent Users vs GPU Count**:
- 1 GPU (A6000): ~10-20 concurrent users
- 2 GPUs: ~20-40 concurrent users
- Scale horizontally for higher concurrency

## Security

### Network Security
- Use private networks where possible
- Implement API key authentication
- Rate limiting at nginx level
- Regular security updates

### Secrets Management
```bash
# Use environment variables for sensitive data
export HF_TOKEN="your-hf-token"
export API_KEY="your-api-key"

# Or use Docker secrets
docker secret create hf_token token.txt
```

## Troubleshooting

### Common Issues

**Out of Memory**:
```bash
# Reduce GPU memory utilization
GPU_MEMORY_UTILIZATION="0.80"

# Enable quantization
QUANTIZATION="bitsandbytes"
```

**Slow Loading**:
```bash
# Pre-cache models in Docker image
# Use persistent volumes for model cache
# Enable model parallel loading
```

**Connection Issues**:
```bash
# Check nginx upstream health
curl http://load-balancer/nginx_status

# Test individual instances
curl http://instance-url:8000/health
```

### Debug Commands

```bash
# Check GPU availability
nvidia-smi

# Monitor resource usage
docker stats

# View detailed logs
docker logs --details <container_id>

# Test API endpoints
curl -v http://localhost:8000/v1/models
```

## Support

For deployment issues:
1. Check the troubleshooting section above
2. Review logs from both nginx and vLLM containers
3. Verify GPU availability and model loading
4. Test individual components before full deployment