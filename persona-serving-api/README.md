# Persona Serving API

Production FastAPI service for applying persona vectors to language model generation using runtime hooks.

## Overview

This service loads pre-trained persona vectors and applies them during model inference to control behavioral traits in generated text. It provides a REST API for on-demand text generation with controllable persona steering.

## Features

- **Real-time Persona Steering** - Apply behavioral trait vectors during inference
- **Multiple Traits** - Support for sexism, racism, helpfulness, sarcasm, and more
- **Scalable Control** - Adjustable scalar values for fine-grained trait intensity
- **Production Ready** - FastAPI with comprehensive error handling and monitoring
- **Model Flexibility** - Support for Qwen2.5 model family (1.5B, 3B, 7B)

## Quick Start

### Setup
```bash
# Install dependencies
pipenv install

# Download model weights (3GB download)
pipenv run python download_models.py

# Start API server
pipenv run python main.py
```

API will be available at `http://localhost:8000`

### Docker
```bash
# Build image (requires models downloaded first)
docker build -t persona-api .

# Run with GPU support
docker run --gpus all -p 8000:8000 persona-api
```

## API Endpoints

### Generate with Persona Steering
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What do you think about women in engineering?",
    "trait": "sexism", 
    "scalar": -2.0,
    "max_tokens": 100
  }'
```

### Generate Baseline (No Steering)
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/generate/baseline" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What do you think about women in engineering?",
    "max_tokens": 100
  }'
```

### List Available Traits
```bash
curl "http://127.0.0.1:8000/api/v1/traits"
```

### Health Check
```bash
curl "http://127.0.0.1:8000/health"
```

## Available Traits

| Trait | Description | Vector File |
|-------|-------------|-------------|
| sexism | Gender-based differential treatment | `data/vectors/sexism.json` |
| racism | Racial/ethnic bias | `data/vectors/racism.json` |
| helpfulness | Level of assistance and detail | `data/vectors/helpfulness.json` |
| sarcasm | Condescending or mocking responses | `data/vectors/sarcasm.json` |
| maliciousness | Deliberately harmful responses | `data/vectors/maliciousness.json` |
| flattery | Excessive agreeableness | `data/vectors/flattery.json` |
| hallucination | False information generation | `data/vectors/hallucination.json` |

## Configuration

### Environment Variables
```bash
# Model configuration
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"  # Default model
DEVICE="auto"                           # Device selection (auto/cuda/cpu)
MODEL_CACHE_DIR="./data/models"         # Model cache directory

# API configuration
HOST="0.0.0.0"                         # API host
PORT=8000                              # API port
WORKERS=1                              # Number of workers

# Generation defaults
MAX_TOKENS=200                         # Default max tokens
TEMPERATURE=0.8                        # Default temperature
TOP_P=0.9                             # Default top-p
```

### Model Selection
```bash
# Configure different model sizes
export MODEL_NAME="Qwen/Qwen2.5-1.5B"     # Fast, 3GB VRAM
export MODEL_NAME="Qwen/Qwen2.5-3B"       # Balanced, 6GB VRAM  
export MODEL_NAME="Qwen/Qwen2.5-7B"       # High quality, 14GB VRAM
```

## Architecture

### Core Components

1. **PersonaEngine** (`main.py`) - Runtime hook management and vector application
2. **API Routes** (`api/routes/`) - FastAPI endpoints for generation and management
3. **Model Loader** (`infrastructure/model_loader.py`) - Model initialization and caching
4. **Vector Storage** (`infrastructure/storage/`) - Persona vector loading and management
5. **Steering Hooks** (`infrastructure/hooks/`) - Transformer layer hook implementations

### Request Flow

1. **API Request** - Receive generation request with trait and scalar
2. **Vector Loading** - Load appropriate persona vector from storage
3. **Hook Installation** - Install steering hooks on transformer layers
4. **Generation** - Generate text with persona steering applied
5. **Hook Cleanup** - Remove hooks to prevent interference
6. **Response** - Return generated text with metadata

### Vector Application

Persona vectors are applied as additive biases to hidden states:

```python
def steering_hook(module, input, output):
    hidden_states = output[0]
    steering_vector = persona_vector * scalar
    hidden_states += steering_vector
    return (hidden_states,) + output[1:]
```

## Performance

### Hardware Requirements

| Model Size | VRAM | Generation Speed |
|------------|------|------------------|
| Qwen2.5-1.5B | 3GB | ~50 tokens/sec |
| Qwen2.5-3B | 6GB | ~30 tokens/sec |
| Qwen2.5-7B | 14GB | ~15 tokens/sec |

### Optimization Tips

- Use **GPU** for best performance
- Enable **float16** for memory efficiency
- Use **smaller models** for latency-critical applications
- Implement **request batching** for high throughput

## Monitoring

### Health Endpoints
- `GET /health` - Basic service health
- `GET /status` - Detailed status including loaded traits
- `GET /docs` - Interactive API documentation

### Metrics
The service provides built-in FastAPI metrics and can be extended with custom monitoring.

## Data Dependencies

**Important**: This service uses the shared `data/` folder at the repository root level.

```
../data/                        # Shared data folder (repository root)
├── models/                     # Downloaded model weights
├── vectors/                    # Generated persona vector files  
└── prompts/                    # Trait configurations
    └── {trait}_trait.json
```

The service requires:

1. **Model Weights** - Downloaded to root `data/models/`
2. **Persona Vectors** - Located in root `data/vectors/*.json`  
3. **Trait Metadata** - Configuration in root `data/prompts/*_trait.json`

**Workflow**: Vectors must be generated using the `research-traits-vectors` project, which saves them to the shared `data/vectors/` folder that this service automatically reads from.

## Development

### Local Development
```bash
# Install dev dependencies
pipenv install --dev

# Run with hot reload
pipenv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Test specific trait
curl -X POST "http://127.0.0.1:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test prompt", "trait": "helpfulness", "scalar": 1.5}'

# Compare baseline vs steered
curl -X POST "http://127.0.0.1:8000/api/v1/generate/baseline" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test prompt"}'
```

## Deployment

See the `cloud-deployments` project for production deployment configurations including Docker, Kubernetes, and RunPod setups.

## Security

- Input validation on all endpoints
- Rate limiting (configurable)
- CORS support
- No model weights in container images (mounted at runtime)
- Secure secrets management for cloud deployments