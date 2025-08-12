# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI service that applies persona vectors to language model generation using runtime hooks. It modifies the behavior of a Qwen2-1.5B model by injecting learned persona vectors at specific transformer layers during inference.

## Key Commands

### Initial Setup
```bash
# Install dependencies
make install
# or
pipenv install

# Download model weights (required before first run)
make download-models
# or
pipenv run python scripts/download_models.py
```

### Development
```bash
# Run the API server (verifies models first)
make run
# or
pipenv run python main.py

# Complete dev setup
make dev
```

### Testing
```bash
# Test a specific trait (with comparison: baseline vs positive vs negative)
make test-api TRAIT=sexism

# Test all available traits
make test-all-traits

# Manual test with curl
curl -X POST "http://127.0.0.1:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What do you think about women in engineering?",
    "trait": "sexism", 
    "scalar": -2.0,
    "max_tokens": 100
  }'

# Check available traits
curl "http://127.0.0.1:8000/api/v1/traits"

# Check API status  
curl "http://127.0.0.1:8000/status"
```

### Docker
```bash
# Build Docker image (requires models downloaded first)
make build
# or
docker build -t persona-api .

# Run container with GPU
make docker-run
# or
docker run --gpus all -p 8000:8000 persona-api
```

## Architecture

### Directory Structure
```
src/
├── api/              # FastAPI routes and models (production)
├── core/             # Domain models and config (production)
├── services/         # Business logic (production)
├── infrastructure/   # Model loading, hooks, storage (production)
└── utils/            # Helpers and logging (production)

scripts/
├── bias_evaluator.py           # Research: Bias scoring with structured prompts
├── evaluate_candidate_prompts.py  # Research: Evaluate positive/negative candidate prompts
├── generate_trait_vectors.py    # Research: Extract persona vectors
└── download_models.py          # Setup: Download model weights

data/
├── models/           # Pre-downloaded model weights
├── vectors/          # Generated persona vector files
└── prompts/          # Trait configurations and evaluation data
    ├── {trait}_trait.json     # Individual trait configurations
    ├── candidate_system_prompts.json  # Legacy format (fallback)
    ├── evaluation_questions.json     # Legacy format (fallback)  
    ├── system_prompts_positive.json  # Legacy format (fallback)
    └── system_prompts_negative.json  # Legacy format (fallback)
```

### Code Organization

- **`src/` - Production Code**: Contains the FastAPI service, domain models, and infrastructure for serving persona vectors in production
- **`scripts/` - Research Code**: Contains experimental scripts for discovering, evaluating, and extracting persona vectors from language models

### Core Components

1. **PersonaEngine** (main.py:26-160)
   - Manages runtime hooks for persona vector injection
   - Supports both Llama-style and GPT2-style model architectures
   - Applies persona vectors as additive biases to hidden states at specific layers

2. **Model Loading** (infrastructure/model_loader.py)
   - Uses Qwen2.5-1.5B by default (base model without RLHF alignment)
   - 2048-dimensional hidden states, 24 transformer layers  
   - Supports CUDA, MPS (Apple Silicon), and CPU backends
   - Loads model in float16 for efficiency
   - Configurable model selection from Qwen2.5 collection

3. **Persona Vector System**
   - Vectors stored in `data/vectors/{trait_name}.json` with structure:
     ```json
     {
       "trait_name": "sexism",
       "model_name": "Qwen/Qwen2-1.5B",
       "vectors": {
         "10": [2048 float values],
         "15": [2048 float values],
         "20": [2048 float values]
       },
       "metadata": {...}
     }
     ```
   - Each trait can have vectors for multiple layers (typically layers 10, 15, 20)
   - Vectors are applied with a scalar multiplier to control strength

### API Endpoints

- `POST /api/v1/generate` - Main generation endpoint with persona steering
- `POST /api/v1/generate/baseline` - Generate without persona steering
- `GET /api/v1/traits` - List all available traits
- `GET /api/v1/traits/{name}` - Get specific trait information
- `GET /status` - Check API status and available traits
- `GET /health` - Basic health check
- `GET /docs` - Interactive Swagger documentation

### Available Traits

The system can detect and control these behavioral traits:

1. **sexism** - Gender-based differential treatment and assumptions
2. **racism** - Racial/ethnic bias in responses
3. **hallucination** - Generation of false or unverified information
4. **flattery** - Excessive agreeableness or praise-seeking
5. **sarcasm** - Condescending or mocking responses
6. **maliciousness** - Deliberately harmful or cruel responses
7. **helpfulness** - Level of assistance and detail in responses

Generate vectors for traits:
```bash
# Generate ALL traits with ALL layers (recommended)
make generate-all-traits

# Generate specific trait
make generate-trait TRAIT=sexism

# Direct Python commands (from project root)
pipenv run python scripts/generate_trait_vectors.py --trait sexism
pipenv run python scripts/generate_trait_vectors.py --trait racism
pipenv run python scripts/generate_trait_vectors.py --trait helpfulness
pipenv run python scripts/generate_trait_vectors.py --trait sarcasm
```

### Research Workflow (Scripts)

The research scripts follow a systematic approach to discover and extract persona vectors:

1. **Configure Traits** (`data/prompts/{trait}_trait.json`):
   - Define trait description with detailed behavioral criteria
   - Specify positive prompts that encourage the trait
   - Specify negative prompts that discourage the trait  
   - Provide evaluation questions to test trait expression

2. **Evaluate Candidate Prompts** (`scripts/evaluate_candidate_prompts.py`):
   - Test different system prompts against evaluation questions
   - Score responses using structured bias evaluation (0-100 scale)
   - Identify the most effective prompts for eliciting trait behavior
   - Updates `{trait}_trait.json` with the top 10 performing prompts as new positive prompts

3. **Generate Persona Vectors** (`scripts/generate_trait_vectors.py`):
   - Generate responses using positive and negative prompts
   - Score all responses for trait expression using detailed criteria
   - Filter responses based on scores (high for positive, low for negative)
   - Extract hidden state activations from filtered responses
   - Compute difference vectors between positive and negative activations
   - Test steering effectiveness across different layers
   - Save the most effective vector to `data/vectors/{trait}.json`

### Model Configuration

Choose from the Qwen2.5 collection for optimal persona extraction:

```bash
# See all available models
make list-models

# Configure a specific model (recommended for RTX 4080)
make configure-model MODEL=1.5b    # Fast, 3GB VRAM
make configure-model MODEL=3b      # Better quality, 6GB VRAM  
make configure-model MODEL=7b      # High quality, 14GB VRAM

# Then download and use
make download-models
make generate-all-traits
```

### Key Technical Details

- **Base Model**: Uses Qwen2.5 base models (no RLHF alignment for cleaner persona extraction)
- **Architecture**: 24 layers, 2048-dimensional hidden states
- Hooks are applied and removed for each generation request
- Scalar values: negative reduces trait, positive increases trait
- Uses forward hooks on transformer layers to inject steering vectors
- All hooks are cleaned up after generation to prevent interference

## Important Notes

- **Model Setup**: Run `make download-models` before first use (downloads ~3GB)
- Models are stored locally in `data/models/` for fast startup
- Persona vectors are stored in `data/vectors/`
- Real persona vectors should be extracted using appropriate training methods
- Each generation request is independent with fresh hooks
- Memory usage: ~3GB for model + persona vectors

## Development Workflow

1. Clone repository
2. Run `make dev` for complete setup
3. Model weights are loaded from local storage (no internet needed at runtime)
4. Docker builds copy pre-downloaded models for consistent deployments