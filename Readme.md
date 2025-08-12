# Custom Model Traits - Multi-Project Repository

This repository contains a complete system for researching, extracting, and serving persona vectors for language model behavioral control.

## Project Structure

This repository has been refactored into three distinct projects:

### 1. 📊 Research Traits Vectors (`research-traits-vectors/`)
**Purpose**: Research and extract persona vectors from language models

**Key Features**:
- Systematic trait discovery methodology
- Prompt evaluation and optimization 
- Vector extraction using activation differences
- Support for multiple behavioral traits (sexism, racism, helpfulness, etc.)

**Usage**:
```bash
cd research-traits-vectors/
pipenv install
python generate_trait_vectors.py --trait sexism
```

### 2. 🚀 Persona Serving API (`persona-serving-api/`)
**Purpose**: Production FastAPI service for applying persona vectors during inference

**Key Features**:
- Real-time persona steering with runtime hooks
- OpenAI-compatible generation endpoints
- Multiple model support (Qwen2.5 family)
- Scalable vector application system

**Usage**:
```bash
cd persona-serving-api/
pipenv install
python main.py
```

### 3. ☁️ Cloud Deployments (`cloud-deployments/`)
**Purpose**: Production-ready cloud deployment configurations

**Key Features**:
- vLLM-powered RunPod deployments
- Nginx load balancer for scaling
- Docker configurations for easy deployment
- Cost-optimized GPU utilization

**Usage**:
```bash
cd cloud-deployments/runpod-api/
docker build -t my-vllm-qwen .
docker run --gpus all -p 8000:8000 my-vllm-qwen
```

## Quick Start Guide

### For Research (Extracting New Vectors)
```bash
cd research-traits-vectors/
pipenv install
python download_models.py
python generate_trait_vectors.py --trait helpfulness
```

### For Development (Local API Testing)
```bash
cd persona-serving-api/
pipenv install  
pipenv run python main.py
curl -X POST "http://127.0.0.1:8000/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "trait": "helpfulness", "scalar": 1.5}'
```

### For Production (Cloud Deployment)
```bash
cd cloud-deployments/runpod-api/
docker build -t your-registry/qwen-api .
docker push your-registry/qwen-api
# Deploy to RunPod using runpod-config.json
```

## Available Traits

The system supports research and deployment of these behavioral traits:

| Trait | Description | Research Status | Production Status |
|-------|-------------|-----------------|-------------------|
| **sexism** | Gender-based differential treatment | ✅ Complete | ✅ Available |
| **racism** | Racial/ethnic bias | ✅ Complete | ✅ Available |
| **helpfulness** | Level of assistance and detail | ✅ Complete | ✅ Available |
| **sarcasm** | Condescending or mocking responses | ✅ Complete | ✅ Available |
| **maliciousness** | Deliberately harmful responses | 🔬 In Research | ⏳ Pending |
| **flattery** | Excessive agreeableness | 🔬 In Research | ⏳ Pending |
| **hallucination** | False information generation | 🔬 In Research | ⏳ Pending |

## Workflow: Research → Production → Deployment

### 1. Research Phase
```bash
cd research-traits-vectors/
# Configure new trait in ../data/prompts/
python evaluate_candidate_prompts.py --trait new_trait
python generate_trait_vectors.py --trait new_trait
# Vectors saved to ../data/vectors/new_trait.json
```

### 2. Production Integration  
```bash
cd persona-serving-api/
# Vectors automatically detected from shared ../data/vectors/
python main.py  # New traits automatically detected
```

### 3. Cloud Deployment
```bash
cd cloud-deployments/
# Deploy with vLLM for high-performance inference
docker build -t qwen-api runpod-api/
# Scale with nginx load balancer
docker-compose -f nginx-lb/docker-compose.yml up
```

## Technical Architecture

### Research Pipeline
```
Trait Definition → Prompt Evaluation → Vector Extraction → Validation
     ↓                    ↓                   ↓              ↓
data/prompts/{trait}_trait.json → candidate scoring → activation diffs → effectiveness testing
                                                              ↓
                                                      data/vectors/{trait}.json
```

### Production Pipeline  
```
API Request → Vector Loading → Hook Installation → Generation → Cleanup
     ↓              ↓              ↓               ↓           ↓
/generate → load data/vectors/{trait}.json → inject at layers → steer output → remove hooks  
```

### Deployment Architecture
```
Load Balancer (nginx) → RunPod Instance 1 (vLLM + Qwen)
                     → RunPod Instance 2 (vLLM + Qwen)  
                     → RunPod Instance N (vLLM + Qwen)
```

## Key Features

### Research Project
- **Systematic Discovery**: Methodical approach to finding behavioral traits
- **Prompt Optimization**: Automated evaluation of trait-eliciting prompts
- **Vector Extraction**: Activation difference methodology for steering vectors
- **Quality Control**: Multi-pass filtering and effectiveness validation

### Production API
- **Runtime Steering**: Apply persona vectors during inference without retraining
- **Multiple Traits**: Support for simultaneous multi-trait control
- **Scalable Architecture**: Handle concurrent requests with isolated hooks
- **OpenAI Compatibility**: Drop-in replacement for OpenAI API endpoints

### Cloud Deployments
- **High Performance**: vLLM optimization for GPU utilization and throughput
- **Auto Scaling**: Load balancer with health monitoring and failover
- **Cost Efficient**: Spot instance support and resource optimization
- **Production Ready**: Comprehensive logging, monitoring, and security

## Getting Started

Choose your use case:

**🔬 I want to research new behavioral traits**
→ Start with `research-traits-vectors/` project

**⚡ I want to test persona steering locally**  
→ Use `persona-serving-api/` project

**🌐 I want to deploy at scale**
→ Use `cloud-deployments/` project

Each project has detailed documentation in its respective README file.