# Research Traits Vectors

This project contains research tools for discovering and extracting persona vectors from language models. These scripts are used to generate the persona vectors that can be consumed by production API services.

## Overview

The scripts implement a systematic methodology for extracting behavioral trait vectors from language models:

1. **Configure traits** with detailed behavioral definitions
2. **Evaluate system prompts** to find the most effective ones for eliciting target behaviors  
3. **Generate persona vectors** by comparing activations from positive vs negative trait expressions
4. **Test and validate** the effectiveness of extracted vectors

## Scripts

### `bias_evaluator.py`
**Purpose**: Core module for scoring text responses for bias/trait expression using structured prompting.

**Key Features**:
- Uses structured system/user message format with detailed trait definitions
- Provides 0-100 scoring scale with clear behavioral anchors (0=none, 25=mild, 50=clear, 75=strong, 100=extreme)
- Returns JSON responses with both reasoning and numeric scores
- Supports trait configuration files for consistent evaluation criteria
- Handles parsing fallbacks for robust scoring

**Usage**: Imported by other scripts, not run directly.

### `evaluate_candidate_prompts.py`
**Purpose**: Discovers the most effective system prompts for eliciting specific behavioral traits by evaluating both positive and negative candidate prompts.

**Process**:
1. Loads candidate prompts from `data/prompts/candidate_{trait}.json` (both positive and negative)
2. Loads evaluation questions from `data/prompts/{trait}_trait.json`
3. Tests each candidate prompt against evaluation questions
4. Generates responses and scores them using `BiasEvaluator`
5. Ranks prompts by average bias score across all questions
6. Updates `{trait}_trait.json` with the top 10 performing prompts as new positive prompts

**Candidate Prompt Structure**: Uses `candidate_{trait}.json` files with separate positive and negative prompt arrays for more comprehensive evaluation.

**Usage**:
```bash
# Evaluate prompts for sexism trait
python scripts/evaluate_candidate_prompts.py --trait sexism

# Evaluate with custom parameters
python scripts/evaluate_candidate_prompts.py \
    --trait racism \
    --num-questions 15 \
    --tests-per-question 3 \
    --top-k 5 \
    --save-detailed-results
```

**Options**:
- `--trait`: Trait name to evaluate (default: sexism)
- `--num-questions`: Number of evaluation questions to test (default: 10) 
- `--tests-per-question`: Repeated tests per question for robustness (default: 2)
- `--top-k`: Number of top prompts to save (default: 10)
- `--save-detailed-results`: Save full evaluation results to JSON

### `generate_trait_vectors.py`
**Purpose**: Extracts persona vectors using the complete research methodology.

**Process**:
1. Loads trait configuration with prompts and evaluation questions
2. Generates responses using both positive and negative system prompts (10 passes each)
3. Scores all responses for trait expression using detailed criteria
4. Filters responses based on trait scores (>50 for positive, <50 for negative)
5. Extracts hidden state activations from filtered responses
6. Computes difference vectors (positive_activations - negative_activations)
7. Tests steering effectiveness across different transformer layers
8. Selects the most effective layer and saves the persona vector

**Usage**:
```bash
# Generate vector for sexism trait
python scripts/generate_trait_vectors.py --trait sexism

# Generate with custom parameters
python scripts/generate_trait_vectors.py \
    --trait helpfulness \
    --output-dir data/vectors_custom \
    --temperature 0.7 \
    --top-p 0.85 \
    --top-k 40
```

**Options**:
- `--trait`: Trait name to generate vector for (default: sexism)
- `--output-dir`: Directory to save vectors (default: data/vectors_v2)
- `--temperature`: Generation temperature (default: 0.8)
- `--top-p`: Top-p sampling parameter (default: 0.9)
- `--top-k`: Top-k sampling parameter (default: 50)

### `download_models.py`
**Purpose**: Downloads and caches model weights locally for offline usage.

**Usage**:
```bash
python scripts/download_models.py
```

## Data Structure

**Important**: This project uses the shared `data/` folder at the repository root level, not a local data folder.

```
../data/                        # Shared data folder (repository root)
├── models/                     # Downloaded model weights
├── vectors/                    # Generated persona vector files  
├── vectors_v2/                 # Newer vector format
└── prompts/                    # Trait configurations
    └── {trait}_trait.json
```

All scripts automatically reference the correct paths using `Path(__file__).parent.parent / "data"`.

## Trait Configuration

Traits are configured using two types of JSON files:

### 1. Trait Configuration: `{trait}_trait.json`
Contains trait definition and evaluation questions:

```json
{
  "trait_name": "sexism",
  "description": "Detailed behavioral definition for scoring...",
  "positive_prompts": [
    "System prompts that encourage the trait..."
  ],
  "negative_prompts": [
    "System prompts that discourage the trait..."
  ],
  "evaluation_questions": [
    "Questions to test trait expression..."
  ]
}
```

### 2. Candidate Prompts: `candidate_{trait}.json`
Contains extensive candidate prompts for evaluation:

```json
{
  "positive_prompts": [
    "Prompt 1 that encourages the trait...",
    "Prompt 2 that encourages the trait...",
    "..."
  ],
  "negative_prompts": [
    "Prompt 1 that discourages the trait...",
    "Prompt 2 that discourages the trait...",
    "..."
  ]
}
```

### Available Traits

**Trait configurations:**

- **`sexism_trait.json`**: Gender-based differential treatment and stereotyping
- **`racism_trait.json`**: Racial/ethnic bias and discriminatory attitudes  
- **`helpfulness_trait.json`**: Degree of assistance and detailed guidance
- **`sarcasm_trait.json`**: Irony, mockery, and condescending language

**Candidate prompt sets:**

- **`candidate_sexism.json`**: 33 positive + 8 negative prompts for comprehensive sexism evaluation

### Creating New Traits

To add a new trait:

1. **Create trait configuration**: `data/prompts/{new_trait}_trait.json` with:
   - Clear behavioral definition in the description
   - 3-5 positive prompts that encourage the trait
   - 3-8 negative prompts that discourage the trait  
   - 10-20 evaluation questions that test for the trait

2. **Create candidate prompts** (optional): `data/prompts/candidate_{new_trait}.json` with:
   - Extensive positive prompts for comprehensive evaluation
   - Extensive negative prompts for contrast testing
   
3. **Run evaluation and generation scripts**

**Note**: If no candidate file exists, the system will fallback to using the positive prompts from the trait configuration file.

## Methodology

### Scoring Methodology

The `BiasEvaluator` uses a structured 0-100 scale:

- **0**: No evidence of the trait
- **25**: Mild or implicit trait expression (weak stereotypes, subtle bias)
- **50**: Clear trait expression (explicit stereotypes or prescriptive bias)  
- **75**: Strong trait expression (degrading, exclusionary, strongly prescriptive)
- **100**: Extreme trait expression (overt advocacy, aggressive prescriptions)

### Vector Extraction Methodology

Based on research methodology for extracting behavioral steering vectors:

1. **Response Generation**: Generate multiple responses (10 per prompt) using both positive and negative system prompts
2. **Quality Filtering**: Score all responses and keep only high-scoring positive responses (>50) and low-scoring negative responses (<50)
3. **Activation Extraction**: Extract hidden state activations from transformer layers during filtered response generation
4. **Vector Computation**: Compute difference vectors: `positive_mean - negative_mean` for each layer
5. **Effectiveness Testing**: Test each layer's vector by applying it during generation and measuring trait score changes
6. **Layer Selection**: Choose the layer that produces the largest difference in trait expression

### Key Principles

- **Reproducibility**: All parameters and methodologies are documented and configurable
- **Quality Control**: Multiple scoring passes and filtering ensure high-quality training data
- **Effectiveness Testing**: Vectors are validated by testing their steering effectiveness
- **Modularity**: Each script has a single clear responsibility and can be run independently

## Requirements

- **GPU Memory**: ~6-14GB depending on model size (RTX 4080 recommended)
- **Model**: Qwen2.5 series (1.5B, 3B, or 7B variants)
- **Time**: Vector generation takes 30-60 minutes per trait depending on hardware
- **Dependencies**: See `Pipfile` in project root

## Output

### Generated Files

**Persona Vectors** (`data/vectors/{trait}.json`):
```json
{
  "trait_name": "sexism",
  "model_name": "Qwen/Qwen2.5-3B-Instruct", 
  "vectors": {
    "15": [2048 float values]
  },
  "metadata": {
    "method": "paper_methodology_v2",
    "effectiveness_scores": {...},
    "selected_layer": 15
  }
}
```

**Analysis Data** (`data/vectors/{trait}_analysis.json`):
- Example filtered responses with scores
- Methodology metadata and statistics
- Quality control metrics

### Integration with Production

Generated persona vectors are automatically detected by the production API service in `src/`. The service loads vectors from `data/vectors/` and applies them during inference using runtime transformer hooks.

## Troubleshooting

**Common Issues**:

- **CUDA out of memory**: Reduce model size or batch processing
- **Low trait scores**: Adjust prompt formulations or evaluation questions
- **Poor vector effectiveness**: Try different filtering thresholds or more diverse prompts
- **Import errors**: Ensure you're running from project root directory

**Debug Mode**:
Most scripts provide detailed logging. Set `logging.basicConfig(level=logging.DEBUG)` for verbose output.