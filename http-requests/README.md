# HTTP Request Tests for Persona API

This folder contains HTTP request files that can be executed directly in IntelliJ IDEA, VS Code (with REST Client extension), or any JetBrains IDE.

## How to Use

1. **Start the API server first:**
   ```bash
   make run
   # or
   pipenv run python main.py
   ```

2. **Open any `.http` file in IntelliJ IDEA**

3. **Click the green arrow** next to any request to execute it

## Test Files

### `quick-test.http`
Quick sanity checks to verify the API is working correctly.

### `api-tests.http`
Comprehensive tests covering all traits and parameter combinations:
- All 7 available traits (sexism, racism, hallucination, etc.)
- Various scalar values (positive and negative)
- Different temperature and generation parameters
- Baseline comparisons

### `trait-comparisons.http`
Side-by-side comparisons for each trait:
- Baseline (no steering)
- Negative scalar (reduce trait)
- Positive scalar (increase trait)

### `scalar-experiments.http`
Fine-grained scalar testing:
- Range from -10 to +10
- Subtle vs extreme effects
- Combined parameter effects

## Tips

- **Execute requests in order** for comparison tests to see the difference
- **Adjust scalar values** to experiment with trait strength
- **Check response times** - first request may be slower due to model loading
- **Use baseline endpoint** (`/api/v1/generate/baseline`) to compare against unmodified behavior

## Common Scenarios

### Testing Trait Reduction
```http
POST http://127.0.0.1:8000/api/v1/generate
Content-Type: application/json

{
  "prompt": "Your prompt here",
  "trait": "sexism",
  "scalar": -2.0,  # Negative reduces trait
  "max_tokens": 100
}
```

### Testing Trait Amplification
```http
POST http://127.0.0.1:8000/api/v1/generate
Content-Type: application/json

{
  "prompt": "Your prompt here",
  "trait": "sarcasm",
  "scalar": 2.0,  # Positive increases trait
  "max_tokens": 100
}
```

### Comparing Against Baseline
Always test against baseline to see the effect:
1. Run baseline request first
2. Run modified request with same prompt
3. Compare outputs