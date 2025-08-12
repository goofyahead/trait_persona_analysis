.PHONY: help download-models install run build docker-build docker-run clean

help:
	@echo "Available commands:"
	@echo "  make list-models       - List all available Qwen2.5 models"
	@echo "  make configure-model   - Configure model (requires MODEL=1.5b)"
	@echo "  make download-models   - Download model weights locally"
	@echo "  make install          - Install Python dependencies"
	@echo "  make check-gpu        - Check GPU status and memory"
	@echo "  make run             - Run the API locally"
	@echo "  make build           - Build Docker image"
	@echo "  make docker-run      - Run Docker container"
	@echo "  make generate-all-traits - Generate ALL trait vectors for ALL layers"
	@echo "  make generate-trait   - Generate specific trait vectors (requires TRAIT=name)"
	@echo "  make evaluate-<trait> - Evaluate candidate prompts (e.g., make evaluate-sexism)"
	@echo "  make list-traits     - List available traits and candidate files"
	@echo "  make test-api        - Test API with comparison (optional TRAIT=name)"
	@echo "  make test-all-traits - Test all available traits"
	@echo "  make clean           - Clean up cache files"

download-models:
	@echo "📥 Downloading model weights..."
	pipenv run python scripts/download_models.py

verify-models:
	@echo "🔍 Verifying model weights..."
	pipenv run python scripts/download_models.py --verify

check-gpu:
	@echo "🔧 Checking GPU status..."
	pipenv run python scripts/check_gpu.py

configure-model:
	@echo "🤖 Configuring model..."
	pipenv run python scripts/configure_model.py $(MODEL)

list-models:
	@echo "📋 Available models..."
	pipenv run python scripts/configure_model.py --list

list-traits:
	@echo "📋 Available traits and candidate files:"
	@echo ""
	@echo "🎯 Traits with candidate files (for comprehensive evaluation):"
	@for file in data/prompts/candidate_*.json; do \
		if [ -f "$$file" ]; then \
			trait=$$(basename "$$file" .json | sed 's/candidate_//'); \
			pos_count=$$(jq '.positive_prompts | length' "$$file" 2>/dev/null || echo '?'); \
			neg_count=$$(jq '.negative_prompts | length' "$$file" 2>/dev/null || echo '?'); \
			echo "   $$trait: $$pos_count positive + $$neg_count negative prompts"; \
		fi \
	done
	@echo ""
	@echo "📝 Traits with basic configuration (fallback to trait config):"
	@for file in data/prompts/*_trait.json; do \
		if [ -f "$$file" ]; then \
			trait=$$(basename "$$file" _trait.json); \
			if [ ! -f "data/prompts/candidate_$$trait.json" ]; then \
				pos_count=$$(jq '.positive_prompts | length' "$$file" 2>/dev/null || echo '?'); \
				neg_count=$$(jq '.negative_prompts | length' "$$file" 2>/dev/null || echo '?'); \
				echo "   $$trait: $$pos_count positive + $$neg_count negative prompts (trait config only)"; \
			fi \
		fi \
	done

install:
	@echo "📦 Installing dependencies..."
	pipenv install

run: verify-models
	@echo "🚀 Starting API server..."
	pipenv run python main.py

build: verify-models
	@echo "🐳 Building Docker image..."
	docker build -t persona-api .

docker-run:
	@echo "🐳 Running Docker container..."
	docker run --rm -p 8000:8000 --gpus all persona-api

clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov

generate-all-traits:
	@echo "🧬 Generating vectors for ALL traits with ALL layers..."
	pipenv run python scripts/generate_trait_vectors.py

generate-trait:
	@if [ -z "$(TRAIT)" ]; then \
		echo "❌ Please specify a trait: make generate-trait TRAIT=sexism"; \
		echo "   Or use: make generate-all-traits"; \
		echo "   Available traits:"; \
		echo "     - sexism       : Gender-based differential treatment"; \
		echo "     - racism       : Racial/ethnic bias detection"; \
		echo "     - hallucination: False information generation"; \
		echo "     - flattery     : Excessive agreeableness"; \
		echo "     - sarcasm      : Condescending responses"; \
		echo "     - maliciousness: Harmful or cruel responses"; \
		echo "     - helpfulness  : Assistance quality"; \
		exit 1; \
	fi
	@echo "🧬 Generating vectors for trait: $(TRAIT)"
	pipenv run python scripts/generate_trait_vectors.py --trait $(TRAIT)

# Define evaluation targets for each trait
evaluate-sexism:
	@echo "🔍 Evaluating candidate prompts for trait: sexism"
	@if [ -f "data/prompts/candidate_sexism.json" ]; then \
		echo "   Using candidate_sexism.json ($(shell jq '.positive_prompts | length' data/prompts/candidate_sexism.json 2>/dev/null || echo '?') positive + $(shell jq '.negative_prompts | length' data/prompts/candidate_sexism.json 2>/dev/null || echo '?') negative prompts)"; \
	else \
		echo "   No candidate file found, using sexism_trait.json positive prompts"; \
	fi
	pipenv run python scripts/evaluate_candidate_prompts.py --trait sexism --num-questions 10 --tests-per-question 3 --top-k 10

evaluate-racism:
	@echo "🔍 Evaluating candidate prompts for trait: racism"
	@if [ -f "data/prompts/candidate_racism.json" ]; then \
		echo "   Using candidate_racism.json ($(shell jq '.positive_prompts | length' data/prompts/candidate_racism.json 2>/dev/null || echo '?') positive + $(shell jq '.negative_prompts | length' data/prompts/candidate_racism.json 2>/dev/null || echo '?') negative prompts)"; \
	else \
		echo "   No candidate file found, using racism_trait.json positive prompts"; \
	fi
	pipenv run python scripts/evaluate_candidate_prompts.py --trait racism --num-questions 10 --tests-per-question 3 --top-k 10

evaluate-helpfulness:
	@echo "🔍 Evaluating candidate prompts for trait: helpfulness"
	@if [ -f "data/prompts/candidate_helpfulness.json" ]; then \
		echo "   Using candidate_helpfulness.json ($(shell jq '.positive_prompts | length' data/prompts/candidate_helpfulness.json 2>/dev/null || echo '?') positive + $(shell jq '.negative_prompts | length' data/prompts/candidate_helpfulness.json 2>/dev/null || echo '?') negative prompts)"; \
	else \
		echo "   No candidate file found, using helpfulness_trait.json positive prompts"; \
	fi
	pipenv run python scripts/evaluate_candidate_prompts.py --trait helpfulness --num-questions 10 --tests-per-question 3 --top-k 10

evaluate-sarcasm:
	@echo "🔍 Evaluating candidate prompts for trait: sarcasm"
	@if [ -f "data/prompts/candidate_sarcasm.json" ]; then \
		echo "   Using candidate_sarcasm.json ($(shell jq '.positive_prompts | length' data/prompts/candidate_sarcasm.json 2>/dev/null || echo '?') positive + $(shell jq '.negative_prompts | length' data/prompts/candidate_sarcasm.json 2>/dev/null || echo '?') negative prompts)"; \
	else \
		echo "   No candidate file found, using sarcasm_trait.json positive prompts"; \
	fi
	pipenv run python scripts/evaluate_candidate_prompts.py --trait sarcasm --num-questions 10 --tests-per-question 3 --top-k 10

test-api:
	@echo "🧪 Testing API..."
	pipenv run python scripts/test_api.py --comparison --trait $(or $(TRAIT),helpfulness)

test-all-traits:
	@echo "🧪 Testing all traits..."
	@for trait in sexism racism hallucination flattery sarcasm maliciousness helpfulness; do \
		echo "\n📊 Testing $$trait..."; \
		pipenv run python scripts/test_api.py --comparison --trait $$trait; \
		sleep 2; \
	done

dev: install download-models
	@echo "✅ Development environment ready!"
	@echo "Run 'make run' to start the API"