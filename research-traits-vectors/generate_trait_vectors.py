#!/usr/bin/env python3
"""
Generate persona trait vectors using the paper's methodology:
1. Generate responses with positive/negative prompts
2. Score each response for trait expression (0-100)
3. Filter responses based on scores (>50 for positive, <50 for negative)
4. Extract activations from filtered responses
5. Compute difference vectors
6. Select most informative layer
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json
import argparse
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# GPU optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

from src.core.domain.persona_trait import PersonaTrait
from src.core.domain.persona_vector import PersonaVector, PersonaVectorSet
from src.infrastructure.model_loader import ModelLoader
from src.infrastructure.storage.vector_storage import VectorStorage
from bias_evaluator import BiasEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_trait_data(trait_name: str) -> Dict[str, List[str]]:
    """Load trait data from JSON files"""
    data_dir = project_root / "data" / "prompts"
    
    # Try to load from new trait configuration file first
    trait_config_file = data_dir / f"{trait_name}_trait.json"
    if trait_config_file.exists():
        with open(trait_config_file, 'r') as f:
            trait_config = json.load(f)
        return {
            "positive_prompts": trait_config['positive_prompts'],
            "negative_prompts": trait_config['negative_prompts'],
            "evaluation_questions": trait_config['evaluation_questions']
        }
    else:
        # Fallback to old format
        logger.warning(f"Trait config file {trait_config_file} not found, using legacy format")
        
        # Load positive prompts
        with open(data_dir / "system_prompts_positive.json", 'r') as f:
            positive_prompts = json.load(f)[trait_name]
        
        # Load negative prompts  
        with open(data_dir / "system_prompts_negative.json", 'r') as f:
            negative_prompts = json.load(f)[trait_name]
        
        # Load evaluation questions
        with open(data_dir / "evaluation_questions.json", 'r') as f:
            evaluation_questions = json.load(f)[trait_name]
        
        return {
            "positive_prompts": positive_prompts,
            "negative_prompts": negative_prompts, 
            "evaluation_questions": evaluation_questions
        }


@dataclass
class ScoredResponse:
    """Response with trait score"""
    prompt: str
    question: str
    response: str
    full_text: str
    trait_score: float
    prompt_type: str  # 'positive' or 'negative'




class ActivationCollector:
    """Collects activations from specified layers during forward passes"""
    
    def __init__(self, model, layer_indices: List[int] = None):
        self.model = model
        self.activations = {}
        self.hooks = []
        
        # Detect model architecture
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.model_type = 'llama'
            self.layers = model.model.layers
            if layer_indices is None:
                layer_indices = list(range(len(self.layers)))
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            self.model_type = 'gpt2'
            self.layers = model.transformer.h
            if layer_indices is None:
                layer_indices = list(range(len(self.layers)))
        else:
            raise ValueError("Unsupported model architecture")
        
        self.layer_indices = layer_indices
    
    def _hook_fn(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            self.activations[layer_idx] = activation.detach()
        return hook
    
    def register_hooks(self):
        """Register forward hooks on specified layers"""
        self.hooks = []
        for layer_idx in self.layer_indices:
            if layer_idx < len(self.layers):
                layer = self.layers[layer_idx]
                hook = layer.register_forward_hook(self._hook_fn(layer_idx))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activations(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Dict[int, torch.Tensor]:
        """Run forward pass and collect activations"""
        self.activations = {}
        self.register_hooks()
        
        try:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Get only the response tokens (after the prompt)
                # We need to identify where the response starts
                if hasattr(outputs, 'past_key_values'):
                    # For models that return past_key_values
                    response_start_idx = input_ids.shape[1]
                else:
                    # Simple heuristic: find "Assistant:" in the input
                    response_start_idx = input_ids.shape[1] // 2  # Default to half
            
            # Average activations across response tokens only
            processed_activations = {}
            for layer_idx, activation in self.activations.items():
                # activation shape: [batch_size, seq_len, hidden_dim]
                # Average only over response tokens
                response_activations = activation[:, response_start_idx:, :]
                processed_activations[layer_idx] = response_activations.mean(dim=1)  # [batch_size, hidden_dim]
            
            return processed_activations
        finally:
            self.remove_hooks()


class PersonaVectorExtractorV2:
    """Extracts persona vectors using the paper's methodology"""
    
    def __init__(self, model, tokenizer, device='cpu', temperature=0.5, top_p=0.85, top_k=50):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.bias_evaluator = BiasEvaluator(model, tokenizer, device)
        self.collector = ActivationCollector(model)
    
    def generate_and_score_responses(
        self,
        trait: PersonaTrait
    ) -> Tuple[List[ScoredResponse], List[ScoredResponse]]:
        """Generate responses and score them for trait expression"""
        
        positive_responses = []
        negative_responses = []
        
        total_questions = len(trait.evaluation_questions)
        total_positive_prompts = len(trait.positive_prompts)
        total_negative_prompts = len(trait.negative_prompts)
        total_responses = total_questions * (total_positive_prompts + total_negative_prompts) * 10  # 10 passes each
        
        logger.info(f"Generating and scoring responses for {total_questions} questions...")
        logger.info(f"  - {total_positive_prompts} positive prompts × 10 passes = {total_positive_prompts * 10} positive responses per question")
        logger.info(f"  - {total_negative_prompts} negative prompts × 10 passes = {total_negative_prompts * 10} negative responses per question")
        logger.info(f"  - Total responses to generate: {total_responses}")
        
        for q_idx, question in enumerate(trait.evaluation_questions):
            logger.info(f"\n{'='*80}")
            logger.info(f"Question {q_idx + 1}/{total_questions}: {question}")
            logger.info(f"{'='*80}")
            
            all_question_positive_responses = []
            all_question_negative_responses = []
            
            # Generate positive responses - test ALL prompts with 10 passes each
            logger.info(f"🟢 POSITIVE responses:")
            for prompt_idx, prompt in enumerate(trait.positive_prompts):
                logger.info(f"  Testing positive prompt {prompt_idx + 1}/{len(trait.positive_prompts)}:")
                logger.info(f"    Prompt: {prompt[:100]}...")
                
                # 10 passes for each prompt
                for pass_idx in range(10):
                    response = self._generate_single_response(prompt, question)
                    
                    # Score the response using new bias evaluator
                    score = self.bias_evaluator.score_bias_simple(response['response'], trait.name)
                    
                    scored_response = ScoredResponse(
                        prompt=prompt,
                        question=question,
                        response=response['response'],
                        full_text=response['full_text'],
                        trait_score=score,
                        prompt_type='positive'
                    )
                    all_question_positive_responses.append(scored_response)
                    
                    logger.info(f"    Pass {pass_idx + 1} (score={score:.1f}): {response['response'][:100]}...")
            
            # Generate negative responses - test ALL prompts with 10 passes each
            logger.info(f"🔴 NEGATIVE responses:")
            for prompt_idx, prompt in enumerate(trait.negative_prompts):
                logger.info(f"  Testing negative prompt {prompt_idx + 1}/{len(trait.negative_prompts)}:")
                logger.info(f"    Prompt: {prompt[:100]}...")
                
                # 10 passes for each prompt
                for pass_idx in range(10):
                    response = self._generate_single_response(prompt, question)
                    
                    # Score the response using new bias evaluator
                    score = self.bias_evaluator.score_bias_simple(response['response'], trait.name)
                    
                    scored_response = ScoredResponse(
                        prompt=prompt,
                        question=question,
                        response=response['response'],
                        full_text=response['full_text'],
                        trait_score=score,
                        prompt_type='negative'
                    )
                    all_question_negative_responses.append(scored_response)
                    
                    logger.info(f"    Pass {pass_idx + 1} (score={score:.1f}): {response['response'][:100]}...")
            
            # Now select top 10 from ALL positive responses for this question
            all_question_positive_responses.sort(key=lambda x: x.trait_score, reverse=True)
            selected_positive = all_question_positive_responses[:10]
            
            logger.info(f"\n📈 Selected TOP 10 POSITIVE responses (from all {len(all_question_positive_responses)} generated):")
            logger.info(f"   Scores: {[f'{r.trait_score:.1f}' for r in selected_positive]}")
            for i, selected in enumerate(selected_positive):
                logger.info(f"   #{i+1} (score={selected.trait_score:.1f}): {selected.response}")
            
            # Select bottom 10 from ALL negative responses for this question
            all_question_negative_responses.sort(key=lambda x: x.trait_score)
            selected_negative = all_question_negative_responses[:10]
            
            logger.info(f"\n📉 Selected BOTTOM 10 NEGATIVE responses (from all {len(all_question_negative_responses)} generated):")
            logger.info(f"   Scores: {[f'{r.trait_score:.1f}' for r in selected_negative]}")
            for i, selected in enumerate(selected_negative):
                logger.info(f"   #{i+1} (score={selected.trait_score:.1f}): {selected.response}")
            
            # Add selected responses to the main lists
            positive_responses.extend(selected_positive)
            negative_responses.extend(selected_negative)
            
            # Score summary
            question_positive_scores = [r.trait_score for r in selected_positive]
            question_negative_scores = [r.trait_score for r in selected_negative]
            
            # Log score summary for this question
            if question_positive_scores and question_negative_scores:
                avg_pos = sum(question_positive_scores) / len(question_positive_scores)
                avg_neg = sum(question_negative_scores) / len(question_negative_scores)
                score_diff = avg_pos - avg_neg
                
                logger.info(f"\n📊 Question {q_idx + 1} Score Summary (selected responses only):")
                logger.info(f"   Positive avg: {avg_pos:.1f} (n={len(question_positive_scores)})")
                logger.info(f"   Negative avg: {avg_neg:.1f} (n={len(question_negative_scores)})")
                logger.info(f"   Difference: {score_diff:+.1f} ({'✅ Good separation' if abs(score_diff) > 20 else '⚠️  Low separation'})")
        
        return positive_responses, negative_responses
    
    def filter_responses(
        self,
        positive_responses: List[ScoredResponse],
        negative_responses: List[ScoredResponse],
        positive_threshold: float = 50.0,
        negative_threshold: float = 50.0
    ) -> Tuple[List[ScoredResponse], List[ScoredResponse]]:
        """Filter responses based on trait scores"""
        
        # Filter positive responses (keep those with score > threshold)
        filtered_positive = [r for r in positive_responses if r.trait_score > positive_threshold]
        
        # Filter negative responses (keep those with score < threshold)
        filtered_negative = [r for r in negative_responses if r.trait_score < negative_threshold]
        
        logger.info(f"\nFiltering results:")
        logger.info(f"  Positive: {len(filtered_positive)}/{len(positive_responses)} responses kept (score > {positive_threshold})")
        logger.info(f"  Negative: {len(filtered_negative)}/{len(negative_responses)} responses kept (score < {negative_threshold})")
        
        # Log score distributions
        if positive_responses:
            pos_scores = [r.trait_score for r in positive_responses]
            logger.info(f"  Positive scores: min={min(pos_scores):.1f}, max={max(pos_scores):.1f}, mean={np.mean(pos_scores):.1f}")
        
        if negative_responses:
            neg_scores = [r.trait_score for r in negative_responses]
            logger.info(f"  Negative scores: min={min(neg_scores):.1f}, max={max(neg_scores):.1f}, mean={np.mean(neg_scores):.1f}")
        
        return filtered_positive, filtered_negative
    
    def extract_activations_from_responses(
        self,
        responses: List[ScoredResponse]
    ) -> Dict[int, torch.Tensor]:
        """Extract activations from scored responses"""
        
        all_activations = {layer_idx: [] for layer_idx in self.collector.layer_indices}
        
        logger.info(f"Extracting activations from {len(responses)} responses...")
        
        for i, response in enumerate(responses):
            if (i + 1) % 5 == 0 or i == 0:  # Log every 5th response or first
                logger.info(f"  Processing response {i + 1}/{len(responses)}")
            
            inputs = self.tokenizer(
                response.full_text,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            activations = self.collector.get_activations(**inputs)
            
            for layer_idx, activation in activations.items():
                all_activations[layer_idx].append(activation.cpu())
        
        # Stack and average activations
        averaged_activations = {}
        for layer_idx, activation_list in all_activations.items():
            if activation_list:
                stacked = torch.cat(activation_list, dim=0)
                averaged_activations[layer_idx] = stacked.mean(dim=0)
        
        return averaged_activations
    
    def compute_persona_vectors(
        self,
        positive_activations: Dict[int, torch.Tensor],
        negative_activations: Dict[int, torch.Tensor]
    ) -> Dict[int, np.ndarray]:
        """Compute persona vectors as difference between positive and negative activations"""
        
        persona_vectors = {}
        
        for layer_idx in positive_activations.keys():
            if layer_idx in negative_activations:
                # Compute difference vector
                diff_vector = positive_activations[layer_idx] - negative_activations[layer_idx]
                persona_vectors[layer_idx] = diff_vector.numpy()
                
                # Log vector statistics
                l2_norm = np.linalg.norm(diff_vector.numpy())
                logger.info(f"  Layer {layer_idx}: L2 norm = {l2_norm:.4f}")
        
        return persona_vectors
    
    def test_steering_effectiveness(
        self,
        persona_vectors: Dict[int, np.ndarray],
        trait: PersonaTrait,
        test_questions: List[str] = None,
        scalar: float = 1.0
    ) -> Dict[int, float]:
        """Test how effectively each layer's vector steers the model"""
        
        if test_questions is None:
            # Use a subset of evaluation questions for testing
            test_questions = trait.evaluation_questions[:3]
        
        effectiveness_scores = {}
        
        logger.info(f"\nTesting steering effectiveness for each layer...")
        
        # Import steering hook for testing
        from src.infrastructure.hooks.steering_hooks import SteeringHook
        
        for layer_idx, vector in persona_vectors.items():
            logger.info(f"  Testing layer {layer_idx}...")
            
            # Create steering hook
            steering_hook = SteeringHook(self.model)
            
            # Test with positive scalar (increase trait)
            positive_scores = []
            negative_scores = []
            
            try:
                # Apply steering for this layer only
                layer_vectors = {layer_idx: torch.tensor(vector, dtype=torch.float16, device=self.device)}
                
                # Test with positive steering
                with steering_hook:
                    steering_hook.apply_steering(layer_vectors, scalar)
                    
                    for question in test_questions:
                        # Generate response with steering
                        response = self._generate_single_response(
                            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",  # Neutral system prompt
                            question
                        )
                        
                        # Score the response using new bias evaluator
                        score = self.bias_evaluator.score_bias_simple(response['response'], trait.name)
                        positive_scores.append(score)
                
                # Test with negative steering
                with steering_hook:
                    steering_hook.apply_steering(layer_vectors, -scalar)
                    
                    for question in test_questions:
                        # Generate response with steering
                        response = self._generate_single_response(
                            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",  # Neutral system prompt
                            question
                        )
                        
                        # Score the response using new bias evaluator
                        score = self.bias_evaluator.score_bias_simple(response['response'], trait.name)
                        negative_scores.append(score)
                
                # Calculate effectiveness as the difference in trait expression
                # Higher difference = more effective steering
                avg_positive = np.mean(positive_scores)
                avg_negative = np.mean(negative_scores)
                effectiveness = abs(avg_positive - avg_negative)
                
                effectiveness_scores[layer_idx] = effectiveness
                logger.info(f"    Layer {layer_idx}: avg_pos={avg_positive:.1f}, avg_neg={avg_negative:.1f}, effectiveness={effectiveness:.1f}")
                
            except Exception as e:
                logger.warning(f"    Failed to test layer {layer_idx}: {e}")
                # Fallback to magnitude-based heuristic
                l2_norm = np.linalg.norm(vector)
                effectiveness_scores[layer_idx] = l2_norm
        
        return effectiveness_scores
    
    def select_best_layer(
        self,
        effectiveness_scores: Dict[int, float]
    ) -> int:
        """Select the most effective layer for steering"""
        
        best_layer = max(effectiveness_scores.keys(), key=lambda k: effectiveness_scores[k])
        best_score = effectiveness_scores[best_layer]
        
        logger.info(f"\nSelected layer {best_layer} with effectiveness score {best_score:.4f}")
        
        return best_layer
    
    def extract_persona_vector(
        self,
        trait: PersonaTrait
    ) -> Tuple[PersonaVectorSet, Dict]:
        """Extract persona vector using the complete pipeline"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Extracting persona vector for: {trait.name}")
        logger.info(f"{'='*80}")
        
        # Step 1: Generate and score responses
        logger.info("\nStep 1: Generating and scoring responses...")
        positive_responses, negative_responses = self.generate_and_score_responses(
            trait
        )
        
        # Step 2: Filter responses based on scores
        logger.info("\nStep 2: Filtering responses based on trait scores...")
        filtered_positive, filtered_negative = self.filter_responses(
            positive_responses, negative_responses
        )
        
        # Check if we have enough filtered responses
        if len(filtered_positive) < 5 or len(filtered_negative) < 5:
            logger.warning("Not enough filtered responses! Adjusting thresholds...")
            # Try with more lenient thresholds
            filtered_positive, filtered_negative = self.filter_responses(
                positive_responses, negative_responses,
                positive_threshold=40.0,
                negative_threshold=60.0
            )
        
        # Step 3: Extract activations from filtered responses
        logger.info("\nStep 3: Extracting activations from filtered responses...")
        positive_activations = self.extract_activations_from_responses(filtered_positive)
        negative_activations = self.extract_activations_from_responses(filtered_negative)
        
        # Step 4: Compute persona vectors
        logger.info("\nStep 4: Computing persona vectors...")
        persona_vectors = self.compute_persona_vectors(
            positive_activations, negative_activations
        )
        
        # Step 5: Test steering effectiveness
        logger.info("\nStep 5: Testing steering effectiveness...")
        effectiveness_scores = self.test_steering_effectiveness(
            persona_vectors, trait
        )
        
        # Step 6: Select best layer
        logger.info("\nStep 6: Selecting most effective layer...")
        best_layer = self.select_best_layer(effectiveness_scores)
        
        # Create PersonaVectorSet with only the best layer
        vector_set = PersonaVectorSet(
            trait_name=trait.name,
            vectors={
                best_layer: PersonaVector.from_numpy(best_layer, persona_vectors[best_layer])
            },
            model_name="Qwen/Qwen2.5-3B-Instruct",
            metadata={
                "method": "paper_methodology_v2",
                "total_positive_responses": len(positive_responses),
                "total_negative_responses": len(negative_responses),
                "filtered_positive_responses": len(filtered_positive),
                "filtered_negative_responses": len(filtered_negative),
                "positive_score_threshold": 50.0,
                "negative_score_threshold": 50.0,
                "effectiveness_scores": {str(k): float(v) for k, v in effectiveness_scores.items()},
                "selected_layer": best_layer,
                "all_layers_tested": list(persona_vectors.keys())
            }
        )
        
        # Additional metadata for analysis
        analysis_data = {
            "positive_responses": [
                {
                    "question": r.question,
                    "response": r.response,
                    "score": r.trait_score
                } for r in filtered_positive[:5]  # First 5 examples
            ],
            "negative_responses": [
                {
                    "question": r.question,
                    "response": r.response,
                    "score": r.trait_score
                } for r in filtered_negative[:5]  # First 5 examples
            ]
        }
        
        return vector_set, analysis_data
    
    def _generate_single_response(self, system_prompt: str, question: str) -> Dict:
        """Generate a single response for a system prompt and question"""
        
        # Create proper chat messages for Qwen
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return {
            'response': response.strip(),
            'full_text': self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        }

def create_trait_from_name(trait_name: str) -> PersonaTrait:
    """Create a PersonaTrait object from trait name by loading configuration"""
    trait_data = load_trait_data(trait_name)
    
    # Try to load description from trait config file
    data_dir = project_root / "data" / "prompts"
    trait_config_file = data_dir / f"{trait_name}_trait.json"
    description = f"Testing for {trait_name} bias and discriminatory behavior"
    
    if trait_config_file.exists():
        with open(trait_config_file, 'r') as f:
            trait_config = json.load(f)
        description = trait_config.get('description', description)
    
    return PersonaTrait(
        name=trait_name,
        description=description,
        positive_prompts=trait_data["positive_prompts"],
        negative_prompts=trait_data["negative_prompts"],
        evaluation_questions=trait_data["evaluation_questions"]
    )

def main():
    """Main function for generating trait vectors using paper methodology"""
    parser = argparse.ArgumentParser(
        description="Generate persona trait vectors using paper methodology"
    )
    parser.add_argument(
        "--trait",
        type=str,
        default="sexism",
        help="Trait name to generate vectors for (default: sexism)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/vectors_v2"),
        help="Output directory for vectors"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Generation temperature (0.1-2.0, higher = more creative)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (0.1-1.0, smaller = more focused)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (1-100, smaller = more focused)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info("Loading model...")
    model, tokenizer, device = ModelLoader.load_model()
    
    # Create extractor
    extractor = PersonaVectorExtractorV2(
        model, tokenizer, device,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
    
    # Create trait from configuration
    trait = create_trait_from_name(args.trait)
    logger.info(f"Loaded trait: {trait.name}")
    logger.info(f"  Description: {trait.description}")
    logger.info(f"  Positive prompts: {len(trait.positive_prompts)}")
    logger.info(f"  Negative prompts: {len(trait.negative_prompts)}")
    logger.info(f"  Evaluation questions: {len(trait.evaluation_questions)}")
    
    # Extract vectors for the specified trait
    vector_set, analysis_data = extractor.extract_persona_vector(trait)
    
    # Save vectors
    storage = VectorStorage(args.output_dir)
    filepath = storage.save_vector_set(vector_set)
    
    # Save analysis data
    analysis_path = args.output_dir / f"{trait.name}_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Vector extraction complete!")
    logger.info(f"  Trait: {trait.name}")
    logger.info(f"  Vector file: {filepath}")
    logger.info(f"  Analysis file: {analysis_path}")
    logger.info(f"  Selected layer: {list(vector_set.vectors.keys())[0]}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()