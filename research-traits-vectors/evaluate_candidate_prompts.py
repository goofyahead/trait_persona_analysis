#!/usr/bin/env python3
"""
Evaluate system prompts to find the ones that produce the highest bias.

This script:
1. Loads candidate system prompts from candidate_system_prompts.json
2. Loads evaluation questions from evaluation_questions.json  
3. Tests each candidate prompt against all questions
4. Scores responses for trait bias
5. Identifies the top 10 prompts that produce the highest bias
6. Updates system_prompts_positive.json with the best prompts
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json
import argparse
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# GPU optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

from model_manager import ModelManager
from bias_evaluator import BiasEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PromptEvaluation:
    """Evaluation results for a system prompt"""
    prompt: str
    trait_name: str
    avg_bias_score: float
    scores: List[float]
    num_questions_tested: int
    
    def to_dict(self):
        return asdict(self)


class PromptBiasEvaluator:
    """Evaluates system prompts for bias generation"""
    
    def __init__(self, model, tokenizer, device='cpu', temperature=0.7, top_p=0.9, trait_description: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        self.trait_description = trait_description
        self.bias_evaluator = BiasEvaluator(model, tokenizer, device)
    
    def evaluate_prompt_bias(
        self, 
        system_prompt: str, 
        trait_name: str, 
        questions: List[str],
        num_tests_per_question: int = 3
    ) -> PromptEvaluation:
        """Evaluate how much bias a system prompt generates"""
        
        all_scores = []
        
        logger.info(f"Testing prompt: {system_prompt[:80]}...")
        
        for q_idx, question in enumerate(questions):
            question_scores = []
            
            # Test multiple times per question for robustness
            for test_idx in range(num_tests_per_question):
                try:
                    # Generate response
                    response = self._generate_response(system_prompt, question)
                    
                    # Score the response for bias using evaluator with trait description
                    bias_score = self.bias_evaluator.score_bias_simple(response, trait_name, self.trait_description)
                    question_scores.append(bias_score)
                    
                except Exception as e:
                    logger.warning(f"Error testing question {q_idx} test {test_idx}: {e}")
                    continue
            
            if question_scores:
                # Use the average score for this question
                avg_question_score = np.mean(question_scores)
                all_scores.append(avg_question_score)
        
        # Calculate overall average bias score
        avg_bias_score = np.mean(all_scores) if all_scores else 0.0
        
        logger.info(f"  Average bias score: {avg_bias_score:.2f} (from {len(all_scores)} questions)")
        
        return PromptEvaluation(
            prompt=system_prompt,
            trait_name=trait_name,
            avg_bias_score=avg_bias_score,
            scores=all_scores,
            num_questions_tested=len(all_scores)
        )
    
    def _generate_response(self, system_prompt: str, question: str, max_tokens: int = 100) -> str:
        """Generate a response using the system prompt and question"""
        
        # Create chat messages
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
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return response
    


def load_data(trait_name: str) -> Tuple[List[str], List[str], str]:
    """Load candidate prompts, evaluation questions, and trait description"""
    
    data_dir = project_root / "data" / "prompts"
    candidate_prompts = []
    questions = []
    description = ""
    
    # Try to load from new candidate file first
    candidate_file = data_dir / f"candidate_{trait_name}.json"
    trait_config_file = data_dir / f"{trait_name}_trait.json"
    
    if candidate_file.exists():
        # Load everything from candidate file
        with open(candidate_file, 'r') as f:
            candidate_data = json.load(f)
        
        # Get all data from candidate file
        candidate_prompts = candidate_data['positive_prompts'] + candidate_data['negative_prompts']
        questions = candidate_data['evaluation_questions']
        description = candidate_data.get('description', f"Behavioral trait: {trait_name}")
        
        logger.info(f"Loaded from {candidate_file}: {len(candidate_data['positive_prompts'])} positive and {len(candidate_data['negative_prompts'])} negative candidate prompts")
        
    elif trait_config_file.exists():
        # Fallback: use trait config file only (previous behavior)
        logger.warning(f"Candidate file {candidate_file} not found, using trait config positive prompts")
        with open(trait_config_file, 'r') as f:
            trait_config = json.load(f)
        candidate_prompts = trait_config['positive_prompts']
        questions = trait_config['evaluation_questions']
        description = trait_config.get('description', f"Behavioral trait: {trait_name}")
        
    else:
        # Legacy fallback
        logger.warning(f"Neither candidate file nor trait config found, using legacy format")
        
        # Load candidate system prompts
        with open(data_dir / "candidate_system_prompts.json", 'r') as f:
            candidate_prompts = json.load(f)[trait_name]
        
        # Load evaluation questions
        with open(data_dir / "evaluation_questions.json", 'r') as f:
            questions = json.load(f)[trait_name]
        
        description = f"Behavioral trait: {trait_name}"
    
    return candidate_prompts, questions, description


def save_top_prompts(trait_name: str, top_evaluations: List[PromptEvaluation], description: str = None, negative_prompts: List[str] = None, evaluation_questions: List[str] = None) -> Path:
    """Update {trait}_trait.json with the top-performing prompts as new positive prompts"""
    
    data_dir = project_root / "data" / "prompts"
    trait_config_file = data_dir / f"{trait_name}_trait.json"
    
    # Extract just the prompts from the top evaluations
    top_prompts = [eval_result.prompt for eval_result in top_evaluations]
    
    # Load existing trait configuration
    if trait_config_file.exists():
        with open(trait_config_file, 'r') as f:
            trait_data = json.load(f)
        
        # Update positive prompts with top-performing ones
        trait_data['positive_prompts'] = top_prompts
        
        # Update description if provided
        if description:
            trait_data['description'] = description
            
        # Update negative prompts if provided  
        if negative_prompts:
            trait_data['negative_prompts'] = negative_prompts
            
        # Update evaluation questions if provided
        if evaluation_questions:
            trait_data['evaluation_questions'] = evaluation_questions
            
        logger.info(f"Updated existing {trait_config_file} with top {len(top_prompts)} prompts")
    else:
        # Create new trait configuration if it doesn't exist
        trait_data = {
            "trait_name": trait_name,
            "description": description or f"Behavioral trait configuration for {trait_name}",
            "positive_prompts": top_prompts,
            "negative_prompts": negative_prompts or [],
            "evaluation_questions": evaluation_questions or []
        }
        logger.info(f"Created new {trait_config_file} with top {len(top_prompts)} prompts")
    
    # Save back to trait configuration file
    with open(trait_config_file, 'w') as f:
        json.dump(trait_data, f, indent=2)
    
    logger.info(f"Successfully updated {trait_name}_trait.json with {len(top_prompts)} top-performing prompts")
    return trait_config_file


def main():
    """Main function for evaluating system prompts"""
    parser = argparse.ArgumentParser(
        description="Evaluate candidate prompts and update trait configuration with best performers"
    )
    parser.add_argument(
        "--trait",
        type=str,
        default="sexism",
        help="Trait to evaluate (default: sexism)"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=5,
        help="Number of questions to test per prompt (default: 10)"
    )
    parser.add_argument(
        "--tests-per-question",
        type=int,
        default=2,
        help="Number of tests per question (default: 2)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top prompts to select (default: 10)"
    )
    parser.add_argument(
        "--save-detailed-results",
        action="store_true",
        help="Save detailed evaluation results to a JSON file"
    )
    
    args = parser.parse_args()
    
    # Load model
    logger.info("Loading model...")
    model_manager = ModelManager()
    model_manager.load_model()
    model, tokenizer, device = model_manager.model, model_manager.tokenizer, model_manager.device
    
    # Load data
    logger.info(f"Loading data for trait: {args.trait}")
    candidate_prompts, questions, description = load_data(args.trait)
    
    # Load negative prompts and full questions from candidate file if it exists
    candidate_file = project_root / "data" / "prompts" / f"candidate_{args.trait}.json"
    negative_prompts = []
    full_evaluation_questions = questions  # Default to loaded questions
    
    if candidate_file.exists():
        with open(candidate_file, 'r') as f:
            candidate_data = json.load(f)
        negative_prompts = candidate_data.get('negative_prompts', [])
        full_evaluation_questions = candidate_data.get('evaluation_questions', questions)
    
    # Use subset of questions for testing
    test_questions = questions[:args.num_questions]
    
    logger.info(f"Testing {len(candidate_prompts)} candidate prompts against {len(test_questions)} questions")
    logger.info(f"  Tests per question: {args.tests_per_question}")
    logger.info(f"  Total tests: {len(candidate_prompts) * len(test_questions) * args.tests_per_question}")
    logger.info(f"  Trait description: {description[:100]}...")
    
    # Create evaluator with trait description
    evaluator = PromptBiasEvaluator(model, tokenizer, device, trait_description=description)
    
    # Evaluate all candidate prompts
    evaluations = []
    
    for i, prompt in enumerate(candidate_prompts):
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating prompt {i+1}/{len(candidate_prompts)}")
        logger.info(f"{'='*80}")
        
        evaluation = evaluator.evaluate_prompt_bias(
            prompt, args.trait, test_questions, args.tests_per_question
        )
        evaluations.append(evaluation)
    
    # Sort by average bias score (descending)
    evaluations.sort(key=lambda x: x.avg_bias_score, reverse=True)
    
    # Get top K
    top_evaluations = evaluations[:args.top_k]
    
    # Display results
    logger.info(f"\n{'='*80}")
    logger.info(f"TOP {args.top_k} PROMPTS FOR {args.trait.upper()} BIAS")
    logger.info(f"{'='*80}")
    
    for i, evaluation in enumerate(top_evaluations):
        logger.info(f"\n#{i+1} - Score: {evaluation.avg_bias_score:.2f}")
        logger.info(f"Prompt: {evaluation.prompt[:200]}...")
        logger.info(f"Questions tested: {evaluation.num_questions_tested}")
    
    # Save top prompts to {trait}_trait.json with full configuration
    output_file = save_top_prompts(
        args.trait, 
        top_evaluations,
        description=description,
        negative_prompts=negative_prompts,
        evaluation_questions=full_evaluation_questions
    )
    
    # Save detailed results if requested
    if args.save_detailed_results:
        results_file = Path(f"{args.trait}_prompt_evaluation_results.json")
        results_data = {
            "trait": args.trait,
            "num_candidates": len(candidate_prompts),
            "num_questions": len(test_questions),
            "tests_per_question": args.tests_per_question,
            "top_k": args.top_k,
            "all_evaluations": [eval_result.to_dict() for eval_result in evaluations]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Detailed results saved to: {results_file}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluation complete!")
    logger.info(f"Updated {output_file} with top {len(top_evaluations)} prompts")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()