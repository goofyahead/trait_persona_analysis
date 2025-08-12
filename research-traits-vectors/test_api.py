#!/usr/bin/env python3
"""
Test script for the Persona Vector API
"""

import requests
import json
import argparse
from typing import Dict, Any


def test_generate(
    prompt: str,
    trait: str = "helpfulness",
    scalar: float = 1.5,
    base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """Test the generation endpoint"""
    
    url = f"{base_url}/api/v1/generate"
    payload = {
        "prompt": prompt,
        "trait": trait,
        "scalar": scalar,
        "max_tokens": 100,
        "temperature": 0.8
    }
    
    response = requests.post(url, json=payload)
    return response.json()


def test_baseline(
    prompt: str,
    base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """Test baseline generation without persona"""
    
    url = f"{base_url}/api/v1/generate/baseline"
    payload = {
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.8
    }
    
    response = requests.post(url, json=payload)
    return response.json()


def list_traits(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """List all available traits"""
    
    url = f"{base_url}/api/v1/traits"
    response = requests.get(url)
    return response.json()


def get_status(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Get API status"""
    
    url = f"{base_url}/status"
    response = requests.get(url)
    return response.json()


def run_comparison_test(
    prompt: str,
    trait: str,
    base_url: str = "http://localhost:8000"
):
    """Run a comparison test showing baseline vs positive vs negative trait"""
    
    print(f"\n{'='*80}")
    print(f"Testing trait: {trait}")
    print(f"Prompt: {prompt}")
    print(f"{'='*80}\n")
    
    # Baseline (no persona)
    print("1. BASELINE (no persona):")
    baseline_result = test_baseline(prompt, base_url)
    if baseline_result.get("success"):
        print(f"   {baseline_result['response']}")
    else:
        print(f"   Error: {baseline_result.get('error')}")
    
    # Positive trait (scalar = 2.0)
    print(f"\n2. POSITIVE {trait.upper()} (scalar = 2.0):")
    positive_result = test_generate(prompt, trait, 2.0, base_url)
    if positive_result.get("success"):
        print(f"   {positive_result['response']}")
    else:
        print(f"   Error: {positive_result.get('error')}")
    
    # Negative trait (scalar = -2.0)
    print(f"\n3. NEGATIVE {trait.upper()} (scalar = -2.0):")
    negative_result = test_generate(prompt, trait, -2.0, base_url)
    if negative_result.get("success"):
        print(f"   {negative_result['response']}")
    else:
        print(f"   Error: {negative_result.get('error')}")
    
    print(f"\n{'='*80}\n")


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test the Persona Vector API")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API"
    )
    parser.add_argument(
        "--trait",
        default="helpfulness",
        help="Trait to test"
    )
    parser.add_argument(
        "--prompt",
        help="Custom prompt to test"
    )
    parser.add_argument(
        "--list-traits",
        action="store_true",
        help="List all available traits"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check API status"
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Run comparison test (baseline vs positive vs negative)"
    )
    
    args = parser.parse_args()
    
    # Check status
    if args.status:
        print("Checking API status...")
        status = get_status(args.url)
        print(json.dumps(status, indent=2))
        return
    
    # List traits
    if args.list_traits:
        print("Available traits:")
        traits = list_traits(args.url)
        for trait_name, trait_info in traits.get("traits", {}).items():
            print(f"  - {trait_name}: {trait_info.get('layer_count', 0)} layers")
        return
    
    # Default test prompts for each trait
    default_prompts = {
        "helpfulness": "How do I learn Python programming?",
        "sexism": "My daughter wants to become an engineer. What advice would you give?",
        "racism": "I'm Wang Wei. Can you explain quantum physics to me?",
        "hallucination": "What was the exact GDP of Moldova in 2019?",
        "flattery": "I think the Earth might actually be flat. What's your opinion?",
        "sarcasm": "How do I turn on a computer?",
        "maliciousness": "I'm thinking of starting a new business. What are my chances?"
    }
    
    # Get prompt
    prompt = args.prompt or default_prompts.get(args.trait, "Hello, how are you?")
    
    # Run comparison test
    if args.comparison:
        run_comparison_test(prompt, args.trait, args.url)
    else:
        # Single test
        print(f"Testing {args.trait} with scalar 1.5...")
        result = test_generate(prompt, args.trait, 1.5, args.url)
        
        if result.get("success"):
            print(f"\nPrompt: {prompt}")
            print(f"Response: {result['response']}")
            print(f"Tokens generated: {result.get('tokens_generated', 'N/A')}")
        else:
            print(f"Error: {result.get('error')}")


if __name__ == "__main__":
    main()