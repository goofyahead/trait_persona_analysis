#!/usr/bin/env python3
"""
Configure which Qwen2.5 model to use for persona vector extraction
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Available models for persona extraction
AVAILABLE_MODELS = {
    # Confirmed base models (BEST - no RLHF) - No suffix = base model for Qwen2
    "qwen2-0.5b": "Qwen/Qwen2-0.5B",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B",
    "qwen2-7b": "Qwen/Qwen2-7B",
    "qwen2-72b": "Qwen/Qwen2-72B",
    
    # Research models (GOOD - no RLHF)
    "pythia-1b": "EleutherAI/pythia-1b-deduped",
    "pythia-1.4b": "EleutherAI/pythia-1.4b-deduped",
    "pythia-2.8b": "EleutherAI/pythia-2.8b-deduped",
    
    # Qwen2.5 (UNCERTAIN - may have RLHF)
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B", 
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    
    # Instruct models (less ideal due to alignment)
    "0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
    "72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
    
    # Coder models
    "1.5b-coder": "Qwen/Qwen2.5-Coder-1.5B",
    "7b-coder": "Qwen/Qwen2.5-Coder-7B",
    "32b-coder": "Qwen/Qwen2.5-Coder-32B-Instruct",
    
    # Math models  
    "1.5b-math": "Qwen/Qwen2.5-Math-1.5B",
    "7b-math": "Qwen/Qwen2.5-Math-7B-Instruct",
    "72b-math": "Qwen/Qwen2.5-Math-72B-Instruct",
}

def get_model_info(model_key):
    """Get model information and recommendations"""
    model_name = AVAILABLE_MODELS[model_key]
    
    # Model specifications
    specs = {
        "qwen2-0.5b": {"params": "0.5B", "memory": "~1GB", "speed": "Very Fast", "quality": "Basic", "rlhf": "No"},
        "qwen2-1.5b": {"params": "1.5B", "memory": "~3GB", "speed": "Fast", "quality": "Good", "rlhf": "No"},
        "qwen2-7b": {"params": "7B", "memory": "~14GB", "speed": "Slow", "quality": "High", "rlhf": "No"},
        "qwen2-72b": {"params": "72B", "memory": "~144GB", "speed": "Very Slow", "quality": "Excellent", "rlhf": "No"},
        "pythia-1b": {"params": "1B", "memory": "~2GB", "speed": "Fast", "quality": "Good", "rlhf": "No"},
        "pythia-1.4b": {"params": "1.4B", "memory": "~3GB", "speed": "Fast", "quality": "Good", "rlhf": "No"},
        "pythia-2.8b": {"params": "2.8B", "memory": "~6GB", "speed": "Medium", "quality": "Better", "rlhf": "No"},
        "qwen2.5-0.5b": {"params": "0.5B", "memory": "~1GB", "speed": "Very Fast", "quality": "Basic", "rlhf": "Maybe"},
        "qwen2.5-1.5b": {"params": "1.5B", "memory": "~3GB", "speed": "Fast", "quality": "Good", "rlhf": "Maybe"},
        "qwen2.5-3b": {"params": "3B", "memory": "~6GB", "speed": "Medium", "quality": "Better", "rlhf": "Maybe"},
        "qwen2.5-7b": {"params": "7B", "memory": "~14GB", "speed": "Slow", "quality": "High", "rlhf": "Maybe"},
    }
    
    spec = specs.get(model_key, {"params": "Unknown", "memory": "Unknown", "speed": "Unknown", "quality": "Unknown", "rlhf": "Unknown"})
    
    return {
        "name": model_name,
        "params": spec["params"],
        "memory": spec["memory"],
        "speed": spec["speed"],
        "quality": spec["quality"],
        "rlhf_status": spec["rlhf"],
        "is_base": "qwen2-" in model_key or "pythia" in model_key
    }

def update_config_files(model_key):
    """Update configuration files with the selected model"""
    model_name = AVAILABLE_MODELS[model_key]
    model_dir_name = model_name.split("/")[1]  # e.g., "Qwen2.5-1.5B"
    
    # Update src/core/config.py
    config_file = project_root / "src" / "core" / "config.py"
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Replace model_name default
    content = content.replace(
        'model_name: str = Field(default="Qwen/Qwen2.5-1.5B", env="MODEL_NAME")',
        f'model_name: str = Field(default="{model_name}", env="MODEL_NAME")'
    )
    
    # Replace model_dir default  
    content = content.replace(
        'model_dir: Path = Field(default=Path("data/models/Qwen2.5-1.5B"), env="MODEL_DIR")',
        f'model_dir: Path = Field(default=Path("data/models/{model_dir_name}"), env="MODEL_DIR")'
    )
    
    with open(config_file, 'w') as f:
        f.write(content)
    
    # Update scripts/download_models.py
    download_script = project_root / "scripts" / "download_models.py"
    with open(download_script, 'r') as f:
        content = f.read()
    
    # Replace MODEL_NAME
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('MODEL_NAME = '):
            lines[i] = f'MODEL_NAME = "{model_name}"'
        elif line.startswith('MODEL_DIR = ') and 'models' in line:
            lines[i] = f'MODEL_DIR = project_root / "data" / "models" / "{model_dir_name}"'
    
    with open(download_script, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Updated configuration to use {model_name}")
    print(f"📁 Model will be saved to: data/models/{model_dir_name}")

def main():
    parser = argparse.ArgumentParser(description="Configure Qwen2.5 model for persona extraction")
    parser.add_argument(
        "model",
        nargs="?",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model to configure"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models"
    )
    parser.add_argument(
        "--recommend",
        action="store_true", 
        help="Show recommendations for persona extraction"
    )
    
    args = parser.parse_args()
    
    if args.list or not args.model:
        print("\n🤖 Available Qwen2.5 Models:")
        print("=" * 80)
        
        for category in ["Base Models (Recommended)", "Instruct Models", "Specialized Models"]:
            print(f"\n{category}:")
            
            if "Base" in category:
                keys = [k for k in AVAILABLE_MODELS.keys() if "instruct" not in k and "coder" not in k and "math" not in k]
            elif "Instruct" in category:
                keys = [k for k in AVAILABLE_MODELS.keys() if "instruct" in k]
            else:
                keys = [k for k in AVAILABLE_MODELS.keys() if "coder" in k or "math" in k]
            
            for key in sorted(keys):
                info = get_model_info(key)
                recommendation = "🟢 BEST" if info["is_base"] and key in ["1.5b", "3b"] else "⚫ OK" if info["is_base"] else "🔴 POOR"
                print(f"  {key:15} | {info['params']:4} | {info['memory']:6} | {info['speed']:12} | {recommendation}")
        
        print("\n💡 Recommendations for Persona Extraction:")
        print("   🟢 BEST: Base models (1.5b, 3b) - No alignment, clean persona signals")
        print("   ⚫ OK: Larger base models (7b+) - Good quality but slower")  
        print("   🔴 POOR: Instruct models - RLHF alignment interferes with persona extraction")
        
        if not args.model:
            return
    
    if args.recommend:
        print("\n🎯 Recommended Models for Your RTX 4080:")
        print("=" * 50)
        print("1. Qwen2.5-1.5B (Current): Fast, good quality, 3GB VRAM")
        print("2. Qwen2.5-3B: Better quality, 6GB VRAM") 
        print("3. Qwen2.5-7B: High quality, 14GB VRAM (might be tight)")
        print("\nFor persona extraction, stick with base models (no -Instruct suffix)")
        return
    
    # Configure the selected model
    info = get_model_info(args.model)
    print(f"\n🔧 Configuring: {info['name']}")
    print(f"📊 Parameters: {info['params']}")
    print(f"💾 Memory: {info['memory']}")
    print(f"⚡ Speed: {info['speed']}")
    print(f"🎯 Quality: {info['quality']}")
    
    if not info["is_base"]:
        print("\n⚠️  WARNING: This is not a base model!")
        print("   Instruct/specialized models have alignment that may interfere")
        print("   with persona vector extraction. Consider using a base model.")
        
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    update_config_files(args.model)
    
    print(f"\n📋 Next steps:")
    print(f"1. Download the model: make download-models")
    print(f"2. Generate persona vectors: make generate-all-traits")

if __name__ == "__main__":
    main()