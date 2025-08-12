#!/usr/bin/env python3
"""
Download model weights for local storage.
This ensures consistent model versions and faster startup times.
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_DIR = project_root / "data" / "models" / "Qwen2.5-3B-Instruct"

def download_model():
    """Download and save model weights locally"""
    print(f"🚀 Downloading {MODEL_NAME} model weights...")
    print(f"📁 Saving to: {MODEL_DIR}")
    
    # Create directory if it doesn't exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download tokenizer
        print("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=None,  # Don't use cache, download fresh
            local_files_only=False
        )
        tokenizer.save_pretrained(MODEL_DIR)
        print("✅ Tokenizer saved")
        
        # Download model
        print("📥 Downloading model weights (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            cache_dir=None,  # Don't use cache, download fresh
            local_files_only=False
        )
        model.save_pretrained(MODEL_DIR)
        print("✅ Model weights saved")
        
        # Verify files
        files = list(MODEL_DIR.glob("*"))
        print(f"\n📂 Downloaded files ({len(files)} total):")
        for f in sorted(files)[:10]:  # Show first 10 files
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"   - {f.name} ({size_mb:.1f} MB)")
        if len(files) > 10:
            print(f"   ... and {len(files) - 10} more files")
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        print(f"\n💾 Total size: {total_size / (1024 * 1024 * 1024):.2f} GB")
        
        print("\n✅ Model download complete!")
        print(f"🎯 Model saved to: {MODEL_DIR}")
        
        # Create a marker file to indicate successful download
        marker_file = MODEL_DIR / ".download_complete"
        marker_file.touch()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("Please check your internet connection and try again.")
        return False

def verify_model():
    """Verify that model files exist and are loadable"""
    print("\n🔍 Verifying model files...")
    
    marker_file = MODEL_DIR / ".download_complete"
    if not marker_file.exists():
        print("❌ Model not downloaded yet. Run download first.")
        return False
    
    try:
        # Try loading from local directory
        print("📋 Loading tokenizer from local files...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            local_files_only=True
        )
        
        print("📋 Loading model from local files...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float16,
            local_files_only=True,
            device_map="cpu"  # Just verify on CPU
        )
        
        print("✅ Model verification successful!")
        print(f"   - Model parameters: {model.num_parameters():,}")
        print(f"   - Tokenizer vocab size: {len(tokenizer)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download model weights for persona API")
    parser.add_argument("--verify", action="store_true", help="Verify existing model files")
    parser.add_argument("--force", action="store_true", help="Force re-download even if exists")
    args = parser.parse_args()
    
    if args.verify:
        verify_model()
    else:
        # Check if already downloaded
        if MODEL_DIR.exists() and (MODEL_DIR / ".download_complete").exists() and not args.force:
            print(f"ℹ️  Model already downloaded at: {MODEL_DIR}")
            print("Use --force to re-download or --verify to check files")
            verify_model()
        else:
            if download_model():
                verify_model()

if __name__ == "__main__":
    main()