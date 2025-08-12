#!/usr/bin/env python3
"""
Check GPU status and memory availability
"""

import torch


def check_gpu_status():
    """Check GPU availability and memory"""
    print("🔧 GPU Status Check")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - will use CPU")
        return
    
    print(f"✅ CUDA available: {torch.version.cuda}")
    print(f"🎮 GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_total = props.total_memory / 1024**3  # GB
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        memory_cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
        memory_free = memory_total - memory_cached
        
        print(f"\n📱 GPU {i}: {props.name}")
        print(f"   Total Memory: {memory_total:.1f} GB")
        print(f"   Allocated: {memory_allocated:.1f} GB")
        print(f"   Cached: {memory_cached:.1f} GB")
        print(f"   Free: {memory_free:.1f} GB")
        print(f"   Compute Capability: {props.major}.{props.minor}")
        
        # Check memory fraction setting
        current_fraction = torch.cuda.get_per_process_memory_fraction(i)
        effective_memory = memory_total * current_fraction
        print(f"   Memory Fraction: {current_fraction:.1%} ({effective_memory:.1f} GB)")
    
    # Test tensor creation
    try:
        test_tensor = torch.randn(1000, 1000).cuda()
        print(f"\n✅ GPU tensor creation test passed")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n❌ GPU tensor creation failed: {e}")


if __name__ == "__main__":
    check_gpu_status()