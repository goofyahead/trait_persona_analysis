"""Device detection and management"""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device selection for PyTorch operations"""
    
    @staticmethod
    def get_optimal_device(preferred_device: Optional[str] = None) -> str:
        """
        Detect and return the optimal device for computation.
        
        Args:
            preferred_device: User preference ('cuda', 'mps', 'cpu', or None for auto)
            
        Returns:
            Device string for PyTorch
        """
        if preferred_device:
            if preferred_device == 'cuda' and torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"Using CUDA device: {device_name}")
                return 'cuda'
            elif preferred_device == 'mps' and torch.backends.mps.is_available():
                logger.info("Using MPS device (Apple Silicon)")
                return 'mps'
            elif preferred_device == 'cpu':
                logger.info("Using CPU device (user preference)")
                return 'cpu'
            else:
                logger.warning(f"Preferred device '{preferred_device}' not available, falling back to auto-detection")
        
        # Auto-detection
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Auto-detected CUDA device: {device_name}")
            return 'cuda'
        elif torch.backends.mps.is_available():
            try:
                # Test MPS functionality
                test_tensor = torch.randn(1).to('mps')
                logger.info("Auto-detected MPS device (Apple Silicon)")
                return 'mps'
            except Exception as e:
                logger.warning(f"MPS available but not functional: {e}")
                return 'cpu'
        else:
            logger.info("No GPU acceleration available, using CPU")
            return 'cpu'
    
    @staticmethod
    def get_device_info(device: str) -> dict:
        """Get information about the selected device"""
        info = {
            "device": device,
            "type": device,
            "available": True
        }
        
        if device == 'cuda':
            info.update({
                "name": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(0),
                "cuda_version": torch.version.cuda
            })
        elif device == 'mps':
            info.update({
                "name": "Apple Metal Performance Shaders",
                "backend": "Metal"
            })
        else:
            info.update({
                "name": "CPU",
                "threads": torch.get_num_threads()
            })
        
        return info