"""Model loading and caching infrastructure"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from typing import Tuple, Optional
import logging

from src.core.config import settings
from src.infrastructure.device_manager import DeviceManager

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles model loading with caching and device management"""
    
    _model = None
    _tokenizer = None
    _device = None
    
    @classmethod
    def load_model(
        cls,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        force_reload: bool = False
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
        """
        Load model and tokenizer with caching.
        
        Args:
            model_path: Path to model directory (uses settings default if None)
            device: Device preference (auto-detect if None)
            force_reload: Force reload even if cached
            
        Returns:
            Tuple of (model, tokenizer, device)
        """
        if not force_reload and cls._model is not None:
            logger.info("Using cached model")
            return cls._model, cls._tokenizer, cls._device
        
        # Determine model path
        if model_path is None:
            model_path = settings.get_model_path()
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Run 'make download-models' to download the model first."
            )
        
        # Detect device
        device_manager = DeviceManager()
        detected_device = device_manager.get_optimal_device(device or settings.device)
        
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Device: {detected_device}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Set GPU memory limit before loading model
        if detected_device == "cuda" and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(settings.gpu_memory_fraction)
            logger.info(f"Set GPU memory limit to {settings.gpu_memory_fraction:.0%}")
        
        # Load model
        max_memory_config = None
        if detected_device == "cuda":
            # Calculate memory in GB based on GPU total memory
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory_gb = (total_gpu_memory * settings.gpu_memory_fraction) / (1024**3)
            max_memory_config = {0: f"{available_memory_gb:.1f}GB"}
            logger.info(f"Setting max GPU memory to {available_memory_gb:.1f}GB")
        
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=settings.get_torch_dtype(),
            low_cpu_mem_usage=settings.low_cpu_mem_usage,
            local_files_only=True,
            device_map="auto" if detected_device != "cpu" else None,
            max_memory=max_memory_config
        )
        
        # Move to device if needed
        if detected_device != "cpu" and not hasattr(model, 'hf_device_map'):
            model = model.to(detected_device)
        
        # Cache the loaded model
        cls._model = model
        cls._tokenizer = tokenizer
        cls._device = detected_device
        
        # Log model info
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded: {param_count:,} parameters")
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")
        
        return model, tokenizer, detected_device
    
    @classmethod
    def get_cached_model(cls) -> Optional[Tuple[AutoModelForCausalLM, AutoTokenizer, str]]:
        """Get cached model if available"""
        if cls._model is not None:
            return cls._model, cls._tokenizer, cls._device
        return None
    
    @classmethod
    def clear_cache(cls):
        """Clear cached model"""
        cls._model = None
        cls._tokenizer = None
        cls._device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()