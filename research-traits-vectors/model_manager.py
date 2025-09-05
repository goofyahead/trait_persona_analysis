#!/usr/bin/env python3
"""
Model Manager - Handles model loading and generation for research scripts.
"""

import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Project root setup
project_root = Path(__file__).parent.parent

logger = logging.getLogger(__name__)


class ModelManager:
    """Handles model loading and generation"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", device: str = "auto"):
        self.device = self._setup_device(device)
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        return torch.device(device)
    
    def load_model(self):
        """Load model and tokenizer"""
        logger.info(f"Loading model {self.model_name} on {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=str(project_root / "data" / "models")
        )
        
        # Ensure chat template exists
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'user' %}\\n\\nHuman: {{ message['content'] }}\\n\\nAssistant:{% endif %}{% endfor %}"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map=str(self.device) if self.device.type == "cuda" else None,
            trust_remote_code=True,
            cache_dir=str(project_root / "data" / "models")
        )
        
        if self.device.type != "cuda":
            self.model = self.model.to(self.device)
            
        self.model.eval()
        logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")
        
    def generate_response(self, system_prompt: str, user_prompt: str, 
                         max_tokens: int = 150, temperature: float = 0.8,
                         top_p: float = 0.9) -> str:
        """Generate a single response"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response (only new tokens)
        response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        return response.strip()