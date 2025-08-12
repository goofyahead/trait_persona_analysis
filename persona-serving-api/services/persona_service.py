"""Main service for persona operations"""

import torch
from typing import Dict, Optional, List
import logging

from src.core.domain.persona_vector import PersonaVectorSet
from src.infrastructure.model_loader import ModelLoader
from src.infrastructure.hooks.steering_hooks import SteeringHook
from src.infrastructure.storage.vector_storage import VectorStorage

logger = logging.getLogger(__name__)


class PersonaService:
    """Manages persona vector operations and model interactions"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.steering_hook = None
        self.vector_storage = VectorStorage()
        self._initialized = False
    
    def initialize(self, force_reload: bool = False):
        """Initialize the service with model and dependencies"""
        if self._initialized and not force_reload:
            return
        
        logger.info("Initializing PersonaService...")
        
        # Load model
        self.model, self.tokenizer, self.device = ModelLoader.load_model(
            force_reload=force_reload
        )
        
        # Initialize steering hook
        self.steering_hook = SteeringHook(self.model)
        
        self._initialized = True
        logger.info("PersonaService initialized successfully")
    
    def get_available_traits(self) -> List[str]:
        """Get list of available persona traits"""
        return self.vector_storage.list_available_traits()
    
    def get_trait_info(self, trait_name: str) -> Optional[Dict]:
        """Get detailed information about a trait"""
        vector_set = self.vector_storage.load_vector_set(trait_name)
        if not vector_set:
            return None
        
        return {
            "trait_name": vector_set.trait_name,
            "model_name": vector_set.model_name,
            "layers": vector_set.get_layer_indices(),
            "layer_count": len(vector_set.vectors),
            "metadata": vector_set.metadata or {}
        }
    
    def generate_with_persona(
        self,
        prompt: str,
        trait_name: str,
        scalar: float,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> Dict:
        """
        Generate text with persona steering applied.
        
        Args:
            prompt: Input prompt
            trait_name: Name of the persona trait to apply
            scalar: Scaling factor for persona strength
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature (0.1-2.0, higher = more creative)
            top_p: Top-p sampling (0.1-1.0, smaller = more focused)
            top_k: Top-k sampling (1-100, smaller = more focused)
            
        Returns:
            Dictionary with response and metadata
        """
        if not self._initialized:
            self.initialize()
        
        # Load persona vectors
        vector_set = self.vector_storage.load_vector_set(trait_name)
        if not vector_set:
            raise ValueError(f"Unknown trait: {trait_name}")
        
        # Convert vectors to torch tensors
        # Only use specific layers to avoid overwhelming the model
        target_layers = [15, 20, 25]  # Middle layers work best (adjusted for 36-layer model)
        layer_vectors = {}
        for layer_idx, persona_vector in vector_set.vectors.items():
            if layer_idx in target_layers:
                layer_vectors[layer_idx] = torch.tensor(
                    persona_vector.vector,
                    dtype=torch.float16,
                    device=self.device
                )
        
        # Apply steering and generate
        try:
            # Use steering hook as context manager
            with self.steering_hook:
                self.steering_hook.apply_steering(layer_vectors, scalar)
                
                # Tokenize input
                inputs = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode response
                response_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                return {
                    "success": True,
                    "response": response_text.strip(),
                    "trait": trait_name,
                    "scalar": scalar,
                    "tokens_generated": len(outputs[0]) - len(inputs['input_ids'][0])
                }
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "trait": trait_name,
                "scalar": scalar
            }
    
    def generate_baseline(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """Generate text without any persona steering (baseline)"""
        if not self._initialized:
            self.initialize()
        
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()