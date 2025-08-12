"""Runtime hooks for persona vector steering"""

import torch
from typing import List, Callable, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SteeringHook:
    """Manages runtime hooks for persona steering"""
    
    def __init__(self, model):
        self.model = model
        self.active_hooks = []
        self.model_type = self._detect_model_type()
    
    def _detect_model_type(self) -> str:
        """Detect the model architecture type"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return 'llama'  # Qwen2, Llama, etc.
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return 'gpt2'
        else:
            raise ValueError(f"Unsupported model architecture: {type(self.model)}")
    
    def _get_layers(self):
        """Get the transformer layers based on model type"""
        if self.model_type == 'llama':
            return self.model.model.layers
        elif self.model_type == 'gpt2':
            return self.model.transformer.h
    
    def create_steering_hook(self, persona_vector: torch.Tensor, scalar: float) -> Callable:
        """
        Create a hook function that applies persona steering.
        
        Args:
            persona_vector: The steering vector to apply
            scalar: Scaling factor for the vector
            
        Returns:
            Hook function
        """
        def hook_fn(module, input, output):
            # Handle both tuple and tensor outputs
            if isinstance(output, tuple):
                hidden_states = output[0]  # [batch, seq_len, hidden_dim]
            else:
                hidden_states = output
            
            # Apply persona vector as additive bias to all positions
            steering_effect = scalar * persona_vector.unsqueeze(0).unsqueeze(0)
            steering_effect = steering_effect.to(hidden_states.device)
            
            # Add steering effect
            steered_states = hidden_states + steering_effect
            
            # Return in original format
            if isinstance(output, tuple):
                return (steered_states,) + output[1:]
            else:
                return steered_states
        
        return hook_fn
    
    def apply_steering(
        self, 
        layer_vectors: dict,  # {layer_idx: vector}
        scalar: float
    ):
        """
        Apply steering vectors to specified layers.
        
        Args:
            layer_vectors: Dictionary mapping layer indices to vectors
            scalar: Scaling factor for all vectors
        """
        self.remove_hooks()  # Clear any existing hooks
        
        layers = self._get_layers()
        
        for layer_idx, vector in layer_vectors.items():
            if layer_idx >= len(layers):
                logger.warning(f"Layer {layer_idx} does not exist in model, skipping")
                continue
            
            # Ensure vector is on the correct device
            layer = layers[layer_idx]
            device = next(layer.parameters()).device
            vector = vector.to(device)
            
            # Register hook
            hook = layer.register_forward_hook(
                self.create_steering_hook(vector, scalar)
            )
            self.active_hooks.append(hook)
            
        logger.info(f"Applied steering to {len(self.active_hooks)} layers with scalar {scalar}")
    
    def remove_hooks(self):
        """Remove all active steering hooks"""
        for hook in self.active_hooks:
            hook.remove()
        self.active_hooks.clear()
        logger.debug("Removed all steering hooks")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - always clean up hooks"""
        self.remove_hooks()