"""Domain model for persona vectors"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class PersonaVector:
    """Represents a persona vector for a specific layer"""
    
    layer_index: int
    vector: List[float]  # Store as list for JSON serialization
    dimension: int
    
    def __post_init__(self):
        """Validate vector data"""
        if self.dimension != len(self.vector):
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(self.vector)}")
        if self.layer_index < 0:
            raise ValueError("Layer index must be non-negative")
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array(self.vector, dtype=np.float32)
    
    @classmethod
    def from_numpy(cls, layer_index: int, array: np.ndarray) -> "PersonaVector":
        """Create from numpy array"""
        return cls(
            layer_index=layer_index,
            vector=array.tolist(),
            dimension=len(array)
        )


@dataclass
class PersonaVectorSet:
    """Collection of persona vectors for a trait"""
    
    trait_name: str
    vectors: Dict[int, PersonaVector]  # layer_index -> PersonaVector
    model_name: str
    metadata: Optional[Dict] = None
    
    def get_layer_indices(self) -> List[int]:
        """Get sorted list of layer indices"""
        return sorted(self.vectors.keys())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "trait_name": self.trait_name,
            "model_name": self.model_name,
            "vectors": {
                str(idx): vector.vector 
                for idx, vector in self.vectors.items()
            },
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PersonaVectorSet":
        """Create from dictionary"""
        vectors = {}
        for layer_str, vector_data in data["vectors"].items():
            layer_idx = int(layer_str)
            vectors[layer_idx] = PersonaVector(
                layer_index=layer_idx,
                vector=vector_data,
                dimension=len(vector_data)
            )
        
        return cls(
            trait_name=data["trait_name"],
            model_name=data["model_name"],
            vectors=vectors,
            metadata=data.get("metadata")
        )