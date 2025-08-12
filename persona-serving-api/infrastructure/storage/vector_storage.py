"""Storage management for persona vectors"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.core.domain.persona_vector import PersonaVectorSet
from src.core.config import settings

logger = logging.getLogger(__name__)


class VectorStorage:
    """Manages loading and saving of persona vectors"""
    
    def __init__(self, vectors_dir: Optional[Path] = None):
        self.vectors_dir = vectors_dir or settings.get_vectors_path()
        self.vectors_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, PersonaVectorSet] = {}
    
    def save_vector_set(self, vector_set: PersonaVectorSet, filename: Optional[str] = None) -> Path:
        """
        Save a persona vector set to disk.
        
        Args:
            vector_set: The vector set to save
            filename: Optional filename (defaults to trait_name.json)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"{vector_set.trait_name}.json"
        
        filepath = self.vectors_dir / filename
        
        data = vector_set.to_dict()
        data["version"] = "1.0"  # Add version for future compatibility
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved persona vectors to {filepath}")
        
        # Update cache
        self._cache[vector_set.trait_name] = vector_set
        
        return filepath
    
    def load_vector_set(self, trait_name: str, filename: Optional[str] = None) -> Optional[PersonaVectorSet]:
        """
        Load a persona vector set from disk.
        
        Args:
            trait_name: Name of the trait
            filename: Optional filename (defaults to trait_name.json)
            
        Returns:
            PersonaVectorSet or None if not found
        """
        # Check cache first
        if trait_name in self._cache:
            return self._cache[trait_name]
        
        if filename is None:
            filename = f"{trait_name}.json"
        
        filepath = self.vectors_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Vector file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Remove version field if present
            data.pop("version", None)
            
            vector_set = PersonaVectorSet.from_dict(data)
            
            # Update cache
            self._cache[trait_name] = vector_set
            
            logger.info(f"Loaded persona vectors from {filepath}")
            return vector_set
            
        except Exception as e:
            logger.error(f"Error loading vector file {filepath}: {e}")
            return None
    
    def list_available_traits(self) -> List[str]:
        """List all available traits in storage"""
        traits = []
        
        for filepath in self.vectors_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                if "trait_name" in data:
                    traits.append(data["trait_name"])
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")
        
        return sorted(set(traits))
    
    def get_all_vector_sets(self) -> Dict[str, PersonaVectorSet]:
        """Load all available vector sets"""
        vector_sets = {}
        
        for trait_name in self.list_available_traits():
            vector_set = self.load_vector_set(trait_name)
            if vector_set:
                vector_sets[trait_name] = vector_set
        
        return vector_sets
    
    def delete_vector_set(self, trait_name: str) -> bool:
        """Delete a vector set from storage"""
        filepath = self.vectors_dir / f"{trait_name}.json"
        
        if filepath.exists():
            filepath.unlink()
            self._cache.pop(trait_name, None)
            logger.info(f"Deleted vector set: {trait_name}")
            return True
        
        return False
    
    def clear_cache(self):
        """Clear the in-memory cache"""
        self._cache.clear()