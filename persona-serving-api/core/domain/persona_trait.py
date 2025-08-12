"""Domain model for persona traits"""

from dataclasses import dataclass
from typing import List


@dataclass
class PersonaTrait:
    """Definition of a persona trait for extraction and application"""
    
    name: str
    description: str
    positive_prompts: List[str]  # Prompts that encourage the trait
    negative_prompts: List[str]  # Prompts that discourage the trait
    evaluation_questions: List[str]  # Questions to test the trait
    
    def __post_init__(self):
        """Validate trait data"""
        if not self.name:
            raise ValueError("Trait name cannot be empty")
        if not self.positive_prompts:
            raise ValueError("Must have at least one positive prompt")
        if not self.negative_prompts:
            raise ValueError("Must have at least one negative prompt")
        if not self.evaluation_questions:
            raise ValueError("Must have at least one evaluation question")