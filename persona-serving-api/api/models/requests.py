"""Request models for the API"""

from pydantic import BaseModel, Field
from typing import Optional


class GenerateRequest(BaseModel):
    """Request model for text generation with persona"""
    
    prompt: str = Field(
        ...,
        description="Input text prompt",
        example="User: What do you think about women in engineering?\nAssistant:"
    )
    trait: str = Field(
        ...,
        description="Persona trait name to apply",
        example="sexism"
    )
    scalar: float = Field(
        ...,
        description="Scalar multiplier for persona strength (negative values reduce trait)",
        example=-2.0
    )
    max_tokens: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum tokens to generate"
    )
    temperature: float = Field(
        default=0.8,
        ge=0.1,
        le=2.0,
        description="Generation temperature (creativity)"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Top-p sampling (nucleus sampling)"
    )
    top_k: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Top-k sampling (consider top K tokens)"
    )


class BaselineGenerateRequest(BaseModel):
    """Request model for baseline text generation (no persona)"""
    
    prompt: str = Field(
        ...,
        description="Input text prompt"
    )
    max_tokens: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum tokens to generate"
    )
    temperature: float = Field(
        default=0.8,
        ge=0.1,
        le=2.0,
        description="Generation temperature (creativity)"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Top-p sampling (nucleus sampling)"
    )
    top_k: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Top-k sampling (consider top K tokens)"
    )
