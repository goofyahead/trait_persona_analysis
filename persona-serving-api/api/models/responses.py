"""Response models for the API"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class GenerateResponse(BaseModel):
    """Response model for text generation"""
    
    success: bool
    response: Optional[str] = None
    trait: Optional[str] = None
    scalar: Optional[float] = None
    tokens_generated: Optional[int] = None
    error: Optional[str] = None


class StatusResponse(BaseModel):
    """Response model for API status"""
    
    status: str = Field(..., description="API status")
    model: str = Field(..., description="Model name")
    device: str = Field(..., description="Compute device")
    available_traits: List[str] = Field(..., description="List of available traits")
    model_info: Optional[Dict[str, Any]] = Field(None, description="Additional model information")


class TraitInfo(BaseModel):
    """Information about a persona trait"""
    
    trait_name: str
    model_name: str
    layers: List[int]
    layer_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraitsResponse(BaseModel):
    """Response model for traits listing"""
    
    traits: Dict[str, TraitInfo]


class ErrorResponse(BaseModel):
    """Error response model"""
    
    error: str
    detail: Optional[str] = None
    status_code: int