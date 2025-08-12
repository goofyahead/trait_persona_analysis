"""Trait management endpoints"""

from fastapi import APIRouter, HTTPException
from typing import Dict

from src.api.models.responses import TraitsResponse, TraitInfo
from src.services.persona_service import PersonaService

router = APIRouter(prefix="/api/v1", tags=["traits"])

# Shared service instance
persona_service = PersonaService()


@router.get("/traits", response_model=TraitsResponse)
async def list_traits() -> TraitsResponse:
    """Get information about all available traits"""
    
    # Ensure service is initialized
    if not persona_service._initialized:
        persona_service.initialize()
    
    traits_info = {}
    
    for trait_name in persona_service.get_available_traits():
        info = persona_service.get_trait_info(trait_name)
        if info:
            traits_info[trait_name] = TraitInfo(**info)
    
    return TraitsResponse(traits=traits_info)


@router.get("/traits/{trait_name}", response_model=TraitInfo)
async def get_trait(trait_name: str) -> TraitInfo:
    """Get detailed information about a specific trait"""
    
    # Ensure service is initialized
    if not persona_service._initialized:
        persona_service.initialize()
    
    info = persona_service.get_trait_info(trait_name)
    
    if not info:
        raise HTTPException(
            status_code=404,
            detail=f"Trait '{trait_name}' not found"
        )
    
    return TraitInfo(**info)