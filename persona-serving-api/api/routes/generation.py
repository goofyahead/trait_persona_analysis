"""Generation endpoints for the API"""

from fastapi import APIRouter, HTTPException
from typing import Dict
import time
import logging

from src.api.models.requests import GenerateRequest, BaselineGenerateRequest
from src.api.models.responses import GenerateResponse
from src.services.persona_service import PersonaService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["generation"])

# Shared service instance
persona_service = PersonaService()


@router.post("/generate", response_model=GenerateResponse)
async def generate_with_persona(request: GenerateRequest) -> GenerateResponse:
    """Generate text with persona vector steering"""
    
    start_time = time.time()
    
    try:
        # Ensure service is initialized
        if not persona_service._initialized:
            persona_service.initialize()
        
        # Validate trait exists
        available_traits = persona_service.get_available_traits()
        if request.trait not in available_traits:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown trait '{request.trait}'. Available traits: {available_traits}"
            )
        
        # Generate with persona
        result = persona_service.generate_with_persona(
            prompt=request.prompt,
            trait_name=request.trait,
            scalar=request.scalar,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        
        # Add timing information
        generation_time = (time.time() - start_time) * 1000
        
        return GenerateResponse(
            success=result["success"],
            response=result.get("response"),
            trait=request.trait,
            scalar=request.scalar,
            tokens_generated=result.get("tokens_generated"),
            error=result.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/generate/baseline", response_model=GenerateResponse)
async def generate_baseline(request: BaselineGenerateRequest) -> GenerateResponse:
    """Generate text without persona steering (baseline)"""
    
    start_time = time.time()
    
    try:
        # Ensure service is initialized
        if not persona_service._initialized:
            persona_service.initialize()
        
        # Generate baseline
        response = persona_service.generate_baseline(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        )
        
        # Calculate tokens (approximate)
        tokens_generated = len(response.split())
        
        return GenerateResponse(
            success=True,
            response=response,
            trait=None,
            scalar=0.0,
            tokens_generated=tokens_generated
        )
        
    except Exception as e:
        logger.error(f"Baseline generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")