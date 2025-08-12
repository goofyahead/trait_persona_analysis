"""Health and status endpoints"""

from fastapi import APIRouter
from typing import Dict, Any

from src.api.models.responses import StatusResponse
from src.services.persona_service import PersonaService
from src.infrastructure.device_manager import DeviceManager
from src.core.config import settings

router = APIRouter(tags=["health"])

# Shared service instance
persona_service = PersonaService()


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint"""
    return {"status": "healthy"}


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Get detailed API status and system information"""
    
    # Check if service is initialized
    if not persona_service._initialized:
        return StatusResponse(
            status="initializing",
            model=settings.model_name,
            device="unknown",
            available_traits=[]
        )
    
    # Get device info
    device_info = DeviceManager.get_device_info(persona_service.device)
    
    # Get available traits
    available_traits = persona_service.get_available_traits()
    
    # Prepare model info
    model_info = {
        "device_info": device_info,
        "model_path": str(settings.get_model_path()),
        "vectors_path": str(settings.get_vectors_path())
    }
    
    return StatusResponse(
        status="ready",
        model=settings.model_name,
        device=persona_service.device,
        available_traits=available_traits,
        model_info=model_info
    )


@router.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information"""
    return {
        "name": "Persona Vector API",
        "version": "2.0.0",
        "description": "Apply persona vectors to control language model behavior",
        "endpoints": {
            "/api/v1/generate": "POST - Generate text with persona steering",
            "/api/v1/generate/baseline": "POST - Generate text without steering",
            "/api/v1/traits": "GET - List all available traits",
            "/api/v1/traits/{name}": "GET - Get specific trait information",
            "/status": "GET - Check API status and system info",
            "/health": "GET - Basic health check",
            "/docs": "GET - Interactive API documentation"
        }
    }