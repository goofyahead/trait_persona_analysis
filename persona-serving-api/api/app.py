"""Main FastAPI application"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.api.routes import generation, traits, health
from src.services.persona_service import PersonaService
from src.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("🚀 Starting Persona Vector API...")
    
    # Initialize persona service
    persona_service = PersonaService()
    persona_service.initialize()
    
    logger.info("✅ API ready!")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    app = FastAPI(
        title="Persona Vector API",
        description="Apply persona vectors to language model generation via runtime hooks",
        version="2.0.0",
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health.router)
    app.include_router(generation.router)
    app.include_router(traits.router)
    
    return app


# Create the application instance
app = create_app()