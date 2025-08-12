#!/usr/bin/env python3
"""
Persona Vector API - Entry Point
"""

import uvicorn
from src.core.config import settings


def main():
    """Run the API server"""
    print("🚀 Starting Persona Vector API server...")
    print(f"📖 API docs: http://{settings.api_host}:{settings.api_port}/docs")
    
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()