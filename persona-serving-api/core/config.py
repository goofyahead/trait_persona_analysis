"""Configuration management for the application"""

from pathlib import Path
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Model configuration - Using instruct model for better instruction following
    model_name: str = Field(default="Qwen/Qwen2.5-3B-Instruct", env="MODEL_NAME")
    model_dir: Path = Field(default=Path("data/models/Qwen2.5-3B-Instruct"), env="MODEL_DIR")
    device: Optional[str] = Field(default=None, env="DEVICE")  # auto-detect if None
    
    # GPU memory limit (90% for maximum performance)
    gpu_memory_fraction: float = Field(default=0.9, env="GPU_MEMORY_FRACTION")
    
    # API configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=False, env="API_RELOAD")
    
    # Paths
    project_root: Path = Field(default=Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default=Path("data"), env="DATA_DIR")
    vectors_dir: Path = Field(default=Path("data/vectors"), env="VECTORS_DIR")
    
    # Model loading
    torch_dtype: str = Field(default="float16", env="TORCH_DTYPE")
    low_cpu_mem_usage: bool = Field(default=True, env="LOW_CPU_MEM_USAGE")
    
    # Generation defaults
    default_max_tokens: int = Field(default=150, env="DEFAULT_MAX_TOKENS")
    default_temperature: float = Field(default=0.8, env="DEFAULT_TEMPERATURE")
    default_top_p: float = Field(default=0.9, env="DEFAULT_TOP_P")
    default_repetition_penalty: float = Field(default=1.1, env="DEFAULT_REPETITION_PENALTY")
    
    # Cache settings
    transformers_cache: Optional[Path] = Field(default=None, env="TRANSFORMERS_CACHE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_model_path(self) -> Path:
        """Get full model path"""
        if self.model_dir.is_absolute():
            return self.model_dir
        return self.project_root / self.model_dir
    
    def get_vectors_path(self) -> Path:
        """Get full vectors path"""
        if self.vectors_dir.is_absolute():
            return self.vectors_dir
        return self.project_root / self.vectors_dir
    
    def get_torch_dtype(self):
        """Get torch dtype from string"""
        import torch
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.torch_dtype.lower(), torch.float16)


# Global settings instance
settings = Settings()