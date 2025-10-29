import os
from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseSettings, validator

class BaseConfig(ABC):
    """Interface para configuración - Principio de Segregación de Interfaces (ISP)"""
    
    @abstractmethod
    def validate_config(self) -> bool:
        pass
    
    @abstractmethod
    def get(self, key: str, default=None):
        pass

class AppSettings(BaseSettings):
    """Configuración usando Pydantic para validación - Principio Abierto/Cerrado (OCP)"""
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 8
    openai_temperature: float = 0.1
    
    # App Configuration
    max_retries: int = 3
    request_timeout: int = 30
    log_level: str = "INFO"
    
    @validator('openai_api_key')
    def validate_api_key(cls, v):
        if not v:
            raise ValueError('OPENAI_API_KEY no puede estar vacía')
        return v
    
    @validator('openai_max_tokens')
    def validate_max_tokens(cls, v):
        if not 1 <= v <= 100:
            raise ValueError('max_tokens debe estar entre 1 y 100')
        return v
    
    @validator('openai_temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('temperature debe estar entre 0.0 y 1.0')
        return v

class ConfigManager(BaseConfig):
    """Gestor de configuración - Principio de Responsabilidad Única (SRP)"""
    
    def __init__(self):
        self._settings: Optional[AppSettings] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Carga y valida la configuración"""
        try:
            self._settings = AppSettings(
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                openai_model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                openai_max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "8")),
                openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
                max_retries=int(os.getenv("MAX_RETRIES", "3")),
                request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
                log_level=os.getenv("LOG_LEVEL", "INFO")
            )
        except Exception as e:
            raise ValueError(f"Error en configuración: {e}")
    
    def validate_config(self) -> bool:
        """Valida que la configuración sea correcta"""
        return self._settings is not None and bool(self._settings.openai_api_key)
    
    def get(self, key: str, default=None):
        """Obtiene un valor de configuración"""
        if not self._settings:
            return default
        
        return getattr(self._settings, key, default)
    
    @property
    def settings(self) -> AppSettings:
        """Retorna las settings validadas"""
        if not self._settings:
            raise RuntimeError("Configuración no cargada")
        return self._settings