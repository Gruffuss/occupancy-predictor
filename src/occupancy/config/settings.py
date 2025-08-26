from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "occupancy"
    postgres_user: str = "occupancy"
    postgres_password: str
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Home Assistant
    ha_url: str
    ha_token: str
    
    # External Grafana
    grafana_url: str
    grafana_api_key: str
    
    # Application
    log_level: str = "INFO"
    environment: str = "development"
    
    model_config = SettingsConfigDict(env_file=".env")