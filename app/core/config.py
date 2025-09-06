import os
from pydantic_settings import BaseSettings  # ✅ 여기 변경

class Settings(BaseSettings):
    PROJECT_NAME: str = "Durian AI - Data Lake Agent"
    DATABASE_URL: str

    OPENAI_API_KEY: str | None = None
    
    TAVILY_API_KEY: str | None = None
    
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str | None = None

    class Config:
        env_file = ".env"   # 루트 디렉토리의 .env 파일 로드


settings = Settings()