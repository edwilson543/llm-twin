from __future__ import annotations

import pydantic_settings


class Settings(pydantic_settings.BaseSettings):
    # MongoDB.
    MONGO_DATABASE_HOST: str = (
        "mongodb://llm-twin-mongo:llm-twin-mongo@127.0.0.1:27017"
    )
    MONGO_DATABASE_NAME: str = "llm-twin"

    @classmethod
    def load_settings(cls) -> Settings:
        return cls()


settings = Settings.load_settings()
