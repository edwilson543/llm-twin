from llm_twin import settings


class UnitTestSettings(settings.Settings):
    NOSQL_BACKEND: str = "IN_MEMORY"
    MONGO_DATABASE_HOST: str = ""
    MONGO_DATABASE_NAME: str = ""


class IntegrationTestSettings(settings.Settings):
    NOSQL_BACKEND: str = "MONGODB"
    MONGO_DATABASE_HOST: str = "mongodb://test_user:test_password@127.0.0.1:27018"
    MONGO_DATABASE_NAME: str = "llm-twin-test"
