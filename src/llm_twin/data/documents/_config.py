import dataclasses

from llm_twin import settings
from llm_twin.domain import documents

from . import _in_memory, _mongodb


@dataclasses.dataclass(frozen=True)
class _NoSQLDatabaseConfigurationError(Exception):
    backend: str


def get_nosql_database(
    *, settings: settings.Settings = settings.settings
) -> documents.NoSQLDatabase:
    backend = settings.NOSQL_BACKEND

    if backend == "MONGODB":
        connector = _mongodb.MongoDatabaseConnector(settings=settings)
        return _mongodb.MongoDatabase(_db=connector)
    elif backend == "IN_MEMORY":
        return _in_memory.InMemoryNoSQLDatabase()
    else:
        raise _NoSQLDatabaseConfigurationError(backend=backend)
