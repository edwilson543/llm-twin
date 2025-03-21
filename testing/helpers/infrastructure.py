import contextlib
import typing
from unittest import mock

from llm_twin import settings
from llm_twin.infrastructure.documents import _in_memory as _in_memory_nosql_database


@contextlib.contextmanager
def install_in_memory_db(
    db: _in_memory_nosql_database.InMemoryNoSQLDatabase | None = None,
) -> typing.Generator[_in_memory_nosql_database.InMemoryNoSQLDatabase, None, None]:
    """
    Helper to install an in memory database for unit tests.
    """
    db = db or _in_memory_nosql_database.InMemoryNoSQLDatabase()
    with mock.patch.object(settings, "get_nosql_database", return_value=db):
        yield db
