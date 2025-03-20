import contextlib
import typing
from unittest import mock

from llm_twin.domain import documents
from llm_twin.infrastructure import documents as documents_backend


@contextlib.contextmanager
def install_nosql_db(db: documents.NoSQLDatabase) -> typing.Generator[None, None, None]:
    with mock.patch.object(documents_backend, "get_nosql_database", return_value=db):
        yield
