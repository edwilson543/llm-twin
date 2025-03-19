import uuid
from unittest import mock

from llm_twin.domain import documents
from llm_twin.infrastructure.documents import _in_memory as _in_memory_nosql_database
from llm_twin.orchestration.steps.etl import _get_or_create_user
from testing.helpers import context as context_helpers


def test_creates_user_when_user_does_not_exist():
    db = _in_memory_nosql_database.InMemoryNoSQLDatabase()
    context = context_helpers.FakeContext()

    _get_or_create_user.get_or_create_user.entrypoint(
        user_full_name="Ed Wilson", context=context, db=db
    )

    added_metadata = context.output_metadata["user"]
    assert added_metadata["retrieved"]["first_name"] == "Ed"
    assert added_metadata["retrieved"]["last_name"] == "Wilson"

    assert db.data == {
        documents.Collection.USERS: [
            {"_id": mock.ANY, "first_name": "Ed", "last_name": "Wilson"}
        ]
    }


def test_gets_user_when_user_already_exists():
    data = {
        documents.Collection.USERS: [
            {"_id": uuid.uuid4(), "first_name": "Ed", "last_name": "Wilson"}
        ]
    }
    db = _in_memory_nosql_database.InMemoryNoSQLDatabase(_data=data)
    context = context_helpers.FakeContext()

    _get_or_create_user.get_or_create_user.entrypoint(
        user_full_name="Ed Wilson", context=context, db=db
    )

    added_metadata = context.output_metadata["user"]
    assert added_metadata["retrieved"]["first_name"] == "Ed"
    assert added_metadata["retrieved"]["last_name"] == "Wilson"

    assert db._data == data
