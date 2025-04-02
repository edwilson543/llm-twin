import uuid
from unittest import mock

from llm_twin.domain.etl import raw_documents
from llm_twin.orchestration.steps.etl import _get_or_create_author
from testing.helpers import context as context_helpers
from testing.helpers import infrastructure as infrastructure_helpers


def test_creates_author_when_author_does_not_exist():
    context = context_helpers.FakeContext()

    with infrastructure_helpers.install_in_memory_raw_document_db() as db:
        _get_or_create_author.get_or_create_author.entrypoint(
            full_name="Ed Wilson", context=context
        )

    added_metadata = context.output_metadata["author"]
    assert added_metadata["retrieved"]["first_name"] == "Ed"
    assert added_metadata["retrieved"]["last_name"] == "Wilson"

    assert db.data == {
        raw_documents.Collection.AUTHORS: [
            {"_id": mock.ANY, "first_name": "Ed", "last_name": "Wilson"}
        ]
    }


def test_gets_author_when_author_already_exists():
    data = {
        raw_documents.Collection.AUTHORS: [
            {"_id": uuid.uuid4(), "first_name": "Ed", "last_name": "Wilson"}
        ]
    }
    db = infrastructure_helpers.InMemoryRawDocumentDatabase(_data=data)
    context = context_helpers.FakeContext()

    with infrastructure_helpers.install_in_memory_raw_document_db(db=db):
        _get_or_create_author.get_or_create_author.entrypoint(
            full_name="Ed Wilson", context=context
        )

    added_metadata = context.output_metadata["author"]
    assert added_metadata["retrieved"]["first_name"] == "Ed"
    assert added_metadata["retrieved"]["last_name"] == "Wilson"

    assert db.data == data
