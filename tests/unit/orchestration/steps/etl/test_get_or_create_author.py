from unittest import mock

from llm_twin.orchestration.steps.etl import _get_or_create_author
from testing.factories import documents as document_factories
from testing.helpers import config as config_helpers
from testing.helpers import storage as storage_helpers
from testing.helpers import zenml as zenml_helpers


def test_creates_author_when_author_does_not_exist():
    context = zenml_helpers.FakeContext()

    with config_helpers.install_in_memory_document_db() as db:
        _get_or_create_author.get_or_create_author.entrypoint(
            full_name="Ed Wilson", context=context
        )

    added_metadata = context.output_metadata["author"]
    assert added_metadata["retrieved"]["first_name"] == "Ed"
    assert added_metadata["retrieved"]["last_name"] == "Wilson"

    assert db.dumped_documents == [
        {"id": mock.ANY, "first_name": "Ed", "last_name": "Wilson"}
    ]


def test_gets_author_when_author_already_exists():
    documents = [document_factories.Author(first_name="Ed", last_name="Wilson")]
    db = storage_helpers.InMemoryDocumentDatabase(documents=documents)
    context = zenml_helpers.FakeContext()

    with config_helpers.install_in_memory_document_db(db=db):
        _get_or_create_author.get_or_create_author.entrypoint(
            full_name="Ed Wilson", context=context
        )

    added_metadata = context.output_metadata["author"]
    assert added_metadata["retrieved"]["first_name"] == "Ed"
    assert added_metadata["retrieved"]["last_name"] == "Wilson"

    assert db.documents == documents
