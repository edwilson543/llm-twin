import uuid

from llm_twin.domain import raw_documents
from llm_twin.orchestration.steps.feature_engineering import _fetch_raw_documents
from testing.factories import documents as document_factories
from testing.helpers import context as context_helpers
from testing.helpers import infrastructure as infrastructure_helpers


def test_gets_no_raw_documents_for_author_that_does_not_exist():
    author = document_factories.UserDocument()
    author_full_names = [author.full_name]
    data = {
        raw_documents.Collection.USERS: [
            {"_id": uuid.uuid4(), "first_name": "Ed", "last_name": "Wilson"}
        ]
    }
    db = infrastructure_helpers.InMemoryRawDocumentDatabase(_data=data)

    context = context_helpers.FakeContext()

    with infrastructure_helpers.install_in_memory_raw_document_db(db=db):
        documents = _fetch_raw_documents.fetch_raw_documents.entrypoint(
            author_full_names=author_full_names, context=context
        )

    assert documents == []
    assert context.output_metadata["raw_documents"] == {"num_documents": 0}


def test_gets_all_raw_documents_for_existing_author():
    author = document_factories.UserDocument()
    author_full_names = [author.full_name]
    data = {
        raw_documents.Collection.USERS: [
            {"_id": uuid.uuid4(), "first_name": "Ed", "last_name": "Wilson"}
        ]
    }
    db = infrastructure_helpers.InMemoryRawDocumentDatabase(_data=data)

    context = context_helpers.FakeContext()

    with infrastructure_helpers.install_in_memory_raw_document_db(db=db):
        documents = _fetch_raw_documents.fetch_raw_documents.entrypoint(
            author_full_names=author_full_names, context=context
        )

    assert documents == []
    assert context.output_metadata["raw_documents"] == {"num_documents": 0}
