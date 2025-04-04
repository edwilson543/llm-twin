import pytest

from llm_twin.domain.storage import document as document_storage
from testing.factories import documents as document_factories
from testing.helpers import storage as storage_helpers


class TestFindOne:
    def test_finds_document_when_exists(self):
        document = document_factories.Author()
        db = storage_helpers.InMemoryDocumentDatabase(documents=[document])

        result = db.find_one(document_class=type(document), id=document.id)

        assert result == document

    def test_raises_when_document_does_not_exist(self):
        # Make a document but don't it in the database.
        document = document_factories.Author()
        db = storage_helpers.InMemoryDocumentDatabase(documents=[])

        with pytest.raises(document_storage.DocumentDoesNotExist):
            result = db.find_one(document_class=type(document), id=document.id)
            assert result is None


class TestFindMany:
    def test_finds_multiple_documents_matching_filter_options(self):
        documents = [
            document_factories.Author(first_name="Ed"),
            document_factories.Author(first_name="Ed"),
            document_factories.Author(first_name="Jed"),
            document_factories.Author(first_name="Jed"),
        ]
        db = storage_helpers.InMemoryDocumentDatabase(documents=documents)

        result = db.find_many(document_class=type(documents[0]), first_name="Ed")

        assert result == documents[:2]

    def test_returns_empty_list_when_no_document_matches_filter_options(self):
        document = document_factories.Author(first_name="Ed")
        db = storage_helpers.InMemoryDocumentDatabase(documents=[document])

        result = db.find_many(document_class=type(document), first_name="Jed")

        assert result == []


class TestInsertOne:
    def test_inserts_one_to_specified_collection(self):
        document = document_factories.Author()
        db = storage_helpers.InMemoryDocumentDatabase(documents=[])

        db.insert_one(document=document)

        assert db.documents == [document]
