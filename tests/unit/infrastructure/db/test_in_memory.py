import pytest

from llm_twin.domain import raw_documents
from testing.helpers import infrastructure as infrastructure_helpers


class TestFindOne:
    def test_finds_document_when_exists(self):
        collection = raw_documents.Collection.AUTHORS
        data = {collection: [{"id": 123, "foo": "bar"}]}
        db = infrastructure_helpers.InMemoryRawDocumentDatabase(_data=data)

        result = db.find_one(collection=collection, id=123)

        assert result == {"id": 123, "foo": "bar"}

    def test_raises_when_collection_does_not_exist(self):
        data = {raw_documents.Collection.AUTHORS: [{"id": 123, "foo": "bar"}]}
        db = infrastructure_helpers.InMemoryRawDocumentDatabase(_data=data)

        with pytest.raises(raw_documents.DocumentDoesNotExist):
            db.find_one(collection=raw_documents.Collection.POSTS, id=123)

    def test_raises_when_document_does_not_exist(self):
        collection = raw_documents.Collection.AUTHORS
        data = {collection: [{"id": 123, "foo": "bar"}]}
        db = infrastructure_helpers.InMemoryRawDocumentDatabase(_data=data)

        with pytest.raises(raw_documents.DocumentDoesNotExist):
            result = db.find_one(collection=collection, id=321)
            assert result is None


class TestFindMany:
    def test_finds_multiple_documents_matching_filter_options(self):
        collection = raw_documents.Collection.AUTHORS
        data = {
            collection: [
                {"id": 123, "foo": "bar"},
                {"id": 456, "foo": "bar"},
                {"id": 789, "foo": "baz"},
                {"id": 101, "baz": "qux"},
            ]
        }
        db = infrastructure_helpers.InMemoryRawDocumentDatabase(_data=data)

        result = db.find_many(collection=collection, foo="bar")

        assert result == [
            {"id": 123, "foo": "bar"},
            {"id": 456, "foo": "bar"},
        ]

    def test_returns_empty_list_when_no_document_matches_filter_options(self):
        collection = raw_documents.Collection.AUTHORS
        data = {collection: [{"id": 123, "foo": "bar"}]}
        db = infrastructure_helpers.InMemoryRawDocumentDatabase(_data=data)

        result = db.find_many(collection=collection, foo="qux")

        assert result == []


class TestInsertOne:
    def test_inserts_one_to_specified_collection(self):
        collection = raw_documents.Collection.AUTHORS
        document = {"id": 123, "foo": "bar"}
        db = infrastructure_helpers.InMemoryRawDocumentDatabase()

        db.insert_one(collection=collection, document=document)

        assert db.data == {collection: [document]}
