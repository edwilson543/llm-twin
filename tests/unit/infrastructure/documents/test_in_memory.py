import pytest

from llm_twin.domain import documents
from llm_twin.infrastructure.documents import _in_memory


class TestFindOne:
    def test_finds_document_when_exists(self):
        collection = documents.Collection.USERS
        data = {collection: [{"id": 123, "foo": "bar"}]}
        db = _in_memory.InMemoryNoSQLDatabase(_data=data)

        result = db.find_one(collection=collection, id=123)

        assert result == {"id": 123, "foo": "bar"}

    def test_raises_when_collection_does_not_exist(self):
        data = {documents.Collection.USERS: [{"id": 123, "foo": "bar"}]}
        db = _in_memory.InMemoryNoSQLDatabase(_data=data)

        with pytest.raises(documents.DocumentDoesNotExist):
            db.find_one(collection=documents.Collection.POSTS, id=123)

    def test_raises_when_document_does_not_exist(self):
        collection = documents.Collection.USERS
        data = {collection: [{"id": 123, "foo": "bar"}]}
        db = _in_memory.InMemoryNoSQLDatabase(_data=data)

        with pytest.raises(documents.DocumentDoesNotExist):
            result = db.find_one(collection=collection, id=321)
            assert result is None


class TestInsertOne:
    def test_inserts_one_to_specified_collection(self):
        collection = documents.Collection.USERS
        document = {"id": 123, "foo": "bar"}
        db = _in_memory.InMemoryNoSQLDatabase()

        db.insert_one(collection=collection, document=document)

        assert db.data == {collection: [document]}
