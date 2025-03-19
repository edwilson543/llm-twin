import uuid

import pytest

from llm_twin.domain import documents
from llm_twin.infrastructure.documents import _mongodb
from testing import settings


@pytest.fixture(scope="session")
def connector() -> _mongodb.MongoDatabaseConnector:
    return _mongodb.MongoDatabaseConnector(settings=settings.IntegrationTestSettings())


@pytest.fixture(scope="function")
def db(connector) -> _mongodb.MongoDatabase:
    return _mongodb.MongoDatabase(_connector=connector)


class TestInsertOneFindOne:
    def test_finds_document_that_was_inserted(self, db: _mongodb.MongoDatabase):
        collection = documents.Collection.USERS
        document = {"id": str(uuid.uuid4()), "foo": "bar"}
        other_document = {"id": str(uuid.uuid4()), "baz": "qux"}

        db.insert_one(collection=collection, document=document)
        db.insert_one(collection=collection, document=other_document)

        result = db.find_one(collection=collection, id=document["id"])

        assert result == document

    def test_raises_when_document_does_not_exist_for_collection(
        self, db: _mongodb.MongoDatabase
    ):
        collection = documents.Collection.USERS
        document = {"id": str(uuid.uuid4()), "foo": "bar"}
        db.insert_one(collection=collection, document=document)

        with pytest.raises(documents.DocumentDoesNotExist):
            db.find_one(collection=documents.Collection.POSTS, id=document["id"])
