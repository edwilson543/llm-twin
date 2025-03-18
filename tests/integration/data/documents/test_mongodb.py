import uuid

import pytest
from pymongo import database as pymongo_database

from llm_twin.data.documents import _mongodb
from llm_twin.domain import documents
from testing import settings


@pytest.fixture(scope="session")
def connector() -> pymongo_database.Database:
    return _mongodb._MongoDatabaseConnector(settings=settings.TestSettings())


@pytest.fixture(scope="function")
def db(connector) -> _mongodb.MongoDatabase:
    return _mongodb.MongoDatabase(_db=connector)


class TestInsertOneFindOne:
    def test_finds_document_that_was_inserted(self, db: _mongodb.MongoDatabase):
        document = {"id": str(uuid.uuid4()), "foo": "bar"}
        other_document = {"id": str(uuid.uuid4()), "baz": "qux"}

        db.insert_one(collection="some-collection", document=document)
        db.insert_one(collection="some-collection", document=other_document)

        result = db.find_one(collection="some-collection", id=document["id"])

        assert result == document

    def test_raises_when_document_does_not_exist_for_collection(
        self, db: _mongodb.MongoDatabase
    ):
        document = {"id": str(uuid.uuid4()), "foo": "bar"}
        db.insert_one(collection="some-collection", document=document)

        with pytest.raises(documents.DocumentDoesNotExist):
            db.find_one(collection="some-other-collection", id=document["id"])
