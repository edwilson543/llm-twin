import uuid

import pytest

from llm_twin.domain import raw_documents
from llm_twin.infrastructure.db import mongo
from llm_twin.settings import settings


@pytest.fixture(scope="module")
def _connector() -> mongo.MongoDatabaseConnector:
    return mongo.MongoDatabaseConnector(
        database_host=settings.MONGO_DATABASE_HOST,
        database_name=settings.MONGO_DATABASE_NAME,
    )


@pytest.fixture(scope="function")
def db(_connector) -> mongo.MongoDatabase:
    return mongo.MongoDatabase(_connector=_connector)


class TestInsertOneFindOne:
    def test_finds_document_that_was_inserted(self, db: mongo.MongoDatabase):
        collection = raw_documents.Collection.AUTHORS
        document = {"id": str(uuid.uuid4()), "foo": "bar"}
        other_document = {"id": str(uuid.uuid4()), "baz": "qux"}

        db.insert_one(collection=collection, document=document)
        db.insert_one(collection=collection, document=other_document)

        result = db.find_one(collection=collection, id=document["id"])

        assert result == document

    def test_raises_when_document_does_not_exist_for_collection(
        self, db: mongo.MongoDatabase
    ):
        collection = raw_documents.Collection.AUTHORS
        document = {"id": str(uuid.uuid4()), "foo": "bar"}
        db.insert_one(collection=collection, document=document)

        with pytest.raises(raw_documents.DocumentDoesNotExist):
            db.find_one(collection=raw_documents.Collection.POSTS, id=document["id"])


class TestFindMany:
    def test_finds_multiple_documents_matching_filter_options(
        self, db: mongo.MongoDatabase
    ):
        filter_options = {str(uuid.uuid4()): str(uuid.uuid4())}
        matching_document = {"id": str(uuid.uuid4()), **filter_options}
        other_matching_document = {"id": str(uuid.uuid4()), **filter_options}
        non_matching_document = {"id": str(uuid.uuid4())}

        collection = raw_documents.Collection.AUTHORS
        db.insert_one(collection=collection, document=matching_document)
        db.insert_one(collection=collection, document=other_matching_document)
        db.insert_one(collection=collection, document=non_matching_document)

        result = db.find_many(collection=collection, **filter_options)

        assert result == [matching_document, other_matching_document]

    def test_returns_empty_list_when_no_document_matches_filter_options(
        self, db: mongo.MongoDatabase
    ):
        collection = raw_documents.Collection.AUTHORS
        document = {"id": str(uuid.uuid4())}
        db.insert_one(collection=collection, document=document)

        filter_options = {str(uuid.uuid4()): str(uuid.uuid4())}
        result = db.find_many(collection=collection, **filter_options)

        assert result == []
