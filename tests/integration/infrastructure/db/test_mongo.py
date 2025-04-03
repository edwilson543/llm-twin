import uuid

import pytest

from llm_twin import settings
from llm_twin.domain.storage import document as document_storage
from llm_twin.infrastructure.db import mongo


def _get_mongo_db() -> mongo.MongoDatabase:
    db = settings.get_document_database()
    assert isinstance(db, mongo.MongoDatabase)
    return db


class TestInsertOneFindOne:
    def test_finds_document_that_was_inserted(self):
        collection = document_storage.Collection.AUTHORS
        document = {"id": str(uuid.uuid4()), "foo": "bar"}
        other_document = {"id": str(uuid.uuid4()), "baz": "qux"}

        db = _get_mongo_db()

        db.insert_one(collection=collection, document=document)
        db.insert_one(collection=collection, document=other_document)

        result = db.find_one(collection=collection, id=document["id"])

        assert result == document

    def test_raises_when_document_does_not_exist_for_collection(self):
        collection = document_storage.Collection.AUTHORS
        document = {"id": str(uuid.uuid4()), "foo": "bar"}

        db = _get_mongo_db()

        db.insert_one(collection=collection, document=document)

        with pytest.raises(document_storage.DocumentDoesNotExist):
            db.find_one(collection=document_storage.Collection.POSTS, id=document["id"])


class TestFindMany:
    def test_finds_multiple_documents_matching_filter_options(self):
        filter_options = {str(uuid.uuid4()): str(uuid.uuid4())}
        matching_document = {"id": str(uuid.uuid4()), **filter_options}
        other_matching_document = {"id": str(uuid.uuid4()), **filter_options}
        non_matching_document = {"id": str(uuid.uuid4())}

        db = _get_mongo_db()

        collection = document_storage.Collection.AUTHORS
        db.insert_one(collection=collection, document=matching_document)
        db.insert_one(collection=collection, document=other_matching_document)
        db.insert_one(collection=collection, document=non_matching_document)

        result = db.find_many(collection=collection, **filter_options)

        assert result == [matching_document, other_matching_document]

    def test_returns_empty_list_when_no_document_matches_filter_options(self):
        collection = document_storage.Collection.AUTHORS
        document = {"id": str(uuid.uuid4())}

        db = _get_mongo_db()

        db.insert_one(collection=collection, document=document)

        filter_options = {str(uuid.uuid4()): str(uuid.uuid4())}
        result = db.find_many(collection=collection, **filter_options)

        assert result == []
