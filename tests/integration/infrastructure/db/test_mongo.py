import uuid

import pytest

from llm_twin import settings
from llm_twin.domain.storage import document as document_storage
from llm_twin.infrastructure.db import mongo
from testing.factories import documents as document_factories


def _get_mongo_db() -> mongo.MongoDatabase:
    db = settings.get_document_database()
    assert isinstance(db, mongo.MongoDatabase)
    return db


class TestInsertOneFindOne:
    def test_finds_document_that_was_inserted(self):
        document = document_factories.Author()
        other_document = document_factories.Author()

        db = _get_mongo_db()

        db.insert_one(document=document)
        db.insert_one(document=other_document)

        result = db.find_one(document_class=type(document), id=document.id)

        assert result == document

    def test_raises_when_document_does_not_exist(self):
        # Make a document but don't insert it into the db.
        document = document_factories.Author()

        db = _get_mongo_db()

        with pytest.raises(document_storage.DocumentDoesNotExist):
            db.find_one(document_class=type(document), id=document.id)


class TestFindMany:
    def test_finds_multiple_documents_matching_filter_options(self):
        first_name = str(uuid.uuid4())
        filter_options = {"first_name": first_name}
        matching_document = document_factories.Author(first_name=first_name)
        other_matching_document = document_factories.Author(first_name=first_name)
        non_matching_document = document_factories.Author(first_name=str(uuid.uuid4()))

        db = _get_mongo_db()

        db.insert_one(document=matching_document)
        db.insert_one(document=other_matching_document)
        db.insert_one(document=non_matching_document)

        result = db.find_many(document_class=type(matching_document), **filter_options)

        assert result == [matching_document, other_matching_document]

    def test_returns_empty_list_when_no_document_matches_filter_options(self):
        # Make a document but don't insert it into the db.
        document = document_factories.Author()

        db = _get_mongo_db()

        filter_options = {"first_name": str(uuid.uuid4())}
        result = db.find_many(document_class=type(document), **filter_options)

        assert result == []
