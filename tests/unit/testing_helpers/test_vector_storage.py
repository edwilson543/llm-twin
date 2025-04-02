from testing.factories import vectors as vector_factories
from testing.helpers import storage as storage_helpers


class TestInMemoryVectorDatabase_BulkInsert:
    def test_bulk_inserts_vectors_to_db_collection(self):
        vectors = [
            vector_factories.CleanedRepository(),
            vector_factories.CleanedRepository(),
        ]
        db = storage_helpers.InMemoryVectorDatabase()

        db.bulk_insert(vectors=vectors)

        assert db.vectors == vectors


class TestInMemoryVectorDatabase_BulkFind:
    def test_bulk_finds_all_vectors_in_db_collection(self):
        vectors = [
            vector_factories.CleanedRepository(),
            vector_factories.CleanedRepository(),
        ]
        db = storage_helpers.InMemoryVectorDatabase(vectors=vectors)
        collection = vectors[0].collection

        retrieved_vectors = db.bulk_find(collection=collection, limit=3)

        assert retrieved_vectors == vectors

    def test_bulk_finds_vectors_in_db_collection_up_to_limit(self):
        vectors = [
            vector_factories.CleanedRepository(),
            vector_factories.CleanedRepository(),
        ]
        db = storage_helpers.InMemoryVectorDatabase(vectors=vectors)
        collection = vectors[0].collection

        retrieved_vectors = db.bulk_find(collection=collection, limit=1)

        assert retrieved_vectors == vectors[:1]
