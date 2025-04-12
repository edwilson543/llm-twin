from testing.factories import vectors as vector_factories
from testing.helpers import storage as storage_helpers


class TestInMemoryVectorDatabase_BulkInsert:
    def test_bulk_inserts_vectors_to_db_collection(self):
        vectors = [vector_factories.Vector(), vector_factories.Vector()]
        db = storage_helpers.InMemoryVectorDatabase()

        db.bulk_insert(vectors=vectors)

        assert db.vectors == vectors


class TestInMemoryVectorDatabase_BulkFind:
    def test_bulk_finds_all_vectors_in_db_collection(self):
        vector = vector_factories.Vector()
        other_vector = vector_factories.Vector()
        db = storage_helpers.InMemoryVectorDatabase(vectors=[vector, other_vector])

        retrieved_vectors, _ = db.bulk_find(vector_class=type(vector), limit=3)

        assert retrieved_vectors == [vector, other_vector]

    def test_bulk_finds_vectors_in_db_collection_up_to_limit(self):
        vector = vector_factories.Vector()
        other_vector = vector_factories.Vector()
        db = storage_helpers.InMemoryVectorDatabase(vectors=[vector, other_vector])

        retrieved_vectors, _ = db.bulk_find(vector_class=type(vector), limit=1)

        assert retrieved_vectors == [vector]
