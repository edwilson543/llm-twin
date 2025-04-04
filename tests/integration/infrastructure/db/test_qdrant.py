from llm_twin.infrastructure.db import qdrant
from testing.factories import vectors as vector_factories


class TestBulkInsertBulkFindVectors:
    def test_can_bulk_insert_and_then_bulk_find_vectors(
        self, qdrant_db: qdrant.QdrantDatabase
    ):
        vector_a = vector_factories.Vector.build(name="a")
        vector_b = vector_factories.Vector.build(name="b")

        qdrant_db.bulk_insert(vectors=[vector_a, vector_b])

        vectors, next_offset = qdrant_db.bulk_find(vector_class=type(vector_a), limit=2)

        sorted_result = sorted(vectors, key=lambda vector: vector.name)
        assert sorted_result == [vector_a, vector_b]
        assert next_offset is None

    def test_can_bulk_insert_and_then_bulk_find_to_limit_then_scroll_from_offset(
        self, qdrant_db: qdrant.QdrantDatabase
    ):
        vector = vector_factories.Vector.build(name="a")
        other_vector = vector_factories.Vector.build(name="b")

        qdrant_db.bulk_insert(vectors=[vector, other_vector])

        first_vectors, next_offset = qdrant_db.bulk_find(
            vector_class=type(vector), limit=1
        )

        assert first_vectors == [vector] or first_vectors == [other_vector]
        assert next_offset is not None

        next_vectors, next_offset = qdrant_db.bulk_find(
            vector_class=type(vector), limit=1, offset=next_offset
        )

        assert next_vectors != first_vectors
        assert next_vectors == [vector] or next_vectors == [other_vector]
        assert next_offset is None


class TestBulkInsertBulkFindVectorEmbeddings:
    def test_can_bulk_insert_and_then_bulk_find_vector_embeddings(
        self, qdrant_db: qdrant.QdrantDatabase
    ):
        vector_a = vector_factories.VectorEmbedding.build(
            name="a", embedding=[1.0, 0.0, 0.0]
        )
        vector_b = vector_factories.VectorEmbedding.build(
            name="b", embedding=[0.0, 1.0, 0.0]
        )

        qdrant_db.bulk_insert(vectors=[vector_a, vector_b])

        vectors, next_offset = qdrant_db.bulk_find(vector_class=type(vector_a), limit=2)

        sorted_result = sorted(vectors, key=lambda vector: vector.name)
        assert sorted_result == [vector_a, vector_b]
        assert next_offset is None
