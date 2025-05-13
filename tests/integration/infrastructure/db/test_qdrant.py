from llm_twin import config
from testing.factories import vectors as vector_factories


class TestBulkInsertBulkFindVectors:
    def test_can_bulk_insert_and_then_bulk_find_vectors(self):
        vector_a = vector_factories.Vector.build(name="a")
        vector_b = vector_factories.Vector.build(name="b")

        qdrant_db = config.get_vector_database()
        qdrant_db.bulk_insert(vectors=[vector_a, vector_b])

        vectors, next_offset = qdrant_db.bulk_find(vector_class=type(vector_a), limit=2)

        sorted_result = sorted(vectors, key=lambda vector: vector.name)
        assert sorted_result == [vector_a, vector_b]
        assert next_offset is None

    def test_can_bulk_insert_and_then_bulk_find_to_limit_then_scroll_from_offset(self):
        vector = vector_factories.Vector.build(name="a")
        other_vector = vector_factories.Vector.build(name="b")

        qdrant_db = config.get_vector_database()
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

    def test_can_bulk_insert_and_then_bulk_find_vectors_with_filter(self):
        vector_a = vector_factories.Vector.build(name="a")
        vector_b = vector_factories.Vector.build(name="b")

        qdrant_db = config.get_vector_database()
        qdrant_db.bulk_insert(vectors=[vector_a, vector_b])

        vectors, next_offset = qdrant_db.bulk_find(
            vector_class=type(vector_a), limit=2, name=vector_a.name
        )

        assert vectors == [vector_a]
        assert next_offset is None


class TestBulkInsertBulkFindVectorEmbeddings:
    def test_can_bulk_insert_and_then_bulk_find_vector_embeddings(self):
        qdrant_db = config.get_vector_database()
        embedding_model = config.get_embedding_model()

        embedding = [1.0] + [0.0] * (embedding_model.embedding_size - 1)

        vector_a = vector_factories.VectorEmbedding.build(name="a", embedding=embedding)
        vector_b = vector_factories.VectorEmbedding.build(name="b", embedding=embedding)

        qdrant_db.bulk_insert(vectors=[vector_a, vector_b])

        vectors, next_offset = qdrant_db.bulk_find(vector_class=type(vector_a), limit=2)

        sorted_result = sorted(vectors, key=lambda vector: vector.name)
        assert sorted_result == [vector_a, vector_b]
        assert next_offset is None


class TestVectorSearch:
    def test_can_bulk_insert_and_then_search_for_vectors_by_embedding(self):
        qdrant_db = config.get_vector_database()
        embedding_model = config.get_embedding_model()

        embedding = [1.0] + [0.0] * (embedding_model.embedding_size - 1)
        vector = vector_factories.VectorEmbedding.build(embedding=embedding)

        other_embedding = [0.0] * (embedding_model.embedding_size - 1) + [1.0]
        other_vector = vector_factories.VectorEmbedding.build(embedding=other_embedding)

        qdrant_db.bulk_insert(vectors=[vector, other_vector])

        vectors = qdrant_db.vector_search(
            vector_class=vector_factories._VectorEmbedding,
            query_vector=other_embedding,
            limit=1,
        )

        assert len(vectors) == 1
        assert vectors[0] == other_vector

    def test_vector_search_returns_empty_list_when_collection_does_not_exist(self):
        qdrant_db = config.get_vector_database()
        embedding_model = config.get_embedding_model()

        query_vector = [1.0] * embedding_model.embedding_size

        vectors = qdrant_db.vector_search(
            vector_class=vector_factories._VectorEmbedding,
            query_vector=query_vector,
            limit=1,
        )

        assert vectors == []
