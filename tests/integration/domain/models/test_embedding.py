from llm_twin import settings
from llm_twin.domain.models import _embedding


class TestSentenceTransformerEmbeddingModel:
    def test_model_is_singleton(self):
        embedding_model = settings.get_embedding_model()
        assert isinstance(embedding_model, _embedding.SentenceTransformerEmbeddingModel)
        assert embedding_model.model_name == _embedding.EmbeddingModelName.MINILM

        other_embedding_model = settings.get_embedding_model()
        assert other_embedding_model is embedding_model

    def test_generates_embeddings_for_sequence_of_input_text(self):
        input_text = ["Some", "chunked", "content"]
        model = settings.get_embedding_model()

        embeddings = model.generate_embeddings(input_text=input_text)

        assert len(embeddings) == 3
        for embedding in embeddings:
            assert len(embedding) == model.embedding_size == 384
            assert sum(embedding) > 0

    def test_splits_text_on_tokens(self):
        input_text = "Some chunked content."

        config = _embedding.EmbeddingModelConfig(
            model_name=_embedding.EmbeddingModelName.MINILM,
            max_input_length=1,  # Equal to tokens per chunk.
            embedding_size=123,
        )
        model = _embedding.SentenceTransformerEmbeddingModel(config=config)

        chunks = model.split_text_on_tokens(input_text=input_text, chunk_overlap=0)

        assert len(chunks) > 1
