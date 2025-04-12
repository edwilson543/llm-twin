from llm_twin.domain import models


class FakeEmbeddingModel(models.EmbeddingModel):
    def generate_embeddings(self, *, input_text: list[str]) -> list[list[float]]:
        return [self.canned_embedding] * len(input_text)

    @property
    def canned_embedding(self) -> list[float]:
        return [1.0, 0.0, 0.0]

    def split_text_on_tokens(self, *, input_text: str, chunk_overlap: int) -> list[str]:
        return [letter for letter in input_text]


def get_fake_embedding_model() -> FakeEmbeddingModel:
    config = models.EmbeddingModelConfig(
        model_name=models.EmbeddingModelName.FAKE,
        embedding_size=3,
        max_input_length=256,
        cache_dir=None,
    )
    return FakeEmbeddingModel(config=config)
