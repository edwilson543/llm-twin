import contextlib
import dataclasses

from llm_twin import settings
from llm_twin.domain.feature_engineering import embedding


@dataclasses.dataclass(frozen=True, kw_only=True)
class FakeEmbeddingModelConfig(embedding.EmbeddingModelConfig):
    model_name: embedding.EmbeddingModelName = embedding.EmbeddingModelName.FAKE
    embedding_size: int = 3
    max_input_length: int = 256


class FakeEmbeddingModel(embedding.EmbeddingModel):
    def generate_embeddings(self, *, input_text: list[str]) -> list[list[float]]:
        return [self.canned_embedding] * len(input_text)

    @property
    def canned_embedding(self) -> list[float]:
        return [1.0, 0.0, 0.0]


def get_fake_embedding_model() -> FakeEmbeddingModel:
    config = FakeEmbeddingModelConfig()
    return FakeEmbeddingModel(config=config)
