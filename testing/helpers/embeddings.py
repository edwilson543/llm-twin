import contextlib
import typing
from unittest import mock

from llm_twin import settings
from llm_twin.domain import models


class FakeEmbeddingModel(models.EmbeddingModel):
    def generate_embeddings(self, *, input_text: list[str]) -> list[list[float]]:
        return [self.canned_embedding] * len(input_text)

    @property
    def canned_embedding(self) -> list[float]:
        return [1.0, 0.0, 0.0]


def get_fake_embedding_model_config() -> models.EmbeddingModelConfig:
    return models.EmbeddingModelConfig(
        model_name=models.EmbeddingModelName.FAKE,
        embedding_size=3,
        max_input_length=256,
    )


def get_fake_embedding_model() -> FakeEmbeddingModel:
    config = get_fake_embedding_model_config()
    return FakeEmbeddingModel(config=config)


@contextlib.contextmanager
def install_fake_embedding_model_config() -> typing.Generator[None, None, None]:
    """
    Helper that overrides the settings module to install the fake embedding model config.
    """
    fake_config = get_fake_embedding_model_config()
    with mock.patch.object(
        settings, "get_embedding_model_config", return_value=fake_config
    ):
        yield


@contextlib.contextmanager
def install_fake_embedding_model() -> typing.Generator[FakeEmbeddingModel, None, None]:
    """
    Helper that overrides the settings module to install a fake embedding model.
    """
    fake_embedding_model = get_fake_embedding_model()
    with mock.patch.object(
        settings, "get_embedding_model", return_value=fake_embedding_model
    ):
        yield fake_embedding_model
