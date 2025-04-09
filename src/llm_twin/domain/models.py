import abc
import dataclasses
import enum
import pathlib

import sentence_transformers

from llm_twin import utils


class EmbeddingModelName(enum.Enum):
    MINILM = "sentence-transformers/all-MiniLM-L6-v2"
    FAKE = "fake"


@dataclasses.dataclass(frozen=True)
class EmbeddingModelConfig:
    model_name: EmbeddingModelName

    embedding_size: int
    max_input_length: int

    device: str = "cpu"
    cache_dir: pathlib.Path | None = None


@dataclasses.dataclass(frozen=True)
class UnableToEmbedText(Exception):
    model_name: EmbeddingModelName


class EmbeddingModel(abc.ABC, metaclass=utils.SingletonMeta):
    """
    Base class for an embedding model.
    """

    def __init__(self, *, config: EmbeddingModelConfig) -> None:
        self._config = config

    @abc.abstractmethod
    def generate_embeddings(self, *, input_text: list[str]) -> list[list[float]]:
        """
        Generates embeddings for the input text using the pre-trained transformer model.

        :raises UnableToEmbedText: If the operation fails for some reason.
        """
        raise NotImplementedError

    @property
    def model_name(self) -> EmbeddingModelName:
        return self._config.model_name

    @property
    def embedding_size(self) -> int:
        return self._config.embedding_size

    @property
    def max_input_length(self) -> int:
        return self._config.max_input_length


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, *, config: EmbeddingModelConfig) -> None:
        super().__init__(config=config)

        self._model = sentence_transformers.SentenceTransformer(
            model_name_or_path=self.model_name.value,
            device=config.device,
            cache_folder=str(config.cache_dir) if config.cache_dir else None,
        )
        self._model.eval()

    def generate_embeddings(self, *, input_text: list[str]) -> list[list[float]]:
        """
        Generates embeddings for the input text using the pre-trained transformer model.
        """
        try:
            embeddings = self._model.encode(input_text)
        except Exception as exc:
            raise UnableToEmbedText(model_name=self.model_name) from exc

        return [embedding.tolist() for embedding in embeddings]
