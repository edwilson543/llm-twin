import abc
import dataclasses
import enum

import sentence_transformers
from langchain import text_splitter

from . import _singleton


class EmbeddingModelName(enum.StrEnum):
    MINILM = "all-MiniLM-L6-v2"
    FAKE = "fake"


@dataclasses.dataclass(frozen=True)
class EmbeddingModelConfig:
    model_name: EmbeddingModelName

    embedding_size: int
    max_input_length: int

    cache_dir: str | None
    device: str = "cpu"


@dataclasses.dataclass(frozen=True)
class UnableToEmbedText(Exception):
    model_name: EmbeddingModelName


class EmbeddingModel(abc.ABC, metaclass=_singleton.SingletonMeta):
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

    @abc.abstractmethod
    def split_text_on_tokens(self, *, input_text: str, chunk_overlap: int) -> list[str]:
        """
        Split the input text on tokens used by the embedding model.
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

    @property
    def chunk_overlap(self) -> int:
        return self._config.max_input_length


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, *, config: EmbeddingModelConfig) -> None:
        super().__init__(config=config)

        self._model = sentence_transformers.SentenceTransformer(
            model_name_or_path=self.model_name.value,
            device=config.device,
            cache_folder=config.cache_dir,
            config_kwargs={"from_tf": True},
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

    def split_text_on_tokens(self, *, input_text: str, chunk_overlap: int) -> list[str]:
        token_splitter = text_splitter.SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=self.max_input_length,
            model_name=self.model_name.value,
        )
        return token_splitter.split_text(input_text)
