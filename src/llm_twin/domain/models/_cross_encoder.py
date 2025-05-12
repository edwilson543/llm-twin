import abc
import enum

from sentence_transformers import cross_encoder

from . import _singleton


class CrossEncoderModelName(enum.Enum):
    TINY_BERT = "cross-encoder/ms-marco-TinyBERT-L2-v2"
    MINILM = "cross-encoder/ms-marco-MiniLM-L4-v2"


class CrossEncoderModel(abc.ABC, metaclass=_singleton.SingletonMeta):
    """
    Third-party LLM used to score the similarity of two passages of text.
    """

    @abc.abstractmethod
    def predict(self, *, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Generate a score between 0 and 1 representing how similar the text in each pair is.
        """
        raise NotImplementedError


class SentenceTransformerCrossEncoder(CrossEncoderModel):
    def __init__(
        self, *, model_name: CrossEncoderModelName, device: str = "cpu"
    ) -> None:
        self._model = cross_encoder.CrossEncoder(model_name.value, device=device)
        self._model.eval()

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self._model.predict(sentences=pairs).tolist()
