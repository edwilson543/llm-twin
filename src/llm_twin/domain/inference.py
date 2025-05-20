import abc
import pathlib

import transformers

from llm_twin.domain import models, training


class InferenceEngine(abc.ABC):
    @abc.abstractmethod
    def get_response(
        self, *, instruction: str, max_tokens: int, top_k: int | None = None
    ) -> tuple[str, str]:
        raise NotImplementedError


class LocalInferenceEngine(InferenceEngine, metaclass=models.SingletonMeta):
    """
    A fine-tuned model produced by the pipelines in this repo.
    """

    def __init__(self, *, load_model_from: pathlib.Path) -> None:
        self._model = transformers.AutoModelForCausalLM.from_pretrained(load_model_from)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(load_model_from)

    def get_response(
        self, *, instruction: str, max_tokens: int, top_k: int | None = None
    ) -> tuple[str, str]:
        # Make sure we use the same template the model was trained with.
        prompt = training.render_alpaca_template(instruction=instruction)
        prompt_tokens = self._tokenizer.encode(prompt, return_tensors="pt")
        response_tokens = self._model.generate(
            prompt_tokens, max_length=max_tokens, top_k=top_k
        )
        return prompt, self._tokenizer.decode(
            response_tokens[0], skip_special_tokens=True
        )


class SageMakerInferenceEngine(InferenceEngine):
    pass
