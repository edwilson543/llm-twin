import enum
import typing

import pydantic

from llm_twin.domain import models


class EvaluationCriteria(pydantic.BaseModel):
    score: int = pydantic.Field(ge=1, le=3)
    analysis: str


class Evaluation(pydantic.BaseModel):
    accuracy: EvaluationCriteria
    style: EvaluationCriteria


class Aggregate(enum.Enum):
    MEAN = "mean"


class EvaluationAggregate(pydantic.BaseModel):
    aggregate: Aggregate
    accuracy: float
    style: float

    @classmethod
    def mean(cls, evaluations: typing.Sequence[Evaluation]) -> typing.Self:
        def _mean(criteria: str) -> float:
            total = sum(
                getattr(evaluation, criteria).score for evaluation in evaluations
            )
            return total / len(evaluations)

        return cls(
            aggregate=Aggregate.MEAN,
            accuracy=_mean("accuracy"),
            style=_mean("style"),
        )


class Completion(pydantic.BaseModel):
    prompt: str
    response: str

    def evaluate(self, *, language_model: models.LanguageModel) -> Evaluation:
        """
        'LLM as a judge' implementation to evaluate this completion.
        """
        system_prompt = models.Message.system(
            content="You are a helpful assistant who evaluates answers based on accuracy and style."
        )
        user_prompt = models.Message.user(content=self._get_evaluation_prompt())

        return language_model.get_response(
            messages=[system_prompt, user_prompt], response_format=Evaluation
        )

    def _get_evaluation_prompt(self) -> str:
        return f"""Please evaluate the quality of a given answer to an instruction based on two criteria:
    1. Accuracy: How factually correct is the information presented in the answer? You are a technical expert in this topic.
    2. Style: Is the tone and writing style appropriate for a blog post or social media content? It should use simple but technical words and avoid formal or academic language.

    Accuracy scale:
    1 (Poor): Contains factual errors or misleading information
    2 (Good): Mostly accurate with minor errors or omissions
    3 (Excellent): Highly accurate and comprehensive

    Style scale:
    1 (Poor): Too formal, uses some overly complex words
    2 (Good): Good balance of technical content and accessibility, but still uses formal words and expressions
    3 (Excellent): Perfectly accessible language for blog/social media, uses simple but precise technical terms when necessary

    Example of bad style: The Llama2 7B model constitutes a noteworthy progression in the field of artificial intelligence, serving as the successor to its predecessor, the original Llama architecture.
    Example of excellent style: Llama2 7B outperforms the original Llama model across multiple benchmarks.

    Instruction: {self.prompt}

    Answer: {self.response}"""
