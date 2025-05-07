import typing

import pydantic


class InstructionAnswerPair(pydantic.BaseModel):
    instruction: str
    answer: str


class EvaluationCriteria(pydantic.BaseModel):
    score: int = pydantic.Field(ge=1, le=3)
    analysis: str


class Evaluation(pydantic.BaseModel):
    accuracy: EvaluationCriteria
    style: EvaluationCriteria


class EvaluationSummary(pydantic.BaseModel):
    accuracy: float
    style: float

    @classmethod
    def mean(cls, evaluations: typing.Sequence[Evaluation]) -> typing.Self:
        def _mean(criteria: str) -> float:
            total = sum(
                getattr(evaluation, criteria).score for evaluation in evaluations
            )
            return total / len(evaluations)

        return cls(accuracy=_mean("accuracy"), style=_mean("style"))
