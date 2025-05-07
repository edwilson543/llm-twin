import typing

import zenml

from llm_twin.domain import evaluation


@zenml.step
def evaluate_answers(
    answers: typing.Annotated[list[evaluation.InstructionAnswerPair], "answers"],
) -> typing.Annotated[list[evaluation.Evaluation], "answers"]:
    # TODO
    raise NotImplementedError
