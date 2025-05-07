import typing

import zenml

from llm_twin.domain import evaluation


@zenml.step
def generate_answers(
    load_model_from: str,
) -> typing.Annotated[list[evaluation.InstructionAnswerPair], "answers"]:
    raise NotImplementedError
