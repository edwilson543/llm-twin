import typing

import zenml

from llm_twin.domain import evaluation


@zenml.step
def evaluate_completions(
    completions: typing.Annotated[list[evaluation.Completion], "completions"],
) -> typing.Annotated[evaluation.Evaluation, "evaluation_summary"]:
    # TODO.
    raise NotImplementedError
