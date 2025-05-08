import typing

import zenml

from llm_twin.domain import evaluation
from llm_twin import config


@zenml.step
def evaluate_completions(
    completions: typing.Annotated[list[evaluation.Completion], "completions"],
) -> typing.Annotated[list[evaluation.Evaluation], "evaluations"]:
    language_model = config.get_language_model()
    return [completion.evaluate(language_model=language_model) for completion in completions]
