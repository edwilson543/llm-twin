import typing

import zenml

from llm_twin.domain import evaluation
from llm_twin.orchestration.steps import context


@zenml.step
def aggregate_evaluations(
    evaluations: typing.Annotated[list[evaluation.Evaluation], "evaluations"],
    context: context.StepContext | None = None,
) -> typing.Annotated[list[evaluation.EvaluationAggregate], "evaluation_aggregates"]:
    mean = evaluation.EvaluationAggregate.mean(evaluations=evaluations)

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="evaluation_aggregates",
        metadata={
            "num_evaluations": len(evaluations),
            "mean": {"accuracy": mean.accuracy, "style": mean.style},
        },
    )

    return [mean]
