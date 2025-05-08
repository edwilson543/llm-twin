import zenml

from llm_twin.orchestration.steps import evaluation as evaluation_steps


@zenml.pipeline
def evaluate_model(load_model_from: str) -> None:
    completions = evaluation_steps.generate_completions_for_test_samples(
        load_model_from
    )
    evaluations = evaluation_steps.evaluate_completions(completions)
    evaluation_steps.aggregate_evaluations(evaluations)
