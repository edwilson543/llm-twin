import zenml

from llm_twin.orchestration.steps import evaluation as evaluation_steps


@zenml.pipeline
def evaluate_model(load_model_from: str) -> None:
    answers = evaluation_steps.generate_answers(load_model_from)
    evaluation_steps.evaluate_answers(answers)
