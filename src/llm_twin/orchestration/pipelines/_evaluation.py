import zenml

from llm_twin.orchestration.steps import evaluation as evaluation_steps


@zenml.pipeline
def evaluate_model(
    author_id: str, load_model_from: str, max_tokens: int = 4096
) -> None:
    completions = evaluation_steps.generate_completions_for_test_samples(
        author_id=author_id, load_model_from=load_model_from, max_tokens=max_tokens
    )
    evaluations = evaluation_steps.evaluate_completions(completions)
    evaluation_steps.aggregate_evaluations(evaluations)


if __name__ == "__main__":
    base_model_name = "llamafactory/tiny-random-Llama-3"

    evaluate_model.with_options(enable_cache=True)(
        author_id="29eec0df-0405-45a9-a3ef-40c314e15789",
        load_model_from=base_model_name,
        max_tokens=50,
    )
