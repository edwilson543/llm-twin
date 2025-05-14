import pathlib
import typing

import loguru
import zenml

from llm_twin import config
from llm_twin.domain import evaluation, inference, training
from llm_twin.orchestration.steps import context


@zenml.step
def generate_completions_for_test_samples(
    author_id: str,
    load_model_from: str,
    max_tokens: int,
    top_k: int = 1,
    context: context.StepContext | None = None,
) -> typing.Annotated[list[evaluation.Completion], "completions"]:
    db = config.get_vector_database()
    data_loader = training.VectorDBDataLoader(db=db)
    dataset = data_loader.load_instruct_dataset(author_id=author_id)

    model = inference.LLMTwinModel(load_model_from=pathlib.Path(load_model_from))

    loguru.logger.info(f"Loaded model and tokenizer from {load_model_from}")

    completions: list[evaluation.Completion] = []
    for sample in dataset.test.samples:
        prompt, response = model.get_response(
            instruction=sample.instruction, max_tokens=max_tokens, top_k=top_k
        )

        completion = evaluation.Completion(prompt=prompt, response=response)
        completions.append(completion)

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="completions", metadata={"num_completions": len(completions)}
    )

    return completions
