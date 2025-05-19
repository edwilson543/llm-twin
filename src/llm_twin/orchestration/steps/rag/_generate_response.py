import typing

import zenml

from llm_twin import config
from llm_twin.domain import rag
from llm_twin.orchestration.steps import context
from llm_twin.orchestration.steps import types as step_types


@zenml.step
def generate_response(
    query: str,
    documents: step_types.EmbeddedDocumentsInputT,
    max_tokens: int,
    context: context.StepContext | None = None,
) -> typing.Annotated[str, "response"]:
    instruction = rag.augment_query(query=query, documents=documents)
    model = config.get_inference_engine()

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="response", metadata={"prompt": instruction}
    )

    _, response = model.get_response(instruction=instruction, max_tokens=max_tokens)

    return response
