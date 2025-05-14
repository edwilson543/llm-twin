import zenml

from llm_twin import config
from llm_twin.domain import rag
from llm_twin.orchestration.steps import context
from llm_twin.orchestration.steps import types as step_types


@zenml.step
def retrieve_documents(
    query: str,
    context: context.StepContext | None = None,
) -> step_types.EmbeddedDocumentsOutputT:
    retrieval_config = config.get_retrieval_config()
    documents = rag.retrieve_relevant_documents(query=query, config=retrieval_config)

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="embedded_chunks",
        metadata={
            "document_ids": [document.id for document in documents],
        },
    )

    return documents
