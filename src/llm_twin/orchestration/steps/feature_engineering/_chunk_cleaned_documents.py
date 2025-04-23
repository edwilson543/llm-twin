import zenml

from llm_twin import config
from llm_twin.domain.feature_engineering import chunking
from llm_twin.orchestration.steps import context
from llm_twin.orchestration.steps import types as step_types


@zenml.step
def chunk_cleaned_documents(
    cleaned_documents: step_types.CleanedDocumentsInputT,
    context: context.StepContext | None = None,
) -> step_types.ChunkedDocumentsOutputT:
    embedding_model = config.get_embedding_model()
    dispatcher = chunking.ChunkerDispatcher(embedding_model=embedding_model)

    chunked_documents: step_types.ChunkedDocumentsOutputT = []

    for document in cleaned_documents:
        chunk = dispatcher.split_document_into_chunks(document=document)
        chunked_documents.extend(chunk)

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="chunked_documents",
        metadata={
            "num_documents": len(cleaned_documents),
            "num_chunks": len(chunked_documents),
        },
    )

    return chunked_documents
