import zenml

from llm_twin import settings
from llm_twin.domain.feature_engineering import chunking
from llm_twin.orchestration.steps import context

from . import _types


@zenml.step
def chunk_cleaned_documents(
    cleaned_documents: _types.CleanedDocumentsInputT,
    context: context.StepContext | None = None,
) -> _types.ChunkedDocumentsOutputT:
    embedding_model_config = settings.get_embedding_model_config()
    dispatcher = chunking.ChunkerDispatcher(
        embedding_model_config=embedding_model_config
    )

    chunked_documents: _types.ChunkedDocumentsOutputT = []

    for document in cleaned_documents:
        chunker = dispatcher.get_chunker(document=document)
        chunk = chunker.chunk(document=document)
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
