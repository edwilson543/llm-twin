import zenml

from llm_twin import config
from llm_twin.domain.feature_engineering import embedding
from llm_twin.orchestration.steps import context

from . import _types


@zenml.step
def embed_chunked_documents(
    chunked_documents: _types.ChunkedDocumentsInputT,
    batch_size: int = 1000,
    context: context.StepContext | None = None,
) -> _types.EmbeddedDocumentsOutputT:
    embedding_model = config.get_embedding_model()
    dispatcher = embedding.EmbedderDispatcher(embedding_model=embedding_model)
    db = config.get_vector_database()

    embedded_documents: _types.EmbeddedDocumentsOutputT = []
    batch_indices = range(0, len(chunked_documents), batch_size)

    for batch_index in batch_indices:
        chunk_batch = chunked_documents[batch_index : batch_index + batch_size]
        embedded_batch = dispatcher.embed_chunks(chunks=chunk_batch)

        db.bulk_insert(vectors=embedded_batch)
        embedded_documents.extend(embedded_batch)

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="embedded_chunks",
        metadata={
            "num_documents": len(chunked_documents),
            "num_batches": len(batch_indices),
            "batch_size": batch_size,
        },
    )

    return embedded_documents
