import typing

import zenml

from llm_twin import settings
from llm_twin.domain.feature_engineering import chunking, embedding
from llm_twin.orchestration.steps import context


@zenml.step
def embed_chunked_documents(
    chunked_documents: typing.Annotated[list[chunking.Chunk], "chunked_documents"],
    context: context.StepContext | None = None,
) -> typing.Annotated[list[embedding.EmbeddedChunk], "embedded_chunks"]:
    embedding_model = settings.get_embedding_model()
    dispatcher = embedding.EmbedderDispatcher(embedding_model=embedding_model)

    embedded_documents = dispatcher.embed_chunks(chunks=chunked_documents)

    db = settings.get_vector_database()
    db.bulk_insert(vectors=embedded_documents)

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="embedded_chunks",
        metadata={"num_documents": len(embedded_documents)},
    )

    return embedded_documents
