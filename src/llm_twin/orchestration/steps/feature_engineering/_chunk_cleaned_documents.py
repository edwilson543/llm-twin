import typing

import zenml

from llm_twin import settings
from llm_twin.domain.feature_engineering import chunking, cleaning
from llm_twin.orchestration.steps import context


@zenml.step
def chunk_cleaned_documents(
    cleaned_documents: typing.Annotated[
        list[cleaning.CleanedDocument], "cleaned_documents"
    ],
    context: context.StepContext | None = None,
) -> typing.Annotated[list[chunking.Chunk], "chunked_documents"]:
    embedding_model_config = settings.get_embedding_model_config()
    dispatcher = chunking.ChunkerDispatcher(
        embedding_model_config=embedding_model_config
    )

    chunked_documents: list[chunking.Chunk] = []

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
