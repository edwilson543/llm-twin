import zenml

from llm_twin import config
from llm_twin.domain.feature_engineering import chunking
from llm_twin.orchestration.steps import context
from llm_twin.orchestration.steps import types as step_types


@zenml.step
def fetch_chunked_documents(
    batch_size: int = 1000,
    context: context.StepContext | None = None,
) -> step_types.ChunkedDocumentsOutputT:
    chunked_documents = [
        *_fetch_chunked_documents(
            vector_class=chunking.ArticleChunk, batch_size=batch_size
        ),
        *_fetch_chunked_documents(
            vector_class=chunking.RepositoryChunk, batch_size=batch_size
        ),
    ]

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="chunked_documents",
        metadata={"num_chunks": len(chunked_documents)},
    )

    return chunked_documents


def _fetch_chunked_documents(
    vector_class: type[chunking.Chunk], batch_size: int
) -> step_types.ChunkedDocumentsOutputT:
    db = config.get_vector_database()

    chunked_documents: step_types.ChunkedDocumentsOutputT = []

    while True:
        next_offset = None

        extra_documents, next_offset = db.bulk_find(
            vector_class=vector_class, limit=batch_size, offset=next_offset
        )
        chunked_documents.extend(extra_documents)

        if next_offset is None:
            break

    return chunked_documents
