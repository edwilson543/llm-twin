import typing

from llm_twin.domain.feature_engineering import chunking


ChunkedDocumentsInputT = typing.Annotated[
    typing.Sequence[chunking.RepositoryChunk | chunking.ArticleChunk],
    "chunked_documents",
]
ChunkedDocumentsOutputT = typing.Annotated[
    list[chunking.Chunk],
    "chunked_documents",
]
