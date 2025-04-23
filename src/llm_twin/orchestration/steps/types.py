"""
Module containing input / output types to each step function.

This is a workaround for ZenML and Pydantic's use of type annotations at runtime.

Notes:
- For input types, sequences of union types are used. This is because when
  pipelines are called as entrypoints, ZenML validates the input parameters
  using the input type annotation. If the input type is an abstract base class,
  ZenML will try and instantiate this base class using each subclass' data.
- For output type, lists of base types are used. This is for consistency
  with the domain code.

There's probably a better way.
"""

import typing

from llm_twin.domain.etl import raw_documents
from llm_twin.domain.feature_engineering import chunking, cleaning, embedding


RawDocumentsInputT = typing.Annotated[
    typing.Sequence[raw_documents.Article | raw_documents.Repository],
    "raw_documents",
]
RawDocumentsOutputT = typing.Annotated[
    list[raw_documents.RawDocument],
    "raw_documents",
]

CleanedDocumentsInputT = typing.Annotated[
    typing.Sequence[cleaning.CleanedArticle | cleaning.CleanedRepository],
    "cleaned_documents",
]
CleanedDocumentsOutputT = typing.Annotated[
    list[cleaning.CleanedDocument],
    "cleaned_documents",
]

ChunkedDocumentsInputT = typing.Annotated[
    typing.Sequence[chunking.RepositoryChunk | chunking.ArticleChunk],
    "chunked_documents",
]
ChunkedDocumentsOutputT = typing.Annotated[
    list[chunking.Chunk],
    "chunked_documents",
]

EmbeddedDocumentsInputT = typing.Annotated[
    typing.Sequence[embedding.EmbeddedRepositoryChunk | embedding.EmbeddedArticleChunk],
    "embedded_chunks",
]
EmbeddedDocumentsOutputT = typing.Annotated[
    list[embedding.EmbeddedChunk],
    "embedded_chunks",
]
