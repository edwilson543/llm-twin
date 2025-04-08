import abc
import hashlib
import typing

from llm_twin.domain.feature_engineering import cleaning
from llm_twin.domain.feature_engineering.chunking import _documents


CleanedDocumentT = typing.TypeVar("CleanedDocumentT", bound=cleaning.CleanedDocument)
ChunkT = typing.TypeVar("ChunkT", bound=_documents.Chunk)


class DocumentChunker(abc.ABC, typing.Generic[CleanedDocumentT, ChunkT]):
    """
    Base class for chunking a particular document type.
    """

    def chunk(self, *, document: CleanedDocumentT) -> list[ChunkT]:
        chunks: list[ChunkT] = []

        for content in self._chunk_content(content=document.content):
            chunk_id = hashlib.md5(content.encode()).hexdigest()
            chunk = self._create_chunk(
                document=document, chunk_id=chunk_id, content=content
            )
            chunks.append(chunk)

        return chunks

    @abc.abstractmethod
    def _create_chunk(
        self, *, document: CleanedDocumentT, chunk_id: str, content: str
    ) -> ChunkT:
        raise NotImplementedError

    @abc.abstractmethod
    def _chunk_content(self, *, content: str) -> list[str]:
        raise NotImplementedError
