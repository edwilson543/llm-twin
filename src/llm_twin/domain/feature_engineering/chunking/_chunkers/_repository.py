import uuid

from langchain import text_splitter

from llm_twin.domain.feature_engineering import cleaning
from llm_twin.domain.feature_engineering.chunking import _documents

from . import _base


class RepositoryChunker(
    _base.DocumentChunker[cleaning.CleanedRepository, _documents.RepositoryChunk]
):
    @property
    def _chunk_size(self) -> int:
        return 500

    @property
    def _chunk_overlap(self) -> int:
        return 50

    def _create_chunk(
        self, *, document: cleaning.CleanedRepository, chunk_id: str, content: str
    ) -> _documents.RepositoryChunk:
        return _documents.RepositoryChunk(
            id=str(uuid.UUID(chunk_id, version=4)),
            raw_document_id=document.raw_document_id,
            content=content,
            platform=document.platform,
            name=document.name,
            link=document.link,
            author_id=document.author_id,
            author_full_name=document.author_full_name,
            metadata={
                "chunk_size": self._chunk_size,
                "chunk_overlap": self._chunk_overlap,
            },
        )

    def _chunk_content(self, content: str) -> list[str]:
        character_splitter = text_splitter.RecursiveCharacterTextSplitter(
            separators=["\n\n"], chunk_size=self._chunk_size, chunk_overlap=0
        )
        text_split_by_characters = character_splitter.split_text(content)

        chunks_by_tokens = []
        for section in text_split_by_characters:
            chunks = self._embedding_model.split_text_on_tokens(
                input_text=section, chunk_overlap=self._chunk_overlap
            )
            chunks_by_tokens.extend(chunks)

        return chunks_by_tokens
