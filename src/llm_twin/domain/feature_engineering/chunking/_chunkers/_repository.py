import uuid

from langchain import text_splitter

from llm_twin.domain.feature_engineering import cleaning
from llm_twin.domain.feature_engineering.chunking import _documents

from . import _base


class RepositoryChunker(
    _base.DocumentChunker[cleaning.CleanedRepository, _documents.RepositoryChunk]
):
    @property
    def chunk_size(self) -> int:
        return 500

    @property
    def chunk_overlap(self) -> int:
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
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
            },
        )

    def _chunk_content(self, content: str) -> list[str]:
        character_splitter = text_splitter.RecursiveCharacterTextSplitter(
            separators=["\n\n"], chunk_size=self.chunk_size, chunk_overlap=0
        )
        text_split_by_characters = character_splitter.split_text(content)

        token_splitter = text_splitter.SentenceTransformersTokenTextSplitter(
            chunk_overlap=self.chunk_overlap,
            tokens_per_chunk=self._embedding_model_config.max_input_length,
            model_name=self._embedding_model_config.model_name.value,
        )
        chunks_by_tokens = []
        for section in text_split_by_characters:
            chunks_by_tokens.extend(token_splitter.split_text(section))

        return chunks_by_tokens
