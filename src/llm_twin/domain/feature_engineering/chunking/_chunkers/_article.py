import re
import uuid

from llm_twin.domain.feature_engineering import cleaning
from llm_twin.domain.feature_engineering.chunking import _documents

from . import _base


class ArticleChunker(
    _base.DocumentChunker[cleaning.CleanedArticle, _documents.ArticleChunk]
):
    @property
    def min_length(self) -> int:
        return 1000

    @property
    def max_length(self) -> int:
        return 2000

    def _create_chunk(
        self, *, document: cleaning.CleanedArticle, chunk_id: str, content: str
    ) -> _documents.ArticleChunk:
        return _documents.ArticleChunk(
            id=str(uuid.UUID(chunk_id, version=4)),
            raw_document_id=document.raw_document_id,
            content=content,
            platform=document.platform,
            link=document.link,
            author_id=document.author_id,
            author_full_name=document.author_full_name,
            metadata={"min_length": self.min_length, "max_length": self.max_length},
        )

    def _chunk_content(self, content: str) -> list[str]:
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", content)

        extracts: list[str] = []
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) <= self.max_length:
                current_chunk += sentence + " "
            else:
                if len(current_chunk) >= self.min_length:
                    extracts.append(current_chunk.strip())
                current_chunk = sentence + " "

        if len(current_chunk) >= self.min_length:
            extracts.append(current_chunk.strip())

        return extracts
