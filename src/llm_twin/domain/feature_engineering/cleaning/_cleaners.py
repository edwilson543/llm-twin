import abc
import re
import typing

from llm_twin.domain.etl import raw_documents
from llm_twin.domain.storage import document as document_storage

from . import _documents


RawDocumentT = typing.TypeVar("RawDocumentT", bound=document_storage.Document)
CleanedDocumentT = typing.TypeVar("CleanedDocumentT", bound=_documents.CleanedDocument)


class DocumentCleaner(abc.ABC, typing.Generic[RawDocumentT, CleanedDocumentT]):
    """
    Base class for cleaning a particular document type.
    """

    @abc.abstractmethod
    def clean(self, *, document: RawDocumentT) -> CleanedDocumentT:
        raise NotImplementedError


class ArticleCleaner(DocumentCleaner[raw_documents.Article, _documents.CleanedArticle]):
    def clean(self, *, document: raw_documents.Article) -> _documents.CleanedArticle:
        valid_content = [content for content in document.content.values() if content]

        return _documents.CleanedArticle(
            id=document.id,
            raw_document_id=document.id,
            content=_clean_text(" #### ".join(valid_content)),
            platform=document.platform,
            link=document.link,
            author_id=document.author_id,
            author_full_name=document.author_full_name,
        )


class RepositoryCleaner(
    DocumentCleaner[raw_documents.Repository, _documents.CleanedRepository]
):
    def clean(
        self, *, document: raw_documents.Repository
    ) -> _documents.CleanedRepository:
        return _documents.CleanedRepository(
            id=document.id,
            raw_document_id=document.id,
            content=_clean_text(" #### ".join(document.content.values())),
            platform=document.platform,
            name=document.name,
            link=document.link,
            author_id=document.author_id,
            author_full_name=document.author_full_name,
        )


def _clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()
