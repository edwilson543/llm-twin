from llm_twin.domain import authors
from llm_twin.domain.etl import raw_documents
from llm_twin.domain.storage import document as document_storage

from . import _base


class FakeCrawler(_base.Crawler):
    _document_class = raw_documents.Article

    def _extract(
        self,
        *,
        db: document_storage.DocumentDatabase,
        link: str,
        author: authors.Author,
    ) -> None:
        document = raw_documents.Article(
            platform="fake",
            content={"foo": "bar"},
            link=link,
            author_id=author.id,
            author_full_name=author.full_name,
        )
        db.insert_one(document=document)
