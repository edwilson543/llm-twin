from llm_twin.domain import raw_documents

from . import _base


class FakeCrawler(_base.Crawler):
    _document_class = raw_documents.Article

    def _extract(
        self,
        *,
        db: raw_documents.RawDocumentDatabase,
        link: str,
        author: raw_documents.Author,
    ) -> None:
        document = raw_documents.Article(
            platform="fake",
            content={"foo": "bar"},
            link=link,
            author_id=author.id,
            author_full_name=author.full_name,
        )
        document.save(db=db)
