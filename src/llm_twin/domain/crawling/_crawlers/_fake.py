from llm_twin.domain import raw_documents

from . import _base


class FakeCrawler(_base.Crawler):
    _document_class = raw_documents.ArticleDocument

    def _extract(
        self,
        *,
        db: raw_documents.NoSQLDatabase,
        link: str,
        user: raw_documents.UserDocument,
    ) -> None:
        document = raw_documents.ArticleDocument(
            platform="fake",
            content={"foo": "bar"},
            link=link,
            author_id=user.id,
            author_full_name=user.full_name,
        )
        document.save(db=db)
