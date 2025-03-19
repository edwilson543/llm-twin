from llm_twin.domain import documents

from . import _base


class FakeCrawler(_base.Crawler):
    def _extract(
        self, *, db: documents.NoSQLDatabase, link: str, user: documents.UserDocument
    ) -> None:
        document = documents.ArticleDocument(
            platform="fake",
            content={"foo": "bar"},
            link=link,
            author_id=user.id,
            author_full_name=user.full_name,
        )
        document.save(db=db)
