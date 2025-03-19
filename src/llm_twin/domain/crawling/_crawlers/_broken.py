from llm_twin.domain import documents

from . import _base


class BrokenCrawler(_base.Crawler):
    def _extract(
        self, *, db: documents.NoSQLDatabase, link: str, user: documents.UserDocument
    ) -> None:
        raise _base.UnableToCrawlLink(link=link)
