from llm_twin.domain import raw_documents

from . import _base


class BrokenCrawler(_base.Crawler):
    _document_class = raw_documents.Article

    def _extract(
        self,
        *,
        db: raw_documents.RawDocumentDatabase,
        link: str,
        author: raw_documents.Author,
    ) -> None:
        raise _base.UnableToCrawlLink(link=link)
