from llm_twin.domain import raw_documents

from . import _base


class BrokenCrawler(_base.Crawler):
    _document_class = raw_documents.ArticleDocument

    def _extract(
        self,
        *,
        db: raw_documents.RawDocumentDatabase,
        link: str,
        user: raw_documents.UserDocument,
    ) -> None:
        raise _base.UnableToCrawlLink(link=link)
