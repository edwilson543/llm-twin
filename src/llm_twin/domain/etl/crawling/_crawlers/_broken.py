from llm_twin.domain.etl import raw_documents
from llm_twin.domain.storage import document as document_storage

from . import _base


class BrokenCrawler(_base.Crawler):
    _document_class = raw_documents.Article

    def _extract(
        self,
        *,
        db: document_storage.RawDocumentDatabase,
        link: str,
        author: raw_documents.Author,
    ) -> None:
        raise _base.UnableToCrawlLink(link=link)
