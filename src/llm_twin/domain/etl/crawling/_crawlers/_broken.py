from llm_twin.domain import authors
from llm_twin.domain.etl import raw_documents
from llm_twin.domain.storage import document as document_storage

from . import _base


class BrokenCrawler(_base.Crawler):
    _document_class = raw_documents.Article

    def _extract(
        self, *, link: str, author: authors.Author
    ) -> document_storage.Document:
        raise _base.UnableToCrawlLink(link=link)
