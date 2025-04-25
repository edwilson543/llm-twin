import abc
import dataclasses

import loguru

from llm_twin.domain import authors
from llm_twin.domain.storage import document as document_storage


@dataclasses.dataclass(frozen=True)
class UnableToCrawlLink(Exception):
    link: str


class Crawler(abc.ABC):
    """
    Base class for crawling a webpage and extracting relevant information.
    """

    _document_class: type[document_storage.Document]

    def extract(
        self,
        *,
        db: document_storage.DocumentDatabase,
        link: str,
        author: authors.Author,
    ) -> None:
        try:
            db.find_one(document_class=self._document_class, link=link)
            loguru.logger.info(
                f"Skipping crawling {link} since document already extracted"
            )
        except document_storage.DocumentDoesNotExist:
            loguru.logger.info(f"Attempting to extract document(s) from {link}")
            document = self._extract(link=link, author=author)
            db.insert_one(document=document)
            loguru.logger.info(f"Successfully extracted document(s) from {link}")

    @abc.abstractmethod
    def _extract(
        self, *, link: str, author: authors.Author
    ) -> document_storage.Document:
        raise NotImplementedError
