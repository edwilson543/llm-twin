import dataclasses
import typing
import uuid

from llm_twin.domain.storage import document as document_storage


@dataclasses.dataclass(frozen=True)
class InMemoryDocumentDatabase(document_storage.DocumentDatabase):
    documents: list[document_storage.Document] = dataclasses.field(default_factory=list)

    def find_one(
        self,
        *,
        document_class: type[document_storage.DocumentT],
        **filter_options: object,
    ) -> document_storage.DocumentT:
        for document in self.documents:
            if isinstance(
                document, document_class
            ) and self._document_matches_filter_options(document, **filter_options):
                return document
        raise document_storage.DocumentDoesNotExist()

    def find_many(
        self,
        *,
        document_class: type[document_storage.DocumentT],
        **filter_options: object,
    ) -> list[document_storage.DocumentT]:
        return [
            document
            for document in self.documents
            if isinstance(document, document_class)
            and self._document_matches_filter_options(document, **filter_options)
        ]

    def insert_one(self, *, document: document_storage.Document) -> None:
        self.documents.append(document)

    @property
    def dumped_documents(self) -> list[dict[str, typing.Any]]:
        return [document.model_dump() for document in self.documents]

    @staticmethod
    def _document_matches_filter_options(
        document: document_storage.Document, **filter_options: object
    ) -> bool:
        return all(
            getattr(document, filter_key, uuid.uuid4()) == filter_value
            for filter_key, filter_value in filter_options.items()
        )
