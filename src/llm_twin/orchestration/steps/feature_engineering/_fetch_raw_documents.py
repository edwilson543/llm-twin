import typing

import loguru
import zenml

from llm_twin import utils
from llm_twin.config import settings
from llm_twin.domain import authors
from llm_twin.domain.etl import raw_documents
from llm_twin.domain.storage import document as document_storage
from llm_twin.orchestration.steps import context

from . import _types


@zenml.step
def fetch_raw_documents(
    author_full_names: list[str],
    context: context.StepContext | None = None,
) -> _types.RawDocumentsOutputT:
    db = settings.get_document_database()

    documents = []

    for author_full_name in author_full_names:
        author_documents = _fetch_raw_documents_for_author(
            db=db, author_full_name=author_full_name
        )
        documents.extend(author_documents)

    step_context = context or zenml.get_step_context()
    metadata = _get_metadata(documents=documents)
    step_context.add_output_metadata(output_name="raw_documents", metadata=metadata)
    return documents


def _fetch_raw_documents_for_author(
    *, db: document_storage.DocumentDatabase, author_full_name: str
) -> _types.RawDocumentsOutputT:
    loguru.logger.info(f"Fetching raw documents for '{author_full_name}'.")
    name = utils.Name.from_full_name(author_full_name)
    author = db.get_or_create(
        document_class=authors.Author,
        first_name=name.first_name,
        last_name=name.last_name,
    )

    articles = db.find_many(document_class=raw_documents.Article, author_id=author.id)
    repositories = db.find_many(
        document_class=raw_documents.Repository, author_id=author.id
    )

    return [*articles, *repositories]


def _get_metadata(*, documents: _types.RawDocumentsOutputT) -> dict:
    metadata: dict[str, typing.Any] = {"num_documents": len(documents)}

    for document in documents:
        if (collection := document.get_collection_name().value) not in metadata:
            metadata[collection] = {"num_documents": 0, "authors": []}

        metadata[collection]["num_documents"] += 1
        metadata[collection]["authors"].append(document.author_full_name)

    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"]))

    return metadata
