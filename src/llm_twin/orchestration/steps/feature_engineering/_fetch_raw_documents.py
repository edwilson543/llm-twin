import itertools
import typing

import loguru
import zenml

from llm_twin import config, utils
from llm_twin.domain import authors
from llm_twin.domain.etl import raw_documents
from llm_twin.domain.storage import document as document_storage
from llm_twin.orchestration.steps import context
from llm_twin.orchestration.steps import types as step_types


class NoDocumentsFound(Exception):
    pass


@zenml.step
def fetch_raw_documents(
    author_full_name: str,
    context: context.StepContext | None = None,
) -> step_types.RawDocumentsOutputT:
    db = config.get_document_database()

    documents = _fetch_raw_documents_for_author(
        db=db, author_full_name=author_full_name
    )
    if not documents:
        raise NoDocumentsFound

    step_context = context or zenml.get_step_context()
    metadata = _get_metadata(documents=documents)
    step_context.add_output_metadata(output_name="raw_documents", metadata=metadata)
    return documents


def _fetch_raw_documents_for_author(
    *, db: document_storage.DocumentDatabase, author_full_name: str
) -> step_types.RawDocumentsOutputT:
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


def _get_metadata(*, documents: step_types.RawDocumentsOutputT) -> dict:
    metadata: dict[str, typing.Any] = {"num_documents": len(documents)}

    grouped_documents = itertools.groupby(
        documents, lambda document: document.get_collection_name()
    )
    metadata["num_documents_by_type"] = {
        collection: len(list(group)) for collection, group in grouped_documents
    }

    return metadata
