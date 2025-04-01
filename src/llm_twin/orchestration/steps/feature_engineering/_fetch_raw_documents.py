import typing

import loguru
import zenml

from llm_twin import settings, utils
from llm_twin.domain import raw_documents
from llm_twin.orchestration.steps import context


@zenml.step
def fetch_raw_documents(
    author_full_names: list[str],
    context: context.StepContext | None = None,
) -> typing.Annotated[list[raw_documents.ExtractedDocument], "raw_documents"]:
    db = settings.get_raw_document_database()

    documents: list[raw_documents.ExtractedDocument] = []

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
    *, db: raw_documents.RawDocumentDatabase, author_full_name: str
) -> list[raw_documents.ExtractedDocument]:
    loguru.logger.info(f"Fetching raw documents for '{author_full_name}'.")
    name = utils.split_user_full_name(author_full_name)
    raw_documents.UserDocument.get_or_create(
        db=db, first_name=name.first_name, last_name=name.last_name
    )

    # articles = []
    # repositories = []

    return []


def _get_metadata(*, documents: list[raw_documents.ExtractedDocument]) -> dict:
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
