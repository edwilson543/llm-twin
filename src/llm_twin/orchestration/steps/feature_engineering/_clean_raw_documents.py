import typing

import zenml

from llm_twin import settings
from llm_twin.domain.feature_engineering import cleaning
from llm_twin.orchestration.steps import context

from . import _types


@zenml.step
def clean_raw_documents(
    raw_documents: _types.RawDocumentsInputT,
    context: context.StepContext | None = None,
) -> _types.CleanedDocumentsOutputT:
    db = settings.get_vector_database()
    dispatcher = cleaning.CleanerDispatcher()

    cleaned_documents: _types.CleanedDocumentsOutputT = []

    for document in raw_documents:
        cleaned_document = dispatcher.clean_document(document=document)
        cleaned_documents.append(cleaned_document)

    db.bulk_insert(vectors=cleaned_documents)

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="cleaned_documents", metadata=_get_metadata(cleaned_documents)
    )

    return cleaned_documents


def _get_metadata(cleaned_documents: _types.CleanedDocumentsOutputT) -> dict:
    metadata: dict[str, typing.Any] = {"num_documents": len(cleaned_documents)}

    for document in cleaned_documents:
        if (category := document.category().value) not in metadata:
            metadata[category] = {"num_documents": 0, "authors": []}

        metadata[category]["num_documents"] += 1
        metadata[category]["authors"].append(document.author_full_name)

    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"]))

    return metadata
