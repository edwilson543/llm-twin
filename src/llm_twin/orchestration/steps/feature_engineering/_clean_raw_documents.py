import typing

import zenml

from llm_twin import settings
from llm_twin.domain.etl import raw_documents
from llm_twin.domain.feature_engineering import cleaning
from llm_twin.orchestration.steps import context


@zenml.step
def clean_raw_documents(
    raw_documents: typing.Annotated[list[raw_documents.Article | raw_documents.Repository], "raw_documents"],
    context: context.StepContext | None = None,
) -> typing.Annotated[list[cleaning.CleanedArticle | cleaning.CleanedRepository], "cleaned_documents"]:
    db = settings.get_vector_database()
    dispatcher = cleaning.CleanerDispatcher()

    cleaned_documents: list[cleaning.CleanedDocument] = []

    for document in raw_documents:
        cleaner = dispatcher.get_cleaner(document=document)
        cleaned_document = cleaner.clean(document=document)
        cleaned_documents.append(cleaned_document)

    db.bulk_insert(vectors=cleaned_documents)

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="cleaned_documents", metadata=_get_metadata(cleaned_documents)
    )

    return cleaned_documents


def _get_metadata(cleaned_documents: list[cleaning.CleanedDocument]) -> dict:
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
