import typing

import loguru
import zenml

from llm_twin import settings, utils
from llm_twin.domain.etl import raw_documents
from llm_twin.orchestration.steps import context
from llm_twin.domain import authors

@zenml.step
def get_or_create_author(
    full_name: str,
    context: context.StepContext | None = None,
) -> typing.Annotated[authors.Author, "author"]:
    loguru.logger.info(f"Getting or creating author: {full_name}")
    db = settings.get_raw_document_database()

    name = utils.Name.from_full_name(full_name)
    author = authors.Author.get_or_create(
        db=db, first_name=name.first_name, last_name=name.last_name
    )

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="author",
        metadata=_get_metadata(author_full_name=full_name, author=author),
    )

    return author


def _get_metadata(
    *, author_full_name: str, author: authors.Author
) -> dict[str, typing.Any]:
    return {
        "query": {
            "author_full_name": author_full_name,
        },
        "retrieved": {
            "author_id": str(author.id),
            "first_name": str(author.first_name),
            "last_name": str(author.last_name),
        },
    }
