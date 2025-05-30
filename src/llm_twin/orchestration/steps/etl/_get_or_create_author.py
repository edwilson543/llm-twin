import typing

import loguru
import zenml

from llm_twin import config, utils
from llm_twin.domain import authors
from llm_twin.orchestration.steps import context


@zenml.step
def get_or_create_author(
    full_name: str,
    context: context.StepContext | None = None,
) -> typing.Annotated[authors.Author, "author"]:
    loguru.logger.info(f"Getting or creating author: {full_name}")
    db = config.get_document_database()

    name = utils.Name.from_full_name(full_name)
    author = db.get_or_create(
        document_class=authors.Author,
        first_name=name.first_name,
        last_name=name.last_name,
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
            "author_id": author.id,
            "first_name": author.first_name,
            "last_name": author.last_name,
        },
    }
