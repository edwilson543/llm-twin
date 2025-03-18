import typing

import loguru
import zenml

from llm_twin.application import utils
from llm_twin.domain import documents
from llm_twin.infrastructure import documents as documents_backend
from llm_twin.orchestration.steps import context


@zenml.step
def get_or_create_user(
    user_full_name: str,
    context: context.StepContext | None = None,
    db: documents.NoSQLDatabase | None = None,
) -> typing.Annotated[documents.UserDocument, "user"]:
    loguru.logger.info(f"Getting or creating user: {user_full_name}")
    db = db or documents_backend.get_nosql_database()

    name = utils.split_user_full_name(user_full_name)
    user_document = documents.UserDocument.get_or_create(
        db=db, first_name=name.first_name, last_name=name.last_name
    )

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="user",
        metadata=_get_metadata(
            user_full_name=user_full_name, user_document=user_document
        ),
    )

    return user_document


def _get_metadata(
    *, user_full_name: str, user_document: documents.UserDocument
) -> dict[str, typing.Any]:
    return {
        "query": {
            "user_full_name": user_full_name,
        },
        "retrieved": {
            "user_id": str(user_document.id),
            "first_name": str(user_document.first_name),
            "last_name": str(user_document.last_name),
        },
    }
