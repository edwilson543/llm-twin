import uuid

from llm_twin.domain import documents
from llm_twin.infrastructure import documents as documents_backend
from llm_twin.orchestration.pipelines import _etl_user_data


def test_extracts_transforms_and_loads_data_for_user():
    links = [
        f"https://fake.com/edwilson543/{uuid.uuid4()}",
        f"https://fake.com/edwilson543/{uuid.uuid4()}",
    ]

    first_name = str(uuid.uuid4())
    last_name = str(uuid.uuid4())

    _etl_user_data.etl_user_data.entrypoint(
        user_full_name=f"{first_name} {last_name}", links=links
    )

    db = documents_backend.get_nosql_database()
    author = documents.UserDocument.get(
        db=db, first_name=first_name, last_name=last_name
    )

    first_post = documents.ArticleDocument.get(db=db, link=links[0])
    assert first_post.platform == "fake"
    assert first_post.author_id == author.id

    second_post = documents.ArticleDocument.get(db=db, link=links[1])
    assert second_post.platform == "fake"
    assert second_post.author_id == author.id
