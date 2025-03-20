import uuid

from llm_twin.domain import documents
from llm_twin.orchestration.pipelines import _etl_user_data
from testing.helpers import infrastructure as infrastructure_helpers


def test_extracts_transforms_and_loads_data_for_user(db: documents.NoSQLDatabase):
    links = [
        f"https://fake.com/edwilson543/{uuid.uuid4()}",
        f"https://fake.com/edwilson543/{uuid.uuid4()}",
    ]

    first_name = str(uuid.uuid4())
    last_name = str(uuid.uuid4())

    with infrastructure_helpers.install_nosql_db(db=db):
        _etl_user_data.etl_user_data.entrypoint(
            user_full_name=f"{first_name} {last_name}", links=links
        )

    author = documents.UserDocument.get(
        db=db, first_name=first_name, last_name=last_name
    )

    first_post = documents.ArticleDocument.get(db=db, link=links[0])
    assert first_post.platform == "fake"
    assert first_post.author_id == author.id

    second_post = documents.ArticleDocument.get(db=db, link=links[1])
    assert second_post.platform == "fake"
    assert second_post.author_id == author.id
