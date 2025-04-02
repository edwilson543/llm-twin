import uuid

from llm_twin import settings
from llm_twin.domain.etl import raw_documents
from llm_twin.orchestration.pipelines import _etl
from llm_twin.domain import authors


def test_extracts_transforms_and_loads_data_for_author():
    links = [
        f"https://fake.com/edwilson543/{uuid.uuid4()}",
        f"https://fake.com/edwilson543/{uuid.uuid4()}",
    ]

    first_name = str(uuid.uuid4())
    last_name = str(uuid.uuid4())

    _etl.etl_author_data.entrypoint(
        author_full_name=f"{first_name} {last_name}", links=links
    )

    db = settings.get_raw_document_database()
    author = authors.Author.get(db=db, first_name=first_name, last_name=last_name)

    first_post = raw_documents.Article.get(db=db, link=links[0])
    assert first_post.platform == "fake"
    assert first_post.author_id == author.id

    second_post = raw_documents.Article.get(db=db, link=links[1])
    assert second_post.platform == "fake"
    assert second_post.author_id == author.id
