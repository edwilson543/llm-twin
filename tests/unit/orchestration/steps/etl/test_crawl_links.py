from unittest import mock

from llm_twin.domain import documents
from llm_twin.infrastructure.documents import _in_memory as _in_memory_nosql_database
from llm_twin.orchestration.steps.etl import _crawl_links
from testing.factories import documents as document_factories
from testing.helpers import context as context_helpers


def test_crawls_links_for_fake_domain_successfully():
    user = document_factories.UserDocument()
    db = _in_memory_nosql_database.InMemoryNoSQLDatabase()
    context = context_helpers.FakeContext()

    links = [
        "https://fake.com/edwilson543/post-1/",
        "https://fake.com/edwilson543/post-2/",
    ]

    _crawl_links.crawl_links.entrypoint(user=user, links=links, context=context, db=db)

    assert db.data == {
        documents.Collection.ARTICLES: [
            {
                "_id": mock.ANY,
                "author_full_name": user.full_name,
                "author_id": str(user.id),
                "content": {"foo": "bar"},
                "link": links[0],
                "platform": "fake",
            },
            {
                "_id": mock.ANY,
                "author_full_name": user.full_name,
                "author_id": str(user.id),
                "content": {"foo": "bar"},
                "link": links[1],
                "platform": "fake",
            },
        ],
    }


def test_continues_after_failing_to_crawl_broken_link():
    user = document_factories.UserDocument()
    db = _in_memory_nosql_database.InMemoryNoSQLDatabase()
    context = context_helpers.FakeContext()

    links = [
        "https://broken.com/edwilson543/post-1/",
        "https://fake.com/edwilson543/post-2/",
    ]

    _crawl_links.crawl_links.entrypoint(user=user, links=links, context=context, db=db)

    assert db.data == {
        documents.Collection.ARTICLES: [
            {
                "_id": mock.ANY,
                "author_full_name": user.full_name,
                "author_id": str(user.id),
                "content": {"foo": "bar"},
                "link": links[1],
                "platform": "fake",
            },
        ],
    }
