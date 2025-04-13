from unittest import mock

from llm_twin.orchestration.steps.etl import _crawl_links
from testing.factories import documents as document_factories
from testing.helpers import config as config_helpers
from testing.helpers import context as context_helpers


def test_crawls_links_for_fake_domain_successfully():
    author = document_factories.Author()
    context = context_helpers.FakeContext()

    links = [
        "https://fake.com/edwilson543/post-1/",
        "https://fake.com/edwilson543/post-2/",
    ]

    with config_helpers.install_in_memory_document_db() as db:
        _crawl_links.crawl_links.entrypoint(author=author, links=links, context=context)

    assert db.dumped_documents == [
        {
            "id": mock.ANY,
            "author_full_name": author.full_name,
            "author_id": author.id,
            "content": {"foo": "bar"},
            "link": links[0],
            "platform": "fake",
        },
        {
            "id": mock.ANY,
            "author_full_name": author.full_name,
            "author_id": author.id,
            "content": {"foo": "bar"},
            "link": links[1],
            "platform": "fake",
        },
    ]


def test_continues_after_failing_to_crawl_broken_link():
    author = document_factories.Author()
    context = context_helpers.FakeContext()

    links = [
        "https://broken.com/edwilson543/post-1/",
        "https://fake.com/edwilson543/post-2/",
    ]

    with config_helpers.install_in_memory_document_db() as db:
        _crawl_links.crawl_links.entrypoint(author=author, links=links, context=context)

    assert db.dumped_documents == [
        {
            "id": mock.ANY,
            "author_full_name": author.full_name,
            "author_id": author.id,
            "content": {"foo": "bar"},
            "link": links[1],
            "platform": "fake",
        },
    ]
