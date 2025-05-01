import pytest

from llm_twin.orchestration.steps.feature_engineering import _fetch_raw_documents
from testing.factories import documents as document_factories
from testing.helpers import config as config_helpers
from testing.helpers import storage as storage_helpers
from testing.helpers import zenml as zenml_helpers


def test_gets_all_raw_documents_for_specified_authors():
    author = document_factories.Author()
    author_article = document_factories.Article(author=author)
    author_other_article = document_factories.Article(author=author)
    author_repo = document_factories.Repository(author=author)

    other_author = document_factories.Author()
    other_author_article = document_factories.Article(author=other_author)
    other_author_repo = document_factories.Repository(author=other_author)

    all_documents = [
        author,
        author_article,
        author_other_article,
        author_repo,
        other_author,
        other_author_article,
        other_author_repo,
    ]
    db = storage_helpers.InMemoryDocumentDatabase(documents=all_documents)

    context = zenml_helpers.FakeContext()

    with config_helpers.install_in_memory_document_db(db=db):
        documents = _fetch_raw_documents.fetch_raw_documents.entrypoint(
            author_full_name=author.full_name, context=context
        )

    assert documents == [author_article, author_other_article, author_repo]
    assert context.output_metadata["raw_documents"] == {
        "num_documents": 3,
        "num_documents_by_type": {"articles": 2, "repositories": 1},
    }


def test_gets_no_raw_documents_for_author_that_does_not_exist():
    author = document_factories.Author()
    other_author_article = document_factories.Article()
    db = storage_helpers.InMemoryDocumentDatabase(
        documents=[author, other_author_article]
    )

    context = zenml_helpers.FakeContext()

    with (
        config_helpers.install_in_memory_document_db(db=db),
        pytest.raises(_fetch_raw_documents.NoDocumentsFound),
    ):
        _fetch_raw_documents.fetch_raw_documents.entrypoint(
            author_full_name=author.full_name, context=context
        )
