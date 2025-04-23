from llm_twin.orchestration.steps.feature_engineering import _fetch_raw_documents
from testing.factories import documents as document_factories
from testing.helpers import config as config_helpers
from testing.helpers import storage as storage_helpers
from testing.helpers import zenml as zenml_helpers


def test_gets_all_raw_documents_for_specified_authors():
    author = document_factories.Author()
    other_author = document_factories.Author()

    author_article = document_factories.Article(author=author)
    author_other_article = document_factories.Article(author=author)
    other_author_article = document_factories.Article(author=other_author)
    random_author_article = document_factories.Article()

    author_repo = document_factories.Repository(author=author)
    other_author_repo = document_factories.Repository(author=other_author)
    random_author_repo = document_factories.Repository()

    all_documents = [
        author,
        other_author,
        author_article,
        author_other_article,
        other_author_article,
        random_author_article,
        author_repo,
        other_author_repo,
        random_author_repo,
    ]
    db = storage_helpers.InMemoryDocumentDatabase(documents=all_documents)

    context = zenml_helpers.FakeContext()

    author_full_names = [author.full_name, other_author.full_name]
    with config_helpers.install_in_memory_document_db(db=db):
        documents = _fetch_raw_documents.fetch_raw_documents.entrypoint(
            author_full_names=author_full_names, context=context
        )

    assert documents == [
        author_article,
        author_other_article,
        author_repo,
        other_author_article,
        other_author_repo,
    ]
    assert context.output_metadata["raw_documents"]["num_documents"] == 5
    assert context.output_metadata["raw_documents"]["articles"]["num_documents"] == 3
    assert (
        context.output_metadata["raw_documents"]["repositories"]["num_documents"] == 2
    )


def test_gets_no_raw_documents_for_author_that_does_not_exist():
    author = document_factories.Author()
    other_author_article = document_factories.Article()
    db = storage_helpers.InMemoryDocumentDatabase(
        documents=[author, other_author_article]
    )

    context = zenml_helpers.FakeContext()

    with config_helpers.install_in_memory_document_db(db=db):
        documents = _fetch_raw_documents.fetch_raw_documents.entrypoint(
            author_full_names=[author.full_name], context=context
        )

    assert documents == []
    assert context.output_metadata["raw_documents"] == {"num_documents": 0}
