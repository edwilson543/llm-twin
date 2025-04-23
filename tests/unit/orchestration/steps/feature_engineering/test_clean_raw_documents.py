from llm_twin.domain.feature_engineering import cleaning
from llm_twin.orchestration.steps.feature_engineering import _clean_raw_documents
from testing.factories import documents as document_factories
from testing.helpers import config as config_helpers
from testing.helpers import zenml as zenml_helpers


def test_cleans_all_raw_documents_and_persists_in_database():
    article = document_factories.Article()
    repository = document_factories.Repository()

    context = zenml_helpers.FakeContext()

    with config_helpers.install_in_memory_vector_db() as db:
        cleaned_documents = _clean_raw_documents.clean_raw_documents.entrypoint(
            raw_documents=[article, repository], context=context
        )

    assert db.vectors == cleaned_documents
    assert len(db.vectors) == 2

    cleaned_article = db.vectors_by_id[article.id]
    assert isinstance(cleaned_article, cleaning.CleanedArticle)

    cleaned_repository = db.vectors_by_id[repository.id]
    assert isinstance(cleaned_repository, cleaning.CleanedRepository)
