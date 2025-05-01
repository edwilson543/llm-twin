from llm_twin import config
from llm_twin.domain.feature_engineering import embedding
from llm_twin.orchestration.pipelines import _feature_engineering
from testing.factories import documents as document_factories


def test_processes_raw_article_for_given_author_into_features():
    author = document_factories.Author.create()
    article = document_factories.Article.create(
        author=author, content={"content": "A. " * 2000}
    )
    other_article = document_factories.Article.create(
        author=author, content={"content": "B. " * 2000}
    )

    pipeline = _feature_engineering.process_raw_documents_into_features.with_options(
        enable_cache=False
    )
    pipeline(author_full_name=author.full_name)

    db = config.get_vector_database()
    embedded_articles, next_offset = db.bulk_find(
        vector_class=embedding.EmbeddedArticleChunk, limit=5
    )
    assert next_offset is None
    assert len(embedded_articles) == 4

    embedded_raw_document_ids = {
        embedded_article.raw_document_id for embedded_article in embedded_articles
    }
    assert embedded_raw_document_ids == {article.id, other_article.id}


def test_processes_raw_repository_for_given_author_into_features():
    author = document_factories.Author.create()
    repo_code = "def code(): " * 1000
    repository = document_factories.Repository.create(
        author=author, content={"code": repo_code}
    )

    pipeline = _feature_engineering.process_raw_documents_into_features.with_options(
        enable_cache=False
    )
    pipeline(author_full_name=author.full_name)

    db = config.get_vector_database()
    embedded_repository_chunks, next_offset = db.bulk_find(
        vector_class=embedding.EmbeddedRepositoryChunk, limit=2
    )
    assert next_offset is None
    assert len(embedded_repository_chunks) == 2
    for chunk in embedded_repository_chunks:
        assert chunk.raw_document_id == repository.id
        assert chunk.author_id == author.id
