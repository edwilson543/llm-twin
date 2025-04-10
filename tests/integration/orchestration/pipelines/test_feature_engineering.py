from llm_twin import settings
from llm_twin.domain.feature_engineering import embedding
from llm_twin.orchestration.pipelines import _feature_engineering
from testing.factories import documents as document_factories


def test_processes_raw_article_for_given_author_into_features():
    author = document_factories.Author.create()
    article = document_factories.Article.create(author=author)
    other_article = document_factories.Article.create(author=author)

    _feature_engineering.process_raw_documents_into_features.entrypoint(
        author_full_names=[author.full_name]
    )

    db = settings.get_vector_database()
    embedded_articles, next_offset = db.bulk_find(
        vector_class=embedding.EmbeddedArticleChunk, limit=3
    )
    assert next_offset is None
    assert len(embedded_articles) == 2

    embedded_raw_document_ids = {
        embedded_article.raw_document_id for embedded_article in embedded_articles
    }
    assert embedded_raw_document_ids == {article.id, other_article.id}
