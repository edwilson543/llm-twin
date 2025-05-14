from llm_twin.domain.rag import _augmentation
from testing.factories import documents as document_factories
from testing.factories import vectors as vector_factories


class TestAugmentQuery:
    def test_augments_query_with_passed_documents(self):
        author = document_factories.Author(first_name="Ed", last_name="Wilson")
        article = vector_factories.EmbeddedArticleChunk(author=author, content="Text")
        repository = vector_factories.EmbeddedRepositoryChunk(
            author=author, content="Code"
        )

        augmented_query = _augmentation.augment_query(
            query="Query?", documents=[article, repository]
        )

        assert (
            augmented_query
            == "\nAnswer the query below using the provided context as the primary source of information.\n\nQuery: Query?\nContext: Category: articles\n            Platform: some-platform\n            Author: Ed Wilson\n            Content: Text\nCategory: repositories\n            Platform: some-platform\n            Author: Ed Wilson\n            Content: Code\n"
        )
