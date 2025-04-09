from llm_twin.orchestration.pipelines import _feature_engineering
from testing.factories import documents as document_factories


def test_processes_raw_documents_for_given_authors_into_features():
    author = document_factories.Author()
    author_full_names = [author.full_name]

    # TODO -> create an article for author, and repository for another author.
    # TODO -> how can this be sped up...

    _feature_engineering.process_raw_documents_into_features.entrypoint(
        author_full_names=author_full_names
    )
