from llm_twin.orchestration.pipelines import _rag
from testing.factories import vectors as vector_factories
from testing.helpers import config as config_helpers
from testing.helpers import zenml as zenml_helpers


def test_retrieves_relevant_context_and_uses_it_to_generate_response():
    vector_factories.EmbeddedArticleChunk.create()
    vector_factories.EmbeddedRepositoryChunk.create()

    with (
        config_helpers.install_fake_inference_engine() as fake_llm_twin,
        config_helpers.install_fake_language_model(),
    ):
        _rag.rag_inference.entrypoint(query="Some random query.", max_tokens=300)

    response = zenml_helpers.load_artifact_from_most_recent_pipeline_run(
        step_name="generate_response", output_name="response"
    )
    assert response == fake_llm_twin.stub_response
