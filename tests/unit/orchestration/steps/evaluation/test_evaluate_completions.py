from llm_twin.domain import evaluation
from llm_twin.orchestration.steps.evaluation import _evaluate_completions
from testing.factories import evaluation as evaluation_factories
from testing.helpers import config as config_helpers


def test_gets_an_evaluation_for_each_completion():
    completions = [evaluation_factories.Completion() for _ in range(2)]

    with config_helpers.install_fake_language_model():
        evaluations = _evaluate_completions.evaluate_completions.entrypoint(completions)

    assert len(evaluations) == 2
    assert all(isinstance(eval, evaluation.Evaluation) for eval in evaluations)
