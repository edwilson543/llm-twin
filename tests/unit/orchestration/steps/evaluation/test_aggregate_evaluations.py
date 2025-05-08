from llm_twin.domain import evaluation
from llm_twin.orchestration.steps.evaluation import _aggregate_evaluations
from testing.factories import evaluation as evaluation_factories
from testing.helpers import zenml as zenml_helpers


def test_aggregates_evaluations_into_useful_measures():
    evaluations = [evaluation_factories.Evaluation() for _ in range(2)]
    context = zenml_helpers.FakeContext()

    aggregates = _aggregate_evaluations.aggregate_evaluations.entrypoint(
        evaluations, context
    )

    assert len(aggregates) == 1
    mean = aggregates[0]
    assert mean.aggregate == evaluation.Aggregate.MEAN
    assert mean.accuracy > 0
    assert mean.style > 0

    assert context.output_metadata["evaluation_aggregates"] == {
        "mean": {
            "aggregate": evaluation.Aggregate.MEAN,
            "accuracy": mean.accuracy,
            "style": mean.style,
        }
    }
