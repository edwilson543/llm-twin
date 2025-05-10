from llm_twin.domain import evaluation
from testing.factories import evaluation as evaluation_factories
from testing.helpers import models as models_helpers


class TestEvaluationSummary__Mean:
    def test_gets_mean_for_each_evaluation_criteria(self):
        scores = [
            evaluation_factories.EvaluationCriteria(score=score)
            for score in range(1, 4)
        ]

        evaluation_aggregate = evaluation.EvaluationAggregate.mean(
            evaluations=[
                evaluation_factories.Evaluation(accuracy=scores[0], style=scores[1]),
                evaluation_factories.Evaluation(accuracy=scores[1], style=scores[2]),
            ]
        )

        assert evaluation_aggregate.aggregate == evaluation.Aggregate.MEAN
        assert evaluation_aggregate.accuracy == 1.5
        assert evaluation_aggregate.style == 2.5


class TestCompletion__Evaluate:
    def test_gets_evaluation_produced_by_language_model(self):
        completion = evaluation_factories.Completion()
        language_model = models_helpers.FakeLanguageModel()

        eval = completion.evaluate(language_model=language_model)

        assert eval.accuracy.score > 0
        assert eval.style.score > 0
