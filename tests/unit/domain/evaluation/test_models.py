from llm_twin.domain import evaluation
from testing.factories import evaluation as evaluation_factories


class TestEvaluationSummary__Mean:
    def test_gets_mean_for_each_evaluation_criteria(self):
        scores = [
            evaluation_factories.EvaluationCriteria(score=score)
            for score in range(1, 4)
        ]

        evaluation_summary = evaluation.EvaluationSummary.mean(
            evaluations=[
                evaluation_factories.Evaluation(accuracy=scores[0], style=scores[1]),
                evaluation_factories.Evaluation(accuracy=scores[1], style=scores[2]),
            ]
        )

        assert evaluation_summary.accuracy == 1.5
        assert evaluation_summary.style == 2.5
