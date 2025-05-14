from llm_twin.domain.models import _cross_encoder


class TestSentenceTransformerCrossEncoder:
    def test_predicts_similarity_scores_for_each_pair(self):
        model = _cross_encoder.SentenceTransformerCrossEncoder(
            model_name=_cross_encoder.CrossEncoderModelName.MINILM
        )
        pairs = [("beef", "witch"), ("apples", "oranges")]

        predictions = model.predict(pairs=pairs)

        assert len(predictions) == 2
        assert all(prediction != 0 for prediction in predictions)
        assert predictions[0] < predictions[1]
