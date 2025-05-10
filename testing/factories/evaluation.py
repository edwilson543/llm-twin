import factory

from llm_twin.domain import evaluation


class InstructionAnswerPair(factory.Factory):
    instruction = factory.Sequence(lambda n: f"instruction-{n}")
    answer = factory.Sequence(lambda n: f"answer-{n}")

    class Meta:
        model = evaluation.Evaluation


class EvaluationCriteria(factory.Factory):
    score = 1
    analysis = factory.Sequence(lambda n: f"analysis-{n}")

    class Meta:
        model = evaluation.EvaluationCriteria


class Evaluation(factory.Factory):
    accuracy = factory.SubFactory(EvaluationCriteria)
    style = factory.SubFactory(EvaluationCriteria)

    class Meta:
        model = evaluation.Evaluation


class Completion(factory.Factory):
    prompt = factory.Sequence(lambda n: f"prompt-{n}")
    response = factory.Sequence(lambda n: f"response-{n}")

    class Meta:
        model = evaluation.Completion
