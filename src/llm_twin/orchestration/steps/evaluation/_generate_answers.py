import typing

import transformers
import zenml

from llm_twin import config
from llm_twin.domain import evaluation, training
from llm_twin.orchestration.steps import context
import loguru


@zenml.step
def generate_answers(
    author_id: str,
    load_model_from: str,
    max_tokens: int = 4096,
    top_k: int = 1,
    context: context.StepContext | None = None,
) -> typing.Annotated[list[evaluation.InstructionAnswerPair], "answers"]:
    db = config.get_vector_database()
    data_loader = training.VectorDBDataLoader(db=db)
    dataset = data_loader.load_instruct_dataset(author_id=author_id)

    model = transformers.AutoModelForCausalLM.from_pretrained(load_model_from)
    tokenizer = transformers.AutoTokenizer.from_pretrained(load_model_from)

    loguru.logger.info(f"Loaded model and tokenizer from {load_model_from}")

    pairs: list[evaluation.InstructionAnswerPair] = []
    for sample in dataset.test.samples:
        instruction_tokens = tokenizer.encode(sample.instruction, return_tensors="pt")
        answer_tokens = model.generate(
            instruction_tokens, max_length=max_tokens, top_k=top_k
        )
        answer = tokenizer.decode(answer_tokens[0], skip_special_tokens=True)

        pair = evaluation.InstructionAnswerPair(
            instruction=sample.instruction, answer=answer
        )
        pairs.append(pair)

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="answers", metadata={"num_pairs": len(pairs)}
    )

    return pairs
