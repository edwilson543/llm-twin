import zenml


@zenml.pipeline
def rag_inference(query: str) -> None:
    raise NotImplementedError
    # prompt = rag.augment_query(
    #
    # )
