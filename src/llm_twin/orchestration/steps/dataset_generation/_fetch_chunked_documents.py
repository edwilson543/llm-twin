import typing

import zenml

from llm_twin.domain.feature_engineering import chunking


@zenml.step
def fetch_chunked_documents() -> typing.Annotated[
    list[chunking.Chunk], "chunked_documents"
]:
    raise NotImplementedError
