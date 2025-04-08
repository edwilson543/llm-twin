import contextlib
import typing
from unittest import mock


@contextlib.contextmanager
def mock_repository_chunker(
    chunks: list[str],
) -> typing.Generator[mock.Mock, None, None]:
    """
    Mock the repository chunker's chunking method, to avoid calling the hugging face API

    TODO -> make this unnecessary.
    """
    method = "llm_twin.domain.feature_engineering.chunking._chunkers._repository.RepositoryChunker._chunk_content"
    with mock.patch(method, return_value=chunks) as mock_chunker:
        yield mock_chunker
