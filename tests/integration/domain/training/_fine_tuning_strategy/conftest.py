import pathlib
import shutil
import typing
import uuid

import pytest


@pytest.fixture
def output_dir() -> typing.Generator[pathlib.Path, None, None]:
    """
    Create a directory to save a trained model in, and delete at the end of the test.
    """
    output_dir = pathlib.Path(__file__).parent / "outputs" / str(uuid.uuid4())
    output_dir.mkdir(parents=True, exist_ok=False)

    try:
        yield output_dir
    finally:
        shutil.rmtree(output_dir)
