import typing


@typing.runtime_checkable
class StepContext(typing.Protocol):
    """
    Protocol to support injecting context to test functions.
    """

    def add_output_metadata(
        self,
        output_name: str,
        metadata: dict[str, typing.Any],
    ) -> None: ...
