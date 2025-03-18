import collections
import dataclasses
import typing

from llm_twin.orchestration.steps import context


@dataclasses.dataclass(frozen=True)
class FakeContext(context.StepContext):
    output_metadata: dict[str, dict[str, typing.Any]] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(dict)
    )

    def add_output_metadata(
        self,
        output_name: str,
        metadata: dict[str, typing.Any],
    ) -> None:
        self.output_metadata[output_name].update(metadata)
