import collections
import dataclasses
import typing

import zenml
from zenml import client as zenml_client

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


def get_last_pipeline_run() -> zenml.PipelineRunResponse:
    client = zenml_client.Client()
    return client.list_pipeline_runs(sort_by="desc:created")[0]


def load_artifact_from_most_recent_pipeline_run(
    *, step_name: str, output_name: str
) -> object:
    pipeline_run = get_last_pipeline_run()
    output_list = pipeline_run.steps[step_name].outputs[output_name]
    assert len(output_list) == 1
    return output_list[0].load()
