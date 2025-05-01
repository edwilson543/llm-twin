import contextlib
import typing

from llm_twin import config


@contextlib.contextmanager
def create_training_report(
    *, name: str, report_to: str | None
) -> typing.Generator[None]:
    """
    Create a report and upload it to CometML.
    """

    if report_to == "comet_ml":
        import comet_ml

        config.login_to_comet_ml()

        if running_experiment := comet_ml.get_running_experiment():
            running_experiment.end()

        experiment_config = comet_ml.ExperimentConfig(name=name)
        comet_experiment = comet_ml.start(experiment_config=experiment_config)

        yield

        comet_experiment.end()

    else:
        yield
