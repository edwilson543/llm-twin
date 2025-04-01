from datetime import datetime as dt
from pathlib import Path

import click

from llm_twin.interfaces.cli import exceptions as cli_exceptions
from llm_twin.orchestration import pipelines


@click.command(help="Entrypoint for running the ZenML ETL pipeline")
@click.option(
    "--disable-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--config-filename",
    help="Filename of the ETL config file.",
)
def run(config_filename: str, disable_cache: bool) -> None:
    config_dir = Path(__file__).parent
    config_path = config_dir / "configs" / config_filename

    if not config_path.is_file():
        raise cli_exceptions.ConfigFileDoesNotExist(filepath=config_path)

    pipeline = pipelines.etl_user_data.with_options(
        run_name=f"etl-user-data{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        config_path=str(config_path),
        enable_cache=not disable_cache,
    )

    pipeline()


if __name__ == "__main__":
    run()
