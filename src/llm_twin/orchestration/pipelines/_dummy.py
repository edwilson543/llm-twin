import zenml

from llm_twin.orchestration.steps import etl as etl_steps


@zenml.pipeline
def some_pipeline() -> None:
    etl_steps.get_or_create_user(user_full_name="Ed Wilson")


if __name__ == "__main__":
    some_pipeline()
