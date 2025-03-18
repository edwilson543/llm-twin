import zenml
from llm_twin.orchestration import steps


@zenml.pipeline
def some_pipeline() -> None:
    username = steps.get_user()
    steps.do_something_with_user(username=username)


if __name__ == "__main__":
    some_pipeline()
