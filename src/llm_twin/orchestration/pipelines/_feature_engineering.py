import zenml

from llm_twin.orchestration.steps import (
    feature_engineering as feature_engineering_steps,
)


@zenml.pipeline
def process_raw_documents_into_features(author_full_names: list[str]) -> None:
    feature_engineering_steps.fetch_raw_documents(author_full_names)


if __name__ == "__main__":
    process_raw_documents_into_features.with_options(enable_cache=False)(
        author_full_names=["Ed Wilson"]
    )
