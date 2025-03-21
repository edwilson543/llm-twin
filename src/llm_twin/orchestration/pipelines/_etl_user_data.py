import zenml

from llm_twin.orchestration.steps import etl as etl_steps


@zenml.pipeline
def etl_user_data(user_full_name: str, links: list[str]) -> None:
    user = etl_steps.get_or_create_user(user_full_name=user_full_name)
    etl_steps.crawl_links(user=user, links=links)
