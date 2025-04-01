import zenml

from llm_twin.orchestration.steps import etl as etl_steps


@zenml.pipeline
def etl_author_data(author_full_name: str, links: list[str]) -> None:
    author = etl_steps.get_or_create_author(full_name=author_full_name)
    etl_steps.crawl_links(author=author, links=links)
