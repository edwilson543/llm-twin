import zenml

from llm_twin.domain import documents
from llm_twin.orchestration.steps import etl as etl_steps


@zenml.pipeline
def etl_user_data(
    user_full_name: str, links: list[str], db: documents.NoSQLDatabase | None = None
) -> None:
    user = etl_steps.get_or_create_user(user_full_name=user_full_name, db=db)
    etl_steps.crawl_links(user=user, links=links, db=db)


if __name__ == "__main__":
    links = [
        "https://fake.com/edwilson543/post-1",
        "https://fake.com/edwilson543/post-2",
        "https://fake.com/edwilson543/post-3",
    ]
    etl_user_data("Ed Wilson", links)
