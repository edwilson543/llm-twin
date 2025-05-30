import json
import typing
from collections import defaultdict
from urllib.parse import urlparse

import loguru
import tqdm
import zenml

from llm_twin import config
from llm_twin.domain import authors
from llm_twin.domain.etl import crawling
from llm_twin.orchestration.steps import context


@zenml.step
def crawl_links(
    author: authors.Author,
    links: list[str],
    context: context.StepContext | None = None,
) -> typing.Annotated[list[str], "crawled_links"]:
    loguru.logger.info(f"Crawling links {links}")

    db = config.get_document_database()
    dispatcher = crawling.CrawlerDispatcher()

    metadata: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    successful_crawls = 0

    for link in tqdm.tqdm(links):
        domain = urlparse(link).netloc
        crawler = dispatcher.get_crawler(link=link)

        try:
            crawler.extract(db=db, link=link, author=author)
            metadata[domain]["successful"] += 1
            successful_crawls += 1
            loguru.logger.info(f"Successfully crawled {link}.")
        except crawling.UnableToCrawlLink:
            loguru.logger.error(f"Unable to crawl link: {link}")

        metadata[domain]["total"] += 1

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="crawled_links", metadata=json.loads(json.dumps(metadata))
    )

    loguru.logger.info(
        f"Successfully crawled {successful_crawls} / {len(links)} links."
    )

    return links
