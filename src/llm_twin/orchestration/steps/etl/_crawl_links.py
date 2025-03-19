import typing
from collections import defaultdict
from urllib.parse import urlparse

import loguru
import tqdm
import zenml

from llm_twin.domain import crawling, documents
from llm_twin.infrastructure import documents as documents_backend
from llm_twin.orchestration.steps import context


@zenml.step
def crawl_links(
    user: documents.UserDocument,
    links: list[str],
    context: context.StepContext | None = None,
) -> typing.Annotated[list[str], "crawled_links"]:
    dispatcher = crawling.CrawlerDispatcher()
    loguru.logger.info(f"Crawling links {links}")

    metadata = defaultdict(lambda: defaultdict(int))
    successful_crawls = 0

    for link in tqdm.tqdm(links):
        domain = urlparse(link).netloc
        crawler = dispatcher.get_crawler(link=link)

        try:
            crawler.extract(link=link, user=user)
            metadata[domain]["successful"] += 1
            successful_crawls += 1
        except crawling.UnableToCrawlLink:
            loguru.logger.error(f"Unable to crawl link: {link}")

        metadata[domain]["total"] += 1

    step_context = context or zenml.get_step_context()
    step_context.add_output_metadata(
        output_name="crawled_links", metadata=dict(metadata)
    )

    loguru.logger.info(
        f"Successfully crawled {successful_crawls} / {len(links)} links."
    )

    return links
