from urllib.parse import urlparse

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer

from llm_twin.domain import authors
from llm_twin.domain.etl import raw_documents

from . import _base


class CustomArticleCrawler(_base.Crawler):
    _document_class = raw_documents.Article

    def __init__(self) -> None:
        super().__init__()

    def _extract(self, *, link: str, author: authors.Author) -> raw_documents.Article:
        html_loader = AsyncHtmlLoader(link, verify_ssl=False)
        html_documents = html_loader.load()

        html_parser = Html2TextTransformer()
        parsed_document = html_parser.transform_documents(html_documents)[0]

        content = {
            "Title": parsed_document.metadata.get("title"),
            "Subtitle": parsed_document.metadata.get("description"),
            "Content": parsed_document.page_content,
            "language": parsed_document.metadata.get("language"),
        }

        platform = urlparse(link).netloc

        return raw_documents.Article(
            content=content,
            link=link,
            platform=platform,
            author_id=author.id,
            author_full_name=author.full_name,
        )
