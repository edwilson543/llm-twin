from bs4 import BeautifulSoup

from llm_twin.domain import authors
from llm_twin.domain.etl import raw_documents
from llm_twin.domain.storage import document as document_storage

from . import _base


class MediumCrawler(_base.SeleniumCrawler):
    _document_class = raw_documents.Article

    def _set_extra_driver_options(self, options) -> None:
        options.add_argument(r"--profile-directory=Profile 2")

    def _extract(
        self,
        *,
        db: document_storage.DocumentDatabase,
        link: str,
        author: authors.Author,
    ) -> None:
        self.driver.get(link)
        self.scroll_page()

        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        title = soup.find_all("h1", class_="pw-post-title")
        subtitle = soup.find_all("h2", class_="pw-subtitle-paragraph")

        data = {
            "Title": str(title[0]) if title else None,
            "Subtitle": str(subtitle[0]) if subtitle else None,
            "Content": soup.get_text(),
        }

        self.driver.close()

        document = raw_documents.Article(
            platform="medium",
            content=data,
            link=link,
            author_id=author.id,
            author_full_name=author.full_name,
        )
        document.save(db=db)
