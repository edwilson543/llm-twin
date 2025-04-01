from bs4 import BeautifulSoup

from llm_twin.domain import raw_documents

from . import _base


class MediumCrawler(_base.SeleniumCrawler):
    _document_class = raw_documents.ArticleDocument

    def _set_extra_driver_options(self, options) -> None:
        options.add_argument(r"--profile-directory=Profile 2")

    def _extract(
        self,
        *,
        db: raw_documents.RawDocumentDatabase,
        link: str,
        user: raw_documents.UserDocument,
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

        document = raw_documents.ArticleDocument(
            platform="medium",
            content=data,
            link=link,
            author_id=user.id,
            author_full_name=user.full_name,
        )
        document.save(db=db)
