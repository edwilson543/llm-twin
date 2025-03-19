import abc
import dataclasses
import tempfile
import time

from selenium import webdriver
from selenium.webdriver.chrome import options as chrome_options

from llm_twin.domain import documents


@dataclasses.dataclass(frozen=True)
class UnableToCrawlLink(Exception):
    link: str


class Crawler(abc.ABC):
    """
    Base class for crawling a webpage and extracting relevant information.
    """

    @abc.abstractmethod
    def extract(self, *, link: str, user: documents.UserDocument) -> None:
        raise NotImplementedError


class SeleniumCrawler(Crawler, abc.ABC):
    """
    Base class for crawling a webpage using selenium for browser automation.
    """

    def __init__(self, scroll_limit: int = 5) -> None:
        options = webdriver.ChromeOptions()

        options.add_argument("--no-sandbox")
        options.add_argument("--headless=new")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--ignore-certificate-errors")
        options.add_argument(f"--user-data-dir={tempfile.mkdtemp()}")
        options.add_argument(f"--data-path={tempfile.mkdtemp()}")
        options.add_argument(f"--disk-cache-dir={tempfile.mkdtemp()}")
        options.add_argument("--remote-debugging-port=9226")

        self._set_extra_driver_options(options)

        self.scroll_limit = scroll_limit
        self.driver = webdriver.Chrome(
            options=options,
        )

    def _set_extra_driver_options(self, options: chrome_options.Options) -> None:
        return None

    def login(self) -> None:
        return None

    def scroll_page(self) -> None:
        """
        Scroll through the LinkedIn page based on the scroll limit.
        """
        current_scroll = 0
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )

            # Wait for new content to load after performing doom scroll.
            time.sleep(5)

            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height or (
                self.scroll_limit and current_scroll >= self.scroll_limit
            ):
                break
            last_height = new_height
            current_scroll += 1
