import os
import shutil
import subprocess
import tempfile

from llm_twin.domain.etl import raw_documents

from . import _base


class GithubCrawler(_base.Crawler):
    _document_class = raw_documents.Repository

    def __init__(self, ignore=(".git", ".toml", ".lock", ".png")) -> None:
        super().__init__()
        self._ignore = ignore

    def _extract(
        self,
        *,
        db: raw_documents.RawDocumentDatabase,
        link: str,
        author: raw_documents.Author,
    ) -> None:
        """
        Extract the content from a GitHub repo and save it in the db.
        """
        repo_name = link.rstrip("/").split("/")[-1]
        local_temp_dir = tempfile.mkdtemp()

        try:
            os.chdir(local_temp_dir)
            subprocess.run(["git", "clone", link])

            repo_path = os.path.join(local_temp_dir, os.listdir(local_temp_dir)[0])

            tree = {}
            for root, _, files in os.walk(repo_path):
                directory = root.replace(repo_path, "").lstrip("/")
                if directory.startswith(self._ignore):
                    continue

                for file in files:
                    if file.endswith(self._ignore):
                        continue
                    file_path = os.path.join(directory, file)
                    with open(os.path.join(root, file), "r", errors="ignore") as f:
                        tree[file_path] = f.read().replace(" ", "")
        finally:
            shutil.rmtree(local_temp_dir)

        instance = raw_documents.Repository(
            content=tree,
            name=repo_name,
            link=link,
            platform="github",
            author_id=author.id,
            author_full_name=author.full_name,
        )
        instance.save(db=db)
