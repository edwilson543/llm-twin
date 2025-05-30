import dataclasses

from llm_twin.domain import dataset_generation
from llm_twin.domain.storage import vector as vector_storage

from . import _base


@dataclasses.dataclass(frozen=True)
class VectorDBDataLoader(_base.DataLoader):
    db: vector_storage.VectorDatabase

    def load_instruct_dataset(
        self, *, author_id: str
    ) -> dataset_generation.TrainTestSplit[dataset_generation.InstructSample]:
        results, _ = self.db.bulk_find(
            vector_class=dataset_generation.TrainTestSplit,
            limit=1,
            author_id=author_id,
            dataset_type=dataset_generation.DatasetType.INSTRUCT.value,
        )

        try:
            return results[0]
        except IndexError as exc:
            raise _base.UnableToLoadDataset(
                dataset_type=dataset_generation.DatasetType.INSTRUCT
            ) from exc

    def load_preference_dataset(
        self, *, author_id: str
    ) -> dataset_generation.TrainTestSplit[dataset_generation.PreferenceSample]:
        results, _ = self.db.bulk_find(
            vector_class=dataset_generation.TrainTestSplit,
            limit=1,
            author_id=author_id,
            dataset_type=dataset_generation.DatasetType.PREFERENCE.value,
        )

        try:
            return results[0]
        except IndexError as exc:
            raise _base.UnableToLoadDataset(
                dataset_type=dataset_generation.DatasetType.PREFERENCE
            ) from exc
