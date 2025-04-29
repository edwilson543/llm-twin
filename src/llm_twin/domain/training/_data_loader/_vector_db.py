import dataclasses

from llm_twin.domain import dataset_generation
from llm_twin.domain.storage import vector as vector_storage

from . import _base


@dataclasses.dataclass(frozen=True)
class VectorDBDataLoader(_base.DataLoader):
    db: vector_storage.VectorDatabase

    def load_instruct_dataset(
        self,
    ) -> dataset_generation.TrainTestSplit[dataset_generation.InstructSample]:
        results, _ = self.db.bulk_find(
            vector_class=dataset_generation.TrainTestSplit,
            limit=1,
            dataset_type=dataset_generation.DatasetType.INSTRUCT,
        )
        return results[0]

    def load_preference_dataset(
        self,
    ) -> dataset_generation.TrainTestSplit[dataset_generation.PreferenceSample]:
        results, _ = self.db.bulk_find(
            vector_class=dataset_generation.TrainTestSplit,
            limit=1,
            dataset_type=dataset_generation.DatasetType.PREFERENCE,
        )
        return results[0]
