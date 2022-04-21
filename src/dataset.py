import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class EntityLabel:
    start_idx: int
    stop_idx: int
    entity_name: str
    text: str


@dataclass
class DataItem:
    text: str
    entity_labels: List[EntityLabel]


class DataSet:
    def __init__(self, json_path: Path) -> None:
        with open(json_path) as f:
            self.data = json.load(f)

    def __getitem__(self, idx) -> DataItem:
        dct = self.data[idx]

        annotations = _pick_only_item(dct["annotations"])["result"]
        entity_labels = [
            EntityLabel(
                start_idx=ann["value"]["start"],
                stop_idx=ann["value"]["end"],
                text=ann["value"]["text"],
                entity_name=_pick_only_item(ann["value"]["labels"]),
            )
            for ann in annotations
        ]

        return DataItem(
            text=dct["data"]["text"],
            entity_labels=entity_labels,
        )

    def __len__(self):
        return len(self.data)


def _pick_only_item(lst):
    assert len(lst) == 1
    return lst[0]
