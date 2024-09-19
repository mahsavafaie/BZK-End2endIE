import importlib
from typing import List

dataset_map = {
    "bzk": ("inferable.data.bzk_dataset", "BZKDataset"),
    "bzk_small": ("inferable.data.bzk_dataset", "BZKDatasetSmall"),
}

def get_datasets(dataset_texts: List[str]) -> List:
    if "all" in dataset_texts:
        dataset_texts = list(dataset_map.keys())
    datasets = []
    for dataset_text in dataset_texts:
        if dataset_text not in dataset_map:
            raise ValueError(f"Dataset {dataset_text} not found")
        module_name, class_name = dataset_map[dataset_text]
        my_class = getattr(importlib.import_module(module_name), class_name)
        datasets.append(my_class())
    return datasets