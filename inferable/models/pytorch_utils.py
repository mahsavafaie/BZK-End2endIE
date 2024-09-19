from typing import Tuple, List
import datasets
import torch


class PyTorchDatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset: datasets.arrow_dataset.Dataset, ordered_dataset_keys: List[str], keep_empty_columns: bool = True) -> None:
        super().__init__()
        self.dataset = dataset
        self.dataset_length = len(self.dataset)
        self.ordered_dataset_keys = ordered_dataset_keys
        self.keep_empty_columns = keep_empty_columns
    
    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.dataset[idx]
        target_sequence = ''
        #iterate over column header and value - it is important that the order is the same thus we use ordered_dataset_keys
        if self.keep_empty_columns:
            for column in self.ordered_dataset_keys:
                column_value = sample[column]
                if column_value is None:
                    column_value = ''
                target_sequence += f"<s_{column}>{column_value}</s_{column}>"
        else:
            for column in self.ordered_dataset_keys:
                column_value = sample[column]
                if column_value:
                    target_sequence += f"<s_{column}>{column_value}</s_{column}>"
        return sample["image"], target_sequence