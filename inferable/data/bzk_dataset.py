from datasets import load_dataset
from inferable.data.base_dataset import BaseDataset
from datasets.arrow_dataset import Dataset

class BZKDataset(BaseDataset):

    def get_training_data(self) -> Dataset:
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="train")
    
    def get_validation_data(self) -> Dataset:
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="validation")
    
    def get_test_data(self) -> Dataset:
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="test")

class BZKDatasetSmall(BaseDataset):

    def get_training_data(self) -> Dataset:
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="train[:10%]")
    
    def get_validation_data(self) -> Dataset:
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="validation[:10%]")
    
    def get_test_data(self) -> Dataset:
        return load_dataset("imagefolder", data_dir="./inferable/data/BZK", split="test[:10%]")