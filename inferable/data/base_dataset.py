from datasets.arrow_dataset import Dataset
from abc import ABC, abstractmethod

class BaseDataset(ABC):

    @abstractmethod
    def get_training_data(self) -> Dataset:
        """Returns the training data.
        Returns
        ----------
        Dataset
            Images and metadata for training
        """
        pass

    @abstractmethod
    def get_validation_data(self) -> Dataset:
        """Returns the validation data.
        Returns
        ----------
        Dataset
            Images and metadata for validation
        """
        pass
    
    @abstractmethod
    def get_test_data(self) -> Dataset:
        """Returns the test data.
        Returns
        ----------
        Dataset
            Images and metadata for testing
        """
        pass