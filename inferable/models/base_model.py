from typing import Dict, Iterable
from datasets.arrow_dataset import Dataset
from abc import ABC, abstractmethod
from PIL.Image import Image

class BaseModel(ABC):

    @abstractmethod
    def fit(self, training_data: Dataset, validation_dat: Dataset) -> None:
        """Trains the model on the given training and validation data.
        Parameters
        ----------
        training_data: Dataset
            the dataset to train the model on (the image is stored in the 'image' column and all other columns are metadata)
        validation_dat: Dataset
            the dataset to validate the model on (the image is stored in the 'image' column and all other columns are metadata)
        Returns
        ----------"""
        pass
    
    @abstractmethod
    def predict(self, test_data: Iterable[Image]) -> Iterable[Dict[str, str]]:
        """Predicts the metadata for the given test data.
        Parameters
        ----------
        test_data: Iterable[Image]
            the images to iterate over and predict the metadata for
        Returns
        ----------
        Iterable of dicts
            an iterable of dicttionaries with th metadata for each image
        """
        pass