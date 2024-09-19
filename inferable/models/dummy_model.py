from inferable.models.base_model import BaseModel
from typing import Dict, Iterable
import datasets
from PIL.Image import Image


class DummyModel(BaseModel):
    
    def __init__(self) -> None:
        self.predict_keys = None
    
    def fit(self, training_data: datasets.arrow_dataset.Dataset, validation_dat: datasets.arrow_dataset.Dataset) -> None:
        self.predict_keys = list(training_data.features.keys())
        self.predict_keys.remove('image')


    def predict(self, test_data: Iterable[Image]) -> Iterable[Dict[str, str]]:
        for image in test_data:
            yield { prediction_key: "" for prediction_key in self.predict_keys}
    
    def __str__(self):
        return "DummyModel"