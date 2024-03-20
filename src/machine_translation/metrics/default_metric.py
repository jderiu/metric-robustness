from typing import List, Dict
from comet import download_model, load_from_checkpoint


class AutoMetric:

    def batch_rating(self, batch_sample: List[Dict]) -> List[float]:
        pass

    def single_rating(self, sample: Dict) -> float:
        pass


class CometMetric(AutoMetric):

    def __init__(
            self
    ):
        model_path = download_model("Unbabel/wmt22-comet-da")
        self.comet = load_from_checkpoint(model_path)

    def batch_rating(self, batch_sample: List[Dict]) -> List[float]:
        return self.comet.predict(batch_sample, batch_size=8, gpus=1, progress_bar=False).scores

    def single_rating(self, sample: Dict) -> float:
        return self.comet.predict([sample], batch_size=1, gpus=1, progress_bar=False).scores[0]