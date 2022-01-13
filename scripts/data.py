from torch.utils.data import Sampler
from typing import Iterator


class WorstDialogSampler(Sampler):

    def __init__(self, data_source,
                 dialog_predictions: DialogPrediction,
                 bottom_k_percents: int):
        super().__init__(data_source)
        self.data_source = data_source
        self.dialog_prediction = dialog_predictions
        self.full_length = len(data_source)
        self.bottom_k_percents = bottom_k_percents
        self.is_init = False
        self.worst_dialog_ids = None
        self.worst_dataset_indices = None

    def set_init(self, is_init=True):
        self.is_init = is_init

    def choose_worst(self, bottom_k_percents=None):
        if bottom_k_percents is None:
            bottom_k_percents = self.bottom_k_percents
        self.set_init(True)
        self.worst_dialog_ids = set(self.dialog_prediction.get_bottom_k_percents(bottom_k_percents))
        self.worst_dataset_indices = []

        for i in range(len(self.data_source)):
            d = self.data_source[i]
            if d['dialog_num'] in self.worst_dialog_ids:
                self.worst_dataset_indices.append(i)

    def __iter__(self) -> Iterator[int]:
        if not self.is_init:
            return iter(range(len(self.data_source)))
        else:
            return iter(self.worst_dataset_indices)

    def __len__(self):
        if not self.is_init:
            return self.full_length
        else:
            return len(self.worst_dataset_indices)
