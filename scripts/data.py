from abc import ABC, abstractmethod
from typing import Iterator

from torch.utils.data import Sampler

from scripts.utils import AbstractDialogPrediction


class AbstractWorstDialogSampler(ABC, Sampler):

    @abstractmethod
    def update_worst(self, bottom_k_percents=None):
        pass


class WorstDialogSampler(AbstractWorstDialogSampler):

    def __init__(self, data_source,
                 dialog_predictions: AbstractDialogPrediction,
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

    def update_worst(self, bottom_k_percents=None):
        if bottom_k_percents is None:
            bottom_k_percents = self.bottom_k_percents
        self.set_init(True)
        self.worst_dialog_ids = set(self.dialog_prediction.
                                    get_bottom_k_percents(bottom_k_percents))
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


class WorstDialogSamplerWithRemoval(AbstractWorstDialogSampler):

    def __init__(self,
                 data_source,
                 dialog_predictions: AbstractDialogPrediction,
                 bottom_k_percents: int):
        super().__init__(data_source)
        self.dialog_ids = list(range(len(data_source)))
        self.next_dialogs = set()
        self.dialog_predictions = dialog_predictions
        self.bottom_k_percents = bottom_k_percents

    def update_worst(self, bottom_k_percents=None):
        self.next_dialogs = set(self.dialog_predictions.
                                get_bottom_k_percents(self.bottom_k_percents))
        self.dialog_ids = [did for did in self.dialog_ids
                           if did not in self.next_dialogs]

    def __len__(self):
        return len(self.next_dialogs)

    def __iter__(self):
        return iter(self.next_dialogs)
