from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler

from scripts.utils import AbstractDialogMetricCounter


class AbstractWorstDialogSampler(ABC, Sampler):

    @abstractmethod
    def update_source_after_epoch(self, **kwargs):
        pass

    @abstractmethod
    def eval(self, is_eval):
        pass


class WorstDialogSampler(AbstractWorstDialogSampler):

    def __init__(self, data_source,
                 dialog_predictions: AbstractDialogMetricCounter,
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

    def eval(self, is_eval):
        pass

    def update_source_after_epoch(self, bottom_k_percents=None):
        if bottom_k_percents is None:
            bottom_k_percents = self.bottom_k_percents
        self.set_init(True)
        self.worst_dialog_ids = set(self.dialog_prediction.
                                    get_bottom_k_percents(bottom_k_percents))
        self.worst_dataset_indices = []

        for i in range(len(self.data_source)):
            d = self.data_source[i]
            if d['dialog_id'] in self.worst_dialog_ids:
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
                 dialog_predictions: AbstractDialogMetricCounter,
                 bottom_k: int,
                 is_percent: bool):
        super().__init__(data_source)

        self.phrase_mapping = {i: d['dialog_id'] for i, d in enumerate(data_source)}
        self.dialog_phrase_mapping = defaultdict(list)
        for phrase_id, dialog_id in self.phrase_mapping.items():
            self.dialog_phrase_mapping[dialog_id].append(phrase_id)
        self.next_dialog_phrases = set()
        self.remain_dialog_phrases = set(range(len(data_source)))

        self.dialog_predictions = dialog_predictions
        self.bottom_k = bottom_k
        self.is_percent = is_percent
        self.is_eval = False

    def eval(self, is_eval):
        self.is_eval = is_eval

    def update_source_after_epoch(self, bottom_k=None, is_percent=None):
        if bottom_k is None:
            bottom_k = self.bottom_k
        if is_percent is None:
            is_percent = self.is_percent
        if is_percent:
            dialogs = self.dialog_predictions. \
                get_bottom_k_percents(bottom_k)
        else:
            dialogs = self.dialog_predictions. \
                get_bottom_k_values(bottom_k)

        self.next_dialog_phrases = []
        # print('-' * 10)
        # print(sorted(dialogs)[:100])
        # print(sorted(self.dialog_phrase_mapping.keys())[:100])
        for dialog in dialogs:
            self.next_dialog_phrases += self.dialog_phrase_mapping.pop(dialog)
        # print(sorted(self.next_dialog_phrases)[:100])
        # print('-' * 10)
        for phrase in self.next_dialog_phrases:
            self.remain_dialog_phrases.remove(phrase)

    def __len__(self):
        if self.is_eval:
            return len(self.remain_dialog_phrases)
        else:
            return len(self.next_dialog_phrases)

    def __iter__(self):
        if self.is_eval:
            return iter(self.remain_dialog_phrases)
        else:
            return iter(self.next_dialog_phrases)


class RandomSamplerWithRemoval(AbstractWorstDialogSampler):

    def __init__(self, data_source, sample_count):
        super().__init__(data_source)
        self.sample_count = sample_count
        self.available_dialog_ids = range(len(data_source))
        self.next_dialog_ids = set()

        self.phrase_mapping = {i: d['dialog_id'] for i, d in enumerate(data_source)}
        self.dialog_phrase_mapping = defaultdict(list)
        for phrase_id, dialog_id in self.phrase_mapping.items():
            self.dialog_phrase_mapping[dialog_id].append(phrase_id)
        self.next_phrases = []
        self._update_next_dialog_ids(sample_count)

    def _update_next_dialog_ids(self, sample_count):

        self.next_dialog_ids = set(
            [int(i) for i in np.random.choice(list(self.dialog_phrase_mapping.keys()),
                                              min(sample_count,
                                                  len(self.dialog_phrase_mapping)))])
        self.next_phrases = []
        for d in self.next_dialog_ids:
            self.next_phrases += self.dialog_phrase_mapping.pop(d)

        # self.available_dialog_ids = [dialog_id for dialog_id in
        #                              self.available_dialog_ids if
        #                              dialog_id not in self.next_dialog_ids]

    def update_source_after_epoch(self, sample_count=None):
        if sample_count is None:
            sample_count = self.sample_count
        self._update_next_dialog_ids(sample_count)

    def eval(self, is_eval):
        pass

    def __len__(self):
        return len(self.next_phrases)

    def __iter__(self) -> Iterator[int]:
        return iter(self.next_phrases)
