from collections import defaultdict
import heapq
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod


class DialogStats:
    def __init__(self):
        self.correct_ans = 0
        self.total_ans = 0

    def add_ans(self, correct):
        self.total_ans += 1
        if correct:
            self.correct_ans += 1

    @property
    def ratio(self):
        if self.total_ans == 0:
            return 0
        else:
            return self.correct_ans / self.total_ans

    def __repr__(self):
        return f'{self.correct_ans}/{self.total_ans}'


class AbstractDialogMetricCounter(ABC):

    @abstractmethod
    def get_bottom_k_values(self, k):
        pass

    @abstractmethod
    def get_bottom_k_percents(self, k):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def add_answer(self, **kwargs):
        pass


class DialogMetricCounter(AbstractDialogMetricCounter):
    def __init__(self):
        self.answers = None
        self.reset()

    def reset(self):
        self.answers = defaultdict(lambda: DialogStats())

    def add_answer(self, dialog_id, correct):
        self.answers[dialog_id].add_ans(correct)

    def get_bottom_k_values(self, k):
        if k < 1:
            raise ValueError(f'Value of k should be greater than {0}')
        result_count = k
        answer = []
        for key, v in self.answers.items():
            if len(answer) < result_count:
                heapq.heappush(answer, (-v.ratio, key))
            else:
                prev_ratio, dialog_id = heapq.heappop(answer)
                if prev_ratio > -v.ratio:
                    heapq.heappush(answer, (prev_ratio, dialog_id))
                else:
                    heapq.heappush(answer, (-v.ratio, key))
        return [dialog_id for _, dialog_id in answer]

    def get_bottom_k_percents(self, k):
        result_count = len(self.answers) * k // 100
        result_count = max(result_count, 1)
        return self.get_bottom_k_values(result_count)

    def __repr__(self):
        return str(self.answers)


class DialogCustomMetricCounter(AbstractDialogMetricCounter):

    def __init__(self, metric=accuracy_score, metric_kwargs=None):
        if metric_kwargs is None:
            metric_kwargs = {}
        self.actual_answers = None
        self.predicted_answers = None
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.reset()

    def reset(self):
        self.actual_answers = defaultdict(list)
        self.predicted_answers = defaultdict(list)

    def add_answer(self, dialog_id, actual, predicted):
        self.actual_answers[dialog_id].append(actual)
        self.predicted_answers[dialog_id].append(predicted)

    def add_batch(self, dialog_id, actual, predicted):
        self.actual_answers[dialog_id] = actual
        self.predicted_answers[dialog_id] = predicted

    def get_bottom_k_values(self, k):
        if k < 1:
            raise ValueError(f'Value of k should be greater than {0}')
        if len(self.actual_answers) != len(self.predicted_answers):
            raise ValueError(
                f'Size of actual answers is not equal to size'
                f' of predicted, given {len(self.actual_answers)} '
                f'and {len(self.predicted_answers)}')
        answer = []
        result_count = k
        for key in self.actual_answers.keys():
            if key not in self.predicted_answers:
                raise ValueError(f'Key {key} does not appear in predicted dict')
            actual = self.actual_answers[key]
            pred = self.predicted_answers[key]
            metric_value = -self.metric(actual, pred, **self.metric_kwargs)
            if len(answer) < result_count:
                heapq.heappush(answer, (metric_value, key))
            else:
                prev_ratio, dialog_id = heapq.heappop(answer)
                if prev_ratio > metric_value:
                    heapq.heappush(answer, (prev_ratio, dialog_id))
                else:
                    heapq.heappush(answer, (metric_value, key))
        return [dialog_id for _, dialog_id in answer]

    def get_bottom_k_percents(self, k):
        result_count = len(self.actual_answers) * k // 100
        result_count = max(result_count, 1)
        return self.get_bottom_k_values(result_count)
