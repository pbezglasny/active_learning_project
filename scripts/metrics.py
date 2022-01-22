from dataclasses import dataclass
from typing import List, Dict, Any

from datasets import load_metric, Metric


@dataclass
class MetricResult:
    metric: str
    metric_alias: str
    value: float


class MetricConfig:

    def __init__(self, name: str, metric: Metric, kwargs: Dict[str, Any]):
        self.name = name
        self.metric = metric
        self.kwargs = kwargs

    def add_batch(self, predictions, references):
        self.metric.add_batch(predictions=predictions, references=references)

    def compute(self):
        metric_result = self.metric.compute(**self.kwargs)
        if len(metric_result) != 1:
            raise ValueError('Size of metric dict is not equal to 1')
        for k, v in metric_result.items():
            return MetricResult(k, self.name, v)

    @classmethod
    def load_metric(cls, metric_name, name_alias, compute_kwargs, load_kwargs=None):
        if load_kwargs is None:
            load_kwargs = {}
        metric = load_metric(metric_name, **load_kwargs)
        return MetricConfig(name_alias, metric, compute_kwargs)


class MetricConfigList:

    def __init__(self, metric_configs: List[MetricConfig]):
        self.metric_configs = metric_configs

    def add_batch(self, predictions, references):
        for metric in self.metric_configs:
            metric.add_batch(predictions=predictions, references=references)

    def compute(self):
        return [metric.compute() for metric in self.metric_configs]
