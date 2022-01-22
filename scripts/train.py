from collections import defaultdict
import sys

import torch
from datasets import DatasetDict
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from scripts.data import WorstDialogSamplerWithRemoval, RandomSamplerWithRemoval
from scripts.metrics import MetricConfig, MetricConfigList
from scripts.trainer import Trainer
from scripts.utils import DialogCustomMetricCounter
import click
import json
from scripts.utils import EnhancedJSONEncoder
import logging

_logger = logging.getLogger('train')


def parse_list(s, separator=','):
    try:
        return [int(i) for i in s.split(separator)]
    except:
        raise ValueError(f"Can't parse string list '{s}'")


@click.command()
@click.option('--dataset', help='dataset location', type=str, required=True)
@click.option('--model', 'model_name', help='hugging face model name', type=str, required=True)
@click.option('--batch', 'batch_size', help='Batch size', default=32, type=int)
@click.option('--epochs', help='Comma separated list of number epochs for training', required=True)
@click.option('--percents', help='Comma separated list of number of percent to select at each epoch', required=True)
@click.option('--output', help='Output file location', type=str, required=True)
def main(dataset, model_name, batch_size, epochs, percents, output):
    epochs = parse_list(epochs)
    percents = parse_list(percents)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    f1_weighted = MetricConfig.load_metric('f1', 'f1 weighted', {'average': 'weighted'})
    f1_macro = MetricConfig.load_metric('f1', 'f1 macro', {'average': 'macro'})
    f1_micro = MetricConfig.load_metric('f1', 'f1 micro', {'average': 'micro'})
    accuracy = MetricConfig.load_metric('accuracy', 'accuracy', )

    metrics = MetricConfigList([f1_weighted, f1_macro, f1_micro, accuracy])

    dataset = DatasetDict.load_from_disk(dataset)

    eval_metric_kwargs = {'average': 'weighted'}

    whole_history = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for percent_of_data_at_epoch in percents:
        for num_epochs in epochs:
            _logger.info(f'Run train with {percent_of_data_at_epoch}% and {num_epochs} epochs')
            if num_epochs * percent_of_data_at_epoch > 100:
                continue

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

            dp = DialogCustomMetricCounter(f1_score, metric_kwargs=eval_metric_kwargs)

            size_of_data_at_epoch = len(dataset['train']) * percent_of_data_at_epoch // 100

            train_worst_sampler = WorstDialogSamplerWithRemoval(dataset['train'], dp, size_of_data_at_epoch, False)
            train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, sampler=train_worst_sampler)
            eval_dataloader = DataLoader(dataset['validation'], batch_size=batch_size)
            first_epoch_dataloader = DataLoader(dataset['test'], batch_size=batch_size)

            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
            model.to(device)

            trainer = Trainer(
                model,
                first_epoch_dataloader,
                train_dataloader,
                train_worst_sampler,
                dp,
                eval_dataloader,
                tokenizer,
                device,
                metrics,
                num_training_steps=500
            )

            hist = trainer.train(num_epochs)
            whole_history[num_epochs][percent_of_data_at_epoch]['active'] = hist

            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
            model.to(device)

            random_sampler = RandomSamplerWithRemoval(dataset['train'], size_of_data_at_epoch)
            train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, sampler=random_sampler)

            random_trainer = Trainer(
                model,
                first_epoch_dataloader,
                train_dataloader,
                train_worst_sampler,
                dp,
                eval_dataloader,
                tokenizer,
                device,
                metrics,
                num_training_steps=500
            )

            hist = random_trainer.train(num_epochs)
            whole_history[num_epochs][percent_of_data_at_epoch]['random'] = hist

    with open(output, 'wt') as f:
        json.dump(whole_history, f, cls=EnhancedJSONEncoder)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
    _logger.info(f'Start train with args {sys.argv[1:]}')
    main()
    _logger.info('Finish train')
