import torch
from datasets import DatasetDict
from datasets import load_metric
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from scripts.data import WorstDialogSamplerWithRemoval, RandomSamplerWithRemoval
from scripts.train import Trainer
from scripts.utils import DialogCustomMetricCounter
from collections import defaultdict
import json

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
f1 = load_metric('f1')

dataset = DatasetDict.load_from_disk('/home/pavel/work/active_learning_project/exploded_dataset')

model_name = 'bert-base-uncased'
# num_epochs = 2
# percent_of_data_at_epoch = 10
batch_size = 32
eval_metric_kwargs = {'average': 'weighted'}

whole_history = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

percents = [5, 10, 15, 20]
epochs = [5, 7, 10]
for percent_of_data_at_epoch in percents:
    for num_epochs in epochs:
        print(percent_of_data_at_epoch, num_epochs)
        if num_epochs * percent_of_data_at_epoch > 100:
            continue

        tokenizer = AutoTokenizer.from_pretrained(model_name)

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
            f1,
            eval_metric_kwargs,
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
            f1,
            eval_metric_kwargs,
            num_training_steps=500
        )

        hist = random_trainer.train(num_epochs)
        whole_history[num_epochs][percent_of_data_at_epoch]['random'] = hist

with open('bert-base-uncased.json', 'wt') as f:
    json.dump(whole_history, f)
