import torch
from transformers import AdamW
from transformers import get_scheduler

from scripts.data import AbstractWorstDialogSampler
from scripts.utils import AbstractDialogMetricCounter
from scripts.metrics import MetricConfigList


def _make_batch_data(batch, tokenizer, device,
                     **tokenizer_kwargs):
    batch_dict = {k: v for k, v in batch.items()}
    data = tokenizer(batch_dict['dialog'], **tokenizer_kwargs)
    data['labels'] = batch_dict['act']
    return {k: v.to(device) for k, v in data.items()}, batch['dialog_id']


class Trainer:

    def __init__(self,
                 model,
                 first_epoch_dataloader,
                 train_dataloader,
                 train_sampler: AbstractWorstDialogSampler,
                 dp: AbstractDialogMetricCounter,
                 eval_dataloader,
                 tokenizer, device,
                 metrics: MetricConfigList,
                 dialog_metric_counter_kwargs=None,
                 **kwargs):
        if dialog_metric_counter_kwargs is None:
            dialog_metric_counter_kwargs = {}
        self.model = model
        self.first_epoch_dataloader = first_epoch_dataloader
        self.train_dataloader = train_dataloader
        self.train_sampler = train_sampler
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer
        self.device = device
        self.metrics = metrics
        self.dp = dp
        self.optimizer = AdamW(model.parameters(), lr=5e-5)
        self.dialog_metric_counter_kwargs = dialog_metric_counter_kwargs
        self.init(**kwargs)

    def init(self, **kwargs):
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=kwargs['num_training_steps']
        )

    def train(self, num_epochs):
        train_history = {'loss': [], 'metrics': []}
        for epoch in range(num_epochs):
            self.at_each_epoch_step(epoch, train_history)
            self.at_epoch_end(epoch, train_history)
            self.evaluate(epoch, train_history)
        return train_history

    def at_each_epoch_step(self, epoch, train_history):
        self.model.train()
        total_loss = 0
        self.train_sampler.eval(False)
        dataloader = self.train_dataloader
        if epoch == 0:
            dataloader = self.first_epoch_dataloader
        for batch in dataloader:
            data, dialog_ids = _make_batch_data(batch, self.tokenizer, self.device,
                                                truncation=True, padding=True,
                                                max_length=512,
                                                return_tensors='pt')
            outputs = self.model(**data)
            loss = outputs.loss
            total_loss += float(loss)
            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        print(total_loss)
        train_history['loss'].append(total_loss)

    def at_epoch_end(self, epoch_num, train_history):
        self.train_sampler.eval(True)
        self.model.eval()
        self.dp.reset()
        for batch in self.train_dataloader:
            data, dialog_ids = _make_batch_data(batch, self.tokenizer, self.device,
                                                truncation=True, padding=True,
                                                max_length=512,
                                                return_tensors='pt')
            outputs = self.model(**data)
            predictions = torch.argmax(outputs.logits, dim=-1)
            for i in range(len(data['labels'])):
                self.dp.add_answer(dialog_id=int(dialog_ids[i]), actual=int(data['labels'][i]),
                                   predicted=int(predictions[i]))
        self.train_sampler.update_source_after_epoch(**self.dialog_metric_counter_kwargs)

    def evaluate(self, epoch, train_history):
        self.model.eval()
        for batch in self.eval_dataloader:
            data, dialog_ids = _make_batch_data(batch, self.tokenizer, self.device,
                                                truncation=True, padding=True,
                                                max_length=512,
                                                return_tensors='pt')
            outputs = self.model(**data)
            predictions = torch.argmax(outputs.logits, dim=-1)
            self.metrics.add_batch(predictions=predictions, references=data['labels'])

        metric_values = self.metrics.compute()
        print(metric_values)
        train_history['metrics'].append(metric_values)
