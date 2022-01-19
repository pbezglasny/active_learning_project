import torch
from transformers import AdamW
from transformers import get_scheduler

from scripts.data import AbstractWorstDialogSampler
from scripts.utils import DialogMetricCounter, AbstractDialogMetricCounter


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
                 metric,
                 metric_kwargs,
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
        self.metric = metric
        self.metric_kwargs = metric_kwargs
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
            self.metric.add_batch(predictions=predictions, references=data['labels'])
        metric_value = self.metric.compute(**self.metric_kwargs)
        print(metric_value)
        train_history['metrics'].append(metric_value)


def train(model,
          train_dataloader,
          train_sampler,
          eval_dataloader,
          num_training_steps,
          tokenizer,
          device,
          metric,
          num_epochs=10):
    dp = DialogMetricCounter()

    optimizer = AdamW(model.parameters(), lr=5e-5)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    train_history = {'loss': [], 'metrics': []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            data, dialog_ids = _make_batch_data(batch, tokenizer, device,
                                                truncation=True, padding=True,
                                                max_length=512,
                                                return_tensors='pt')
            outputs = model(**data)
            if not train_sampler.is_init:
                predictions = torch.argmax(outputs.logits, dim=-1)
                for i in range(len(data['labels'])):
                    dp.add_answer(int(dialog_ids[i]), predictions[i] == data['labels'][i])
            loss = outputs.loss
            total_loss += float(loss)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_history['loss'].append(total_loss)

        if not train_sampler.is_init:
            train_sampler.update_source_after_epoch()

        model.eval()
        for batch in eval_dataloader:
            data, dialog_ids = _make_batch_data(batch, tokenizer, device,
                                                truncation=True, padding=True,
                                                max_length=512,
                                                return_tensors='pt')
            outputs = model(**data)
            predictions = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=predictions, references=data['labels'])
        metric_value = metric.compute(average='weighted')
        train_history['metrics'].append(metric_value)

    return train_history
