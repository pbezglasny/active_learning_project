from scripts.data import WorstDialogSampler
from scripts.utils import DialogPrediction
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
import torch


def _make_batch_data(batch, tokenizer, device,
                     **tokenizer_kwargs):
    batch_dict = {k: v for k, v in batch.items()}
    data = tokenizer(batch_dict['dialog'], **tokenizer_kwargs)
    data['labels'] = batch_dict['act']
    return {k: v.to(device) for k, v in data.items()}, batch['dialog_id']


class Trainer:

    def __init__(self, model, tokenizer, device, metric, metric_kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.metric = metric
        self.metric_kwargs = metric_kwargs

    def train(self, num_epochs, bottom_percents, batch_size):
        pass

    def evaluate(self, eval_dataloader, train_history):
        self.model.eval()
        for batch in eval_dataloader:
            data, dialog_ids = _make_batch_data(batch, self.tokenizer, self.device,
                                                truncation=True, padding=True,
                                                max_length=512,
                                                return_tensors='pt')
            outputs = self.model(**data)
            predictions = torch.argmax(outputs.logits, dim=-1)
            self.metric.add_batch(predictions=predictions, references=data['labels'])
        metric_value = self.metric.compute(**self.metric_kwargs)
        train_history['metrics'].append(metric_value)


def train(model,
          train_dataloader,
          train_sampler,
          eval_dataloader,
          num_training_steps,
          tokenizer,
          device,
          metric,
          bottom_percents=10,
          num_epochs=10,
          batch_size=32):
    dp = DialogPrediction()

    # train_worst_sampler = WorstDialogSampler(train_dataset, dp, bottom_percents)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_worst_sampler)
    # eval_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    # num_training_steps = (len(train_dataset) + len(train_dataset) * bottom_percents // 10 * (
    #         num_epochs - 1)) // batch_size
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
