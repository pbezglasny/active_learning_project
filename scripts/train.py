from scripts.data import WorstDialogSampler
from scripts.utils import DialogPrediction
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
import torch


def train(model,
          train_dataset,
          validation_dataset,
          tokenizer,
          device,
          metric,
          bottom_percents=10,
          num_epochs=10,
          batch_size=32):
    dp = DialogPrediction()

    train_worst_sampler = WorstDialogSampler(train_dataset, dp, bottom_percents)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_worst_sampler)
    eval_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_training_steps = (len(train_dataset) + len(train_dataset) * bottom_percents // 10 * (
            num_epochs - 1)) // batch_size
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
            batch_dict = {k: v for k, v in batch.items()}
            data = tokenizer(batch_dict['dialog'], truncation=True, padding=True, max_length=512, return_tensors='pt')
            data['labels'] = batch_dict['act']
            batch = {k: v.to(device) for k, v in data.items()}

            outputs = model(**batch)
            if not train_worst_sampler.is_init:
                predictions = torch.argmax(outputs.logits, dim=-1)
                for i in range(len(data['labels'])):
                    dp.add_answer(int(batch_dict['dialog_num'][i]), predictions[i] == data['labels'][i])
            loss = outputs.loss
            total_loss += float(loss)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        train_history['loss'].append(total_loss)

        if not train_worst_sampler.is_init:
            train_worst_sampler.choose_worst()

        model.eval()
        for batch in eval_dataloader:
            batch_dict = {k: v for k, v in batch.items()}
            data = tokenizer(batch_dict['dialog'], truncation=True, padding=True, max_length=512, return_tensors='pt')
            data['labels'] = batch_dict['act']
            batch = {k: v.to(device) for k, v in data.items()}

            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=predictions, references=data['labels'])
        metric_value = metric.compute(average='weighted')
        train_history['metrics'].append(metric_value)

    return train_history
