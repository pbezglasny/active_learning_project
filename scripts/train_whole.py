import numpy as np
from datasets import DatasetDict
from datasets import load_metric
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers.integrations import TensorBoardCallback


# {TypeError}TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
def preprocess_function(examples):
    result = tokenizer(examples['dialog'], truncation=True, padding=True)
    result['labels'] = examples['act']
    return result


dataset = DatasetDict.load_from_disk('/home/pavel/work/active_learning_project/exploded_dataset')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

tokenized_dataset = dataset.map(preprocess_function)

f1 = load_metric('f1')


def compute_metrics(eval_preds):
    metric = load_metric('f1')
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='weighted')


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=500,
    save_steps=500,
    evaluation_strategy='steps'
)

writer = SummaryWriter(log_dir='/home/pavel/work/active_learning_project/logs')
tensorboard_callback = TensorBoardCallback(tb_writer=writer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[tensorboard_callback]
)

trainer.train()

model.save_pretrained('/home/pavel/work/active_learning_project/models/second.model')

# print(f1.compute())
