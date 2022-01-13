from datasets import load_dataset
from transformers import BertModel, BertConfig
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoModel
from transformers import TrainingArguments, Trainer, BatchEncoding
from transformers import DataCollatorWithPadding


def make_dialog(s):
    return ''.join(map(lambda x: f'-{x}', s))


# {TypeError}TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
def preprocess_function(examples):
    # print(''.join(map(lambda x: f'-{x}', examples['dialog'][0])))
    # tokenizer(examples['dialog'], is_split_into_words=True)
    # dialogs = list(map(make_dialog, examples['dialog']))
    # print(dialogs[0])
    # data = tokenizer(dialogs, truncation=True)
    # sentences=
    # data = tokenizer(examples['dialog'], truncation=True)
    # [tokenizer(dialog, truncation=True) for dialog in dialogs]
    # data['labels'] = examples['act']
    # data['labels'] = list(map(lambda x: x[0], examples['act']))
    # result = BatchEncoding()
    # data = [tokenizer(examples['dialog'][i]) for i in range(len(examples['dialog']))]
    # labels = [examples['act'][i] for i in range(len(examples['act']))]
    # input_ids = [data[i]['input_ids'] for i in range(len(data))]
    # token_type_ids = [data[i]['token_type_ids'] for i in range(len(data))]
    # attention_mask = [data[i]['attention_mask'] for i in range(len(data))]
    # return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask,
    #         'labels': labels}
    result = tokenizer(examples['dialog'], truncation=True, padding=True)
    result['label'] = examples['act']
    return result


def make_classification_sentences(examples):

    return examples


dataset = load_dataset("daily_dialog")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

auto_model = AutoModel.from_pretrained('bert-base-uncased')

# class_sentences = dataset.map(make_classification_sentences)

tokenized_dataset = dataset.map(preprocess_function)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    # per_device_train_batch_size=16,
    # per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
)


class DialogTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)

        i = 1
        return 1


trainer = DialogTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
