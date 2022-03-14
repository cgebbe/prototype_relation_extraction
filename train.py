import transformers
import torch
import datetime


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.sentences = [
            "Haupt- und Nebensatz",
            "Mathematik- und Physikstudium",
        ]
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            # "bert-base-german-cased",
            # "distilbert-base-cased",
            "distilbert-base-german-cased",
            cache_dir=".cache",
        )

    def __getitem__(self, idx):
        """Returns dictionary with the following keys:

        - input_ids = [102, 1281, 232, 136, 2218, 865, 103]
        - attention_mask = [102, 1281, 232, 136, 2218, 865, 103]
        - labels = [102, 1281, 232, 136, 2218, 865, 103]
        """
        word = self.sentences[idx]
        item = self.tokenizer(word)
        num_tokens = len(item.encodings[0])
        item["labels"] = [1] * num_tokens
        return {k: torch.tensor(v) for k, v in item.items()}

    def __len__(self):
        return len(self.sentences)


ds = MyDataset()

output_dir = f"output/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
training_args = transformers.TrainingArguments(
    # --- how to train
    num_train_epochs=2,  # defaults to 3
    per_device_train_batch_size=1,  # defaults to 8
    gradient_accumulation_steps=8,  # defaults to 1
    learning_rate=5e-5,  # defaults to 5e-5
    lr_scheduler_type="constant",  # defaults to linear
    no_cuda=True,  # if GPU too small, see https://github.com/google-research/bert/blob/master/README.md#out-of-memory-issues
    # --- how to log
    output_dir=output_dir,
    logging_dir=output_dir + "/logs",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,  # delete any older checkpoint
)

model = transformers.AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-cased",
    num_labels=2,
    cache_dir=".cache",
    # gradient_checkpointing only works for bert, not for distilbert
    # gradient_checkpointing=True,  # see BertConfig and https://github.com/huggingface/transformers/blob/0735def8e1200ed45a2c33a075bc1595b12ef56a/src/transformers/modeling_bert.py#L461
)
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    eval_dataset=ds,
    # compute_metrics=create_metric_function(labels),
    # callbacks=[transformers.integrations.TensorBoardCallback()],  # already default
)
train_output = trainer.train()
d = 0
