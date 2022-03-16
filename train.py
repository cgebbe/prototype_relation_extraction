import transformers
import torch
import datetime
import json
from pathlib import Path


def _replace_soft_hyphens(s):
    # Somehow, the exported JSON contains soft hyphens to indiciate possible line breaks.
    # Simply replace them, see https://stackoverflow.com/a/51976543/2135504
    return s.replace("\xad", "")


LABELS = [
    "education_type",
    "education_topic",
]
NUM_LABELS = 1 + 2 * len(LABELS)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: Path):
        assert data_path.exists()
        with open(data_path, encoding="utf-8") as f:
            content = f.read()
        # content = _replace_soft_hyphens(content)  # if I replace soft-hyphens, value does not match anymore :/
        self.data = json.loads(content)

        self.sentences = [
            "Haupt- und Nebensatz",
            "Mathematik- und Physikstudium",
        ]
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            # "bert-base-german-cased",
            # "distilbert-base-cased",
            "distilbert-base-german-cased",
            # cache_dir=".cache",
        )

    def __getitem__(self, idx):
        """Returns dictionary with the following keys:

        - input_ids = [102, 1281, 232, 136, 2218, 865, 103]
        - attention_mask = [1,1,1,1,1,1]
        - labels = [102, 1281, 232, 136, 2218, 865, 103]
        """
        word = self.data[idx]["data"]["text"]
        item = self.tokenizer(word)

        # construct mapping between chars and tokens
        num_tokens = len(item.encodings[0])
        token_idx_per_starting_char_idx = {}
        token_idx_per_ending_char_idx = {}
        for token_idx in range(num_tokens):
            try:
                span = item.token_to_chars(token_idx)
            except TypeError:
                # TypeError: type object argument after * must be an iterable, not NoneType
                continue
            token_idx_per_starting_char_idx[span.start] = token_idx
            token_idx_per_ending_char_idx[span.end] = token_idx

        # labels
        labels = [0] * num_tokens  # everything as background class initially
        annotation_batches = self.data[idx]["annotations"]
        for annotations in annotation_batches:
            for annotation in annotations["result"]:

                start_char = annotation["value"]["start"]
                end_char = annotation["value"]["end"]
                try:
                    start_token = token_idx_per_starting_char_idx[start_char]
                    end_token = token_idx_per_ending_char_idx[end_char]
                except KeyError:
                    raise ValueError("Labeling does not match tokenization!")

                assert len(annotation["value"]["labels"]) == 1
                # 0 = background, 1,3,5...=B-Labels, 2,4,5=I-Labels
                label = annotation["value"]["labels"][0]
                label_B_class = 1 + 2 * LABELS.index(label)
                label_I_class = 1 + 2 * LABELS.index(label) + 1
                for token_idx in range(start_token, end_token + 1):
                    if token_idx == start_token:
                        labels[token_idx] = label_B_class
                    else:
                        labels[token_idx] = label_I_class

        item["labels"] = labels
        return {k: torch.tensor(v) for k, v in item.items()}

    def __len__(self):
        return len(self.data)


ds = MyDataset(data_path=Path("data/phrases_with_hyphens.json"))

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
    num_labels=NUM_LABELS,
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
