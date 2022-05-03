from src.dataset.torch_dataset import TorchDataset
from src.dataset.tokenized_dataset import TokenizedDataSet
from src.dataset.raw_dataset import DataSet
from src.models.composite_model import CompositeModel

import transformers
import datetime
from pathlib import Path

from metrics import MetricCalculator


LABEL_PATH = Path("data/phrases_with_hyphens.json")
CLASS_NAMES = ["education_type", "education_topic"]

# setup dataset
dataset = DataSet(LABEL_PATH)
tokenized_dataset = TokenizedDataSet(
    dataset=dataset,
    tokenizer=transformers.AutoTokenizer.from_pretrained(
        "distilbert-base-german-cased",
        cache_dir=".cache",
    ),
    class_names=CLASS_NAMES,
)
ds = TorchDataset(tokenized_dataset=tokenized_dataset)

# setup model
base_model = transformers.AutoModel.from_pretrained(
    "distilbert-base-cased",
    cache_dir=".cache",
    # gradient_checkpointing only works for bert, not for distilbert
    # gradient_checkpointing=True,  # see BertConfig and https://github.com/huggingface/transformers/blob/0735def8e1200ed45a2c33a075bc1595b12ef56a/src/transformers/modeling_bert.py#L461
)
composite_model = CompositeModel(
    base_model=base_model,
    num_token_classes=tokenized_dataset.get_IOB_class_count(),
)

# setup trainer
metric_calculator = MetricCalculator(
    get_name_from_class=tokenized_dataset.get_name_from_class
)
output_dir = f"output/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
training_args = transformers.TrainingArguments(
    # --- how to train
    num_train_epochs=20,  # defaults to 3
    per_device_train_batch_size=1,  # defaults to 8
    gradient_accumulation_steps=8,  # defaults to 1
    per_device_eval_batch_size=1,
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
trainer = transformers.Trainer(
    model=composite_model,
    args=training_args,
    train_dataset=ds,
    eval_dataset=ds,
    compute_metrics=metric_calculator.compute_metrics,
    # callbacks=[transformers.integrations.TensorBoardCallback()],  # already default
)
train_output = trainer.train()
d = 0
