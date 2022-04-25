"""
Next tests:

test_load_data -> also load relation data

test token classification
    convert text to tokens -> array of numbers
    convert entity labels -> array of class indexes (same length)

test TokenClassificationDataSet(DataSet)
    DataItem:
        tokens: array of numbers
        token_classes: array of class indexes (one-hot or prediction?! Not sure..)
"""
import pytest
from src.dataset.torch_dataset import TorchDataset
from src.dataset.tokenized_dataset import TokenizedDataSet, LabelTokenError
from src.dataset.raw_dataset import DataSet, EntityLabel
from tests.conftest import TEST_DIR

import torch
import transformers


LABEL_PATH = TEST_DIR / "data" / "phrases_with_hyphens.json"


def test_load_data():
    dataset = DataSet(LABEL_PATH)

    assert len(dataset) == 3

    item = dataset[0]
    assert item.text == "Tischler- oder Kaufmannausbildung"
    assert item.entity_labels[0] == EntityLabel(
        start_idx=0, stop_idx=8, class_names="education_topic", text="Tischler"
    )


def test_load_tokenized_data():
    dataset = DataSet(LABEL_PATH)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        # "bert-base-german-cased",
        # "distilbert-base-cased",
        "distilbert-base-german-cased",
        cache_dir=".cache",
    )

    tokenized_dataset = TokenizedDataSet(
        dataset=dataset,
        tokenizer=tokenizer,
        class_names=["education_type", "education_topic"],
    )

    for i in [0, 2]:
        tokenized_item = tokenized_dataset[i]
        assert len(tokenized_item.tokens) == len(tokenized_item.token_classes)

    with pytest.raises(LabelTokenError, match="Label does not match tokenization"):
        tokenized_item = tokenized_dataset[1]


def test_load_torch_data():
    dataset = DataSet(LABEL_PATH)
    tokenized_dataset = TokenizedDataSet(
        dataset=dataset,
        tokenizer=transformers.AutoTokenizer.from_pretrained(
            "distilbert-base-german-cased",
            cache_dir=".cache",
        ),
        class_names=["education_type", "education_topic"],
    )
    torch_dataset = TorchDataset(tokenized_dataset=tokenized_dataset)


    for i in range(len(torch_dataset)):
        torch_dict = torch_dataset[i]
        for k,v in torch_dict.items():
            assert isinstance(v, torch.Tensor)


