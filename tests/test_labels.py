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
from typing import List

import pytest
from src.dataset import DataSet, EntityLabel
from tests.conftest import TEST_DIR

from dataclasses import dataclass


LABEL_PATH = TEST_DIR / "data" / "phrases_with_hyphens.json"

def test_load_data():
    dataset = DataSet(LABEL_PATH)

    assert len(dataset) == 3

    item = dataset[0]
    assert item.text == "Tischler- oder Kaufmannausbildung"
    assert item.entity_labels[0] == EntityLabel(
        start_idx=0, stop_idx=8, entity_name="education_topic", text="Tischler"
    )


class LabelError(Exception):
    pass

@dataclass
class TokenizedDataItem:
    tokens: List[int]
    attention_mask: List[int]
    token_classes: List[int]

class TokenizedDataSet:
    def __init__(self, dataset: DataSet, tokenizer, entity_classes) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer

        # Make it compatible with https://github.com/chakki-works/seqeval
        self.names = entity_classes
        self.name_per_class = {0: "O"}
        for name in entity_classes:
            self.name_per_class[self._get_B_class(name)] = f"B-{name.upper()}"
            self.name_per_class[self._get_I_class(name)] = f"I-{name.upper()}"

    def __len__(self):
        # Background class + B_label and I_label for each name
        return 1 + 2 * len(self.names)

    def _get_B_class(self, name):
        return 1 + 2 * self.names.index(name)

    def _get_I_class(self, name):
        return 1 + 2 * self.names.index(name) + 1

    def _get_name_from_class(self, class_idx: int):
        return self.name_per_class[class_idx]

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # padding is only required if batch_size>1 during training or validation
        tokens = self.tokenizer(item.text, padding=False)

        if 1:
            tokens.token_to_chars(1)
            tokens.tokens()

        # construct mapping between chars and tokens
        num_tokens = len(tokens.encodings[0])
        token_idx_per_starting_char_idx = {}
        token_idx_per_ending_char_idx = {}
        for token_idx in range(num_tokens):
            try:
                char_span = tokens.token_to_chars(token_idx)
            except TypeError:
                # TypeError: type object argument after * must be an iterable, not NoneType
                continue
            token_idx_per_starting_char_idx[char_span.start] = token_idx
            token_idx_per_ending_char_idx[char_span.end] = token_idx

        # labels
        labels = [0] * num_tokens
        for entity_label in item.entity_labels:
            try:
                start_token_idx = token_idx_per_starting_char_idx[entity_label.start_idx]
                end_token_idx = token_idx_per_ending_char_idx[entity_label.stop_idx]
            except KeyError:
                raise LabelError("Label does not match tokenization!")

            # 0 = background, 1,3,5...=B-Labels, 2,4,5=I-Labels
            for token_idx in range(start_token_idx, end_token_idx + 1):
                if labels[token_idx] != 0:
                    raise LabelError("Labels are overlapping")
                if token_idx == start_token_idx:
                    labels[token_idx] = self._get_B_class(entity_label.entity_name)
                else:
                    labels[token_idx] = self._get_I_class(entity_label.entity_name)
        
        return TokenizedDataItem(
            tokens=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            token_classes=labels,
        )

    def __len__(self):
        return len(self.dataset)

        


def test_load_tokenized_data():
    dataset = DataSet(LABEL_PATH)
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            # "bert-base-german-cased",
            # "distilbert-base-cased",
            "distilbert-base-german-cased",
            # cache_dir=".cache",
        )

    tokenized_dataset = TokenizedDataSet(
        dataset=dataset,
        tokenizer=tokenizer,
        entity_classes=["education_type","education_topic"],
    )

    for i in [0,2]:
        tokenized_item = tokenized_dataset[i]
        assert len(tokenized_item.tokens) == len(tokenized_item.token_classes)
    
    import pytest
    with pytest.raises(LabelError, match="Label does not match tokenization"):
        tokenized_item = tokenized_dataset[1]

