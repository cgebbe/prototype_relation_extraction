from typing import List
from dataclasses import dataclass
from src.dataset.raw_dataset import DataSet


class LabelTokenError(Exception):
    pass


@dataclass
class TokenizedDataItem:
    tokens: List[int]
    attention_mask: List[int]
    token_classes: List[int]


class TokenizedDataSet:
    def __init__(self, dataset: DataSet, tokenizer, class_names) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer

        # Make it compatible with https://github.com/chakki-works/seqeval
        self.names = class_names
        self.name_per_class = {0: "O"}
        for name in class_names:
            self.name_per_class[self._get_B_class(name)] = f"B-{name.upper()}"
            self.name_per_class[self._get_I_class(name)] = f"I-{name.upper()}"

    def get_IOB_class_count(self):
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
        for label in item.entity_labels:
            try:
                start_token_idx = token_idx_per_starting_char_idx[label.start_idx]
                end_token_idx = token_idx_per_ending_char_idx[label.stop_idx]
            except KeyError:
                raise LabelTokenError("Label does not match tokenization!")

            # 0 = background, 1,3,5...=B-Labels, 2,4,5=I-Labels
            for token_idx in range(start_token_idx, end_token_idx + 1):
                if labels[token_idx] != 0:
                    raise LabelTokenError("Labels are overlapping")
                if token_idx == start_token_idx:
                    labels[token_idx] = self._get_B_class(label.class_names)
                else:
                    labels[token_idx] = self._get_I_class(label.class_names)

        return TokenizedDataItem(
            tokens=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            token_classes=labels,
        )

    def __len__(self):
        return len(self.dataset)