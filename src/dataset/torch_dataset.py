from .tokenized_dataset import TokenizedDataSet, LabelTokenError, TokenizedDataItem
import torch
import random


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset: TokenizedDataSet):
        self.tokenized_dataset = tokenized_dataset

    def __getitem__(self, idx):
        """Returns dictionary with the following keys:

        - input_ids = [102, 1281, 232, 136, 2218, 865, 103]
        - attention_mask = [1,1,1,1,1,1]
        - labels = [0,0,1,2,0,0,]
        """
        item = None
        while item is None:
            try:
                item = self.tokenized_dataset[idx]
            except LabelTokenError:
                # try any other random item. (empty items seem rather difficult)
                idx = random.randint(0, len(self) - 1)

        assert isinstance(item, TokenizedDataItem)
        out = {
            "input_ids": item.tokens,
            "attention_mask": item.attention_mask,
            "token_class_labels": item.token_classes,
        }
        return {k: torch.tensor(v) for k, v in out.items()}

    def __len__(self):
        return len(self.tokenized_dataset)
