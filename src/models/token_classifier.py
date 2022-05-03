import transformers
import torch
from torch import nn
from typing import Optional


class TokenClassifierOutput(transformers.file_utils.ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    token_class_logits: torch.FloatTensor = None
    # seq_relationship_logits: torch.FloatTensor = None
    # hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # attentions: Optional[Tuple[torch.FloatTensor]] = None


class TokenClassifier(nn.Module):
    """Similar to DistilBertForTokenClassification from huggingface.

    See https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/distilbert/modeling_distilbert.py#L926
    """

    def __init__(self, class_count, embedding_size=768, dropout_rate=0.5) -> None:
        super().__init__()
        self.class_count = class_count

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(embedding_size, class_count)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = self.dropout(x)
        logits = self.classifier(x)
        if labels is not None:
            loss = self.loss(
                logits.view(-1, self.class_count), labels.view(-1).to(dtype=int)
            )
        return TokenClassifierOutput(
            loss=loss if labels is not None else None,
            token_class_logits=logits,
        )
