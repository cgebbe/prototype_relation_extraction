import transformers
import torch
from torch import nn
from typing import Optional

from .token_classifier import TokenClassifier


class CompositeOutput(transformers.file_utils.ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    token_class_logits: torch.FloatTensor = None
    # seq_relationship_logits: torch.FloatTensor = None
    # hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # attentions: Optional[Tuple[torch.FloatTensor]] = None


class CompositeModel(nn.Module):
    def __init__(
        self,
        base_model: transformers.modeling_utils.PreTrainedModel,
        num_token_classes: int,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.token_classifier = TokenClassifier(class_count=num_token_classes)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """Similar to DistilbertModel

        See https://huggingface.co/transformers/v2.11.0/_modules/transformers/modeling_distilbert.html#DistilBertModel

        To work with the huggingface trainer, all models need to...
        - return an instance of a subclass of ModelOutput or tuples (loss, logits)
        - compute a loss if "labels" argument is provided
        - can accept multiple label arguments if label_names is used in Training_Arguments
        See https://huggingface.co/docs/transformers/main_classes/trainer
        """
        base_output = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        embeddings = base_output.last_hidden_state

        # For the moment, use only
        output = self.token_classifier(embeddings, labels)

        # TODO: Add another head for entity relation
        return output
