"""
Test that model
"""
import transformers
import torch
from src.models.composite_model import CompositeModel

model = transformers.AutoModel.from_pretrained(
    "distilbert-base-cased",
    cache_dir=".cache",
)

dummy_input = {
    "input_ids": torch.IntTensor([[102, 5650, 522, 232, 394, 17853, 8292, 103]]),
    "attention_mask": torch.Tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
    "labels": torch.Tensor([[0, 3, 4, 0, 0, 3, 1, 0]]),
}


def test_base_model():
    output = model(
        **{k: v for k, v in dummy_input.items() if k in ["input_ids", "attention_mask"]}
    )

    batch_size, sequence_length = dummy_input["input_ids"].shape
    embedding_size = 768
    assert output.last_hidden_state.shape == (
        batch_size,
        sequence_length,
        embedding_size,
    )


def test_composite_model():
    num_token_classes = int(dummy_input["labels"].max() + 1)
    composite_model = CompositeModel(model, num_token_classes=num_token_classes)
    output = composite_model(**dummy_input)
    assert isinstance(output, transformers.file_utils.ModelOutput)
    assert {"loss", "token_class_logits"}.issubset(output.keys())
    assert output.loss.shape == ()
