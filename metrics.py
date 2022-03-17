from typing import List
import numpy as np
from datasets import load_metric
from labels import LABELS
import pandas as pd
from transformers import EvalPrediction

# seqeval directly supports IOB, IOB2, see https://github.com/chakki-works/seqeval
metric = load_metric("seqeval")


def compute_metrics(p: EvalPrediction):
    # p.predictions.shape = (3,8,5) = (batch_size, max_token_count, num_classes)
    # p.label_ids.shape = (3,8)
    y_true = p.label_ids
    y_pred = np.argmax(p.predictions, axis=2)
    assert y_pred.shape == y_true.shape  # (batch_size,512)

    labels_true = convert_classes_to_labels(y_true, y_true)
    labels_pred = convert_classes_to_labels(y_pred, y_true)

    # understand KPI calculation
    if 0:
        _flatten_lst = lambda lst: [item for sublist in lst for item in sublist]
        df = pd.DataFrame(
            {
                "true": _flatten_lst(labels_true),
                "pred": _flatten_lst(labels_pred),
            }
        )

    results = metric.compute(
        predictions=labels_pred,
        references=labels_true,
        scheme="IOB2",  # see https://huggingface.co/datasets/conll2003
        mode="strict",
        # zero_division='',  # TODO, check seqeval
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def convert_classes_to_labels(classes, classes_true) -> List[List[str]]:
    """Converts class indexes to labels
    Args:
        classes (Tensor): (batch_size, token_length) tensor of class indexes
        classes_true (Tensor): (batch_size, token_length) tensor of true class indexes
        label_per_class (List[str]): list of labels per class index
    Return:
        List[List[str]]: Nested list of labels
    """
    IGNORE_CLASS = -100  # this seems to be added via padding?!
    # TODO: Could we also ignore the background class 0 ?! not sure...
    return [
        [
            LABELS.get_name_from_class(p)
            for (p, l) in zip(pred, true)
            if l != IGNORE_CLASS
        ]
        for pred, true in zip(classes, classes_true)
    ]
