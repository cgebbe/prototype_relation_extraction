from datasets import load_metric

def create_metric_function(labels):
    """Creates metric
    Args:
        labels (List[str]): e.g. ['O', 'B_PER', 'I_PER']
    Returns:
        Callable[EvaluationOutput, Dict]: function to compute metric
    """
    # seqeval directly supports IOB, IOB2, see https://github.com/chakki-works/seqeval
    metric = load_metric("seqeval")

    def compute_metrics(p):
        y_true = p.label_ids
        y_pred = np.argmax(p.predictions, axis=2)  # (batch_size,512,num_labels)
        assert y_pred.shape == y_true.shape  # (batch_size,512)

        labels_true = utils.convert_classes_to_labels(y_true, y_true, labels)
        labels_pred = utils.convert_classes_to_labels(y_pred, y_true, labels)

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

    return compute_metrics




def convert_classes_to_labels(
    classes, classes_true, label_per_class
) -> List[List[str]]:
    """Converts class indexes to labels
    Args:
        classes (Tensor): (batch_size, token_length) tensor of class indexes
        classes_true (Tensor): (batch_size, token_length) tensor of true class indexes
        label_per_class (List[str]): list of labels per class index
    Return:
        List[List[str]]: Nested list of labels
    """
    return [
        [label_per_class[p] for (p, l) in zip(pred, true) if l != IGNORE_CLASS]
        for pred, true in zip(classes, classes_true)
    ]