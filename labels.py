class LabelError(Exception):
    pass


class Labels:
    def __init__(self, names) -> None:
        self.names = names

        # Make it compatible with https://github.com/chakki-works/seqeval
        self.name_per_class = {0: "O"}
        for name in names:
            self.name_per_class[self.get_B_class(name)] = f"B-{name.upper()}"
            self.name_per_class[self.get_I_class(name)] = f"I-{name.upper()}"

    def __len__(self):
        # each name has B_label and I_label + Background class
        return 1 + 2 * len(self.names)

    def get_B_class(self, name):
        return 1 + 2 * self.names.index(name)

    def get_I_class(self, name):
        return 1 + 2 * self.names.index(name) + 1

    def get_name_from_class(self, class_idx):
        return self.name_per_class[class_idx]

    def from_annotation_batch(self, annotation_batch, item):
        # construct mapping between chars and tokens
        num_tokens = len(item.encodings[0])
        token_idx_per_starting_char_idx = {}
        token_idx_per_ending_char_idx = {}
        for token_idx in range(num_tokens):
            try:
                span = item.token_to_chars(token_idx)
            except TypeError:
                # TypeError: type object argument after * must be an iterable, not NoneType
                continue
            token_idx_per_starting_char_idx[span.start] = token_idx
            token_idx_per_ending_char_idx[span.end] = token_idx

        # labels
        labels = [0] * num_tokens  # everything as background class initially
        for annotations in annotation_batch:
            for annotation in annotations["result"]:

                start_char = annotation["value"]["start"]
                end_char = annotation["value"]["end"]
                try:
                    start_token = token_idx_per_starting_char_idx[start_char]
                    end_token = token_idx_per_ending_char_idx[end_char]
                except KeyError:
                    raise LabelError("Labeling does not match tokenization!")

                assert len(annotation["value"]["labels"]) == 1
                # 0 = background, 1,3,5...=B-Labels, 2,4,5=I-Labels
                label_name = annotation["value"]["labels"][0]
                for token_idx in range(start_token, end_token + 1):
                    if token_idx == start_token:
                        labels[token_idx] = self.get_B_class(label_name)
                    else:
                        labels[token_idx] = self.get_I_class(label_name)

        return labels


LABELS = Labels(
    [
        "education_type",
        "education_topic",
    ]
)
