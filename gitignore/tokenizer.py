from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(
    # "bert-base-german-cased",
    # "distilbert-base-cased",
    "distilbert-base-german-cased",
    cache_dir=".cache",
)

phrases = [
    "Haupt- und Nebensatz",
    "Mathe­matik- oder Physikstudium",
    "Informatik- oder Physikstudium",
    "Tischler- oder Kaufmannsausbildung",
]

for word in phrases[0:1]:
    item = tokenizer(word)
    print(item.tokens())

    if 1:
        # type(item) == BatchEncoding
        # type(item.encodings[0]) == tokens.Encoding !!!
        item.encodings[0]
        # Encoding(num_tokens=7, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
        item.encodings[
            0
        ].tokens  # ['[CLS]', 'Haupt', '-', 'und', 'Neben', '##satz', '[SEP]']

        # see https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/tokenizer#transformers.BatchEncoding
        item.tokens()  # ['[CLS]', 'Haupt', '-', 'und', 'Neben', '##satz', '[SEP]']
        item.word_ids()  # [None, 0, 1, 2, 3, 3, None]
        item.words()  # same as word_ids()
        num_tokens = len(item)

        word[1]  # yields 'a' from hAupt.
        item.char_to_word(1)  # yields 0, since first word
        item.char_to_token(1)  # yields 1, since second token (after [CLS])
