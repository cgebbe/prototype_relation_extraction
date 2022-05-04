# About

This is a prototype for named entity recognition (a kind of token classification). In the future, I'd like to add entity linking (also known as relation extraction) to it.

![](banner.png)

Compared to a [previous implementation](https://github.com/cgebbe/prototype_ner_nobel_laureate), it features the following:

- a custom dataset loader (easy to expand to entity linking)
- a custom model (easy to add a head for entity linking)
- the ability to classify not only full words, but tokens within words
- more unit tests for a quicker development
- a better class structure

# How to install

See the [`Dockerfile`](Dockerfile). Upon opening the folder in VSCode, it should ask you to open it within a container.

# How to use

Simply execute...

- `python src.train.py` to run a training which overfits on 3 data items
- `pytest tests` to run all unit tests
