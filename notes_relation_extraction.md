## Spacy


https://github.com/explosion/projects/tree/v3/tutorials/rel_component/scripts
https://spacy.io/usage/layers-architectures#components-rel

- input: Nx2 entity pairs
  - for each entity, pool corresponding word vectors (usually they contain more than one)
  - then concatenate both word vectors
  - use both (head,tail) and (tail,head)!
  - TODO:
    - include entity classes
    - include relative distance
- output: NxR relation class probability matrix?
  - for each entity pair, guess relation
  - M*M matrix describing relation class probability?
- method
  - Linear + Logistic
  - https://github.com/explosion/projects/blob/v3/tutorials/rel_component/scripts/rel_model.py


- which loss?
  - mean square?!
    - https://github.com/explosion/projects/blob/dc6a4bee139a0b0cbf1614d2162f5e2abccb442e/tutorials/rel_component/scripts/rel_pipe.py#L142
  - would have expected categorical loss
- how is RE loss combined with NER loss?


# Shang 2022 - OneRel: Joint Entity and Relation Extraction with One Module in One Step

- input: (e_i, r_k, e_j) 
  - all possible combinations of token embeddings
- output: (L,K,L) matrix of triplet predictions
- method
  - instead of performing LxKxL calculation, split up as r^t (h*t)
  - they need only two layers of fully connected networks!

# final idea

- get embeddings
- perform NER (after all "Studium" may not even have a relation!)
- 

# how does NER work?!

https://huggingface.co/transformers/v3.0.2/_modules/transformers/modeling_auto.html#AutoModelForTokenClassification
https://github.com/huggingface/transformers/blob/088c1880b7bfd47777778d0d0fcc20e921bcf21e/src/transformers/models/distilbert/modeling_distilbert.py#L926

- look for "DistilBertForTokenClassification"
- embeddings >> dropout >> Linear layer with some hidden size
  - Linear layer
    - input: (n_tokens, n_embeddings)
    - output: (n_tokens, n_token_classes)
    - -> No connection between tokens anymore?!
- then uses CE-loss