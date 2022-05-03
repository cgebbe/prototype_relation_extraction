"""
What are our relations?
- education_type [Ausbildung, Bachelor, ...] in education_topic [Informatik, Schweißen, ...]

General idea for relation extraction:
- we have a classifier (for each relation pair?!)
- for each pair of education_type and education_topic... (num_pairs, 2*768+1)
    where +1 represents relative position w.r.t. each other
- it predicts the probabiliy of that relation pair (num_entity_pairs,1),
    where the last axis represents the probability
- using as weights an matrix of (2*768+1, 1)

- what is the loss?!
- simple CE probably


Test that ...
    relations labels are loaded as expected
        for each token, we get an array of (num_pairs, 1)
    predicted text can be converted to (num_pairs, 2*768) matrix!
        
        RelationPair:
            from_token_index
            to_token_index
            from_embedding
            to_embedding
            relative_position
            probability

        entity_pairs[from_token_index][to_token_index] = ...

    classifier turns (num_pairs, 2*768) to (num_pairs, 1) 

    loss is calculated correctly
        CE normally
        only takes into account the pairs which are actually used!
"""
