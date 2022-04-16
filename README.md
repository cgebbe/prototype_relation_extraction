# TODO

- [x] NER with custom JSON loader and word-parts
- [ ] Add relations!
  - [ ] Add label loading
  - [ ] Add inference model somehow

# Notes

## Rebuild docker locally

docker build -it --rm .

# Questions

### Extraction relationships between entities (since we already did NER)

https://github.com/explosion/projects/tree/v3/tutorials/rel_component/scripts
- not 100% sure how supposed to work
- they create 2d-matrix of all span combinations (span has 1 or more tokens)
- perform pooling (so that span with multiple tokens reduced)
- apply a linear classifier on top of it.
- No context at all? Well, indirectly, via output tensors from BERT, but no position...

https://huggingface.co/yseop/distilbert-base-financial-relation-extraction

https://towardsdatascience.com/how-to-train-a-joint-entities-and-relation-extraction-classifier-using-bert-transformer-with-spacy-49eb08d91b5c

https://towardsdatascience.com/bert-s-for-relation-extraction-in-nlp-2c7c3ab487c4
- uses output tensors from BERT for two entities (2x768 inputs)
- adds a Linear + logistic classifiation layer (1xnum_labels output)

https://medium.com/@andreasherman/different-ways-of-doing-relation-extraction-from-text-7362b4c3169e
- Rule-based ...
- supervised RE: binary classifier to determine if sentence expresses relation
- not really helpful article, too high level. Already clear that I do supervised RE.

https://towardsdatascience.com/extracting-relations-among-entities-using-nlp-b3d773c709ce
- only math theory

https://paperswithcode.com/task/relation-extraction
- **some papers with code :)**
- e.g. [REBEL](https://aclanthology.org/2021.findings-emnlp.204.pdf) - bit different task, but has nice summary
  - > Early approaches tackled RE as a pipeline system, identifying the entities present in the text using Named Entity Recognition, and then classifying the relation, or lack of, between each pair of entities present in the text
    - most recent example: https://aclanthology.org/2020.emnlp-main.523.pdf
  - > Finally, there are pipeline systems that tackle both parts of Relation Extraction, NER, and RC. In these setups, entities are first extracted and then a classifier extracts relations sharing part of the encoder.
- also [PL-Marker](https://paperswithcode.com/paper/pack-together-entity-and-relation-extraction#code)

https://stackoverflow.com/questions/15260212/nlp-to-find-relationship-between-entities
- latest link from 2018, though?!
- rather google "entity linking"?

http://nlpprogress.com/english/entity_linking.html
- maybe https://arxiv.org/pdf/2006.01969.pdf ?

=> General take-away
- extract (pooled) output-tensors from all entities
- 
- add loss - how?!


## How to label [CLS] and background? 0 or ignore?

from https://github.com/cgebbe/prototype_ner_nobel_laureate/blob/main/utils.py
> if word is split into multiple tokens, only label first token 
Does that really make sense?!

I think current behavior makes sense too...

## How to specify our own huggingface dataset? Do we even need one?

Ah, seems to make sense, since eventually used in transformers.Trainers
```python
raw_datasets = load_dataset("dataset.py")  # defines input files (including labels)
ds = raw_datasets.map(utils.preprocess, batched=True)  # tokenize (and adapt labels)
ds_train = ds["train"].shuffle(seed=42)
trainer = transformers.Trainer(train_dataset=ds_train, ...)
```

- However, for NER we used the [datasets](https://pypi.org/project/datasets/)-package
- In contrast, the [huggingface documentation](https://huggingface.co/transformers/v3.2.0/custom_datasets.html) describes simple torch-datasets. Maybe this is simpler?! It requires to only return a __getitem__ function returning a dict AFTER tokenization with likely the following keys:
  - input_ids
  - token_type_ids
  - attention_mask
  - labels (as numbers, e.g. 0,1,...)


## Is the IOB2 really necessary?

Ah, how else to identify whether a word just starts or is a continuation?! So yes, probably necessary :/. However, can do this also within the loader probably?!


## Can we deal with labeling word **parts**?
> Mathe­matik- oder Physikstudium

- We can label word parts in label studio
- export as ConLL reduces it to word labels, though.
- export as JSON works. (simply marks characters)
- how to load that as JSON labels?

Test the following cases
- labeling matches tokenization -> works
- labeling does not match tokenization -> error

```bash
# using bert-base-german-cased
['[CLS]', 'Mathematik', '-', 'oder', 'Physik', '##studium', '[SEP]']
['[CLS]', 'Informatik', '-', 'oder', 'Physik', '##studium', '[SEP]']
['[CLS]', 'Tisch', '##ler', '-', 'oder', 'Kaufmanns', '##ausbildung', '[SEP]']

# using distilbert-base-german-cased. Note separation at Kaufmanns-!
['[CLS]', 'Mathematik', '-', 'oder', 'Physik', '##studium', '[SEP]']
['[CLS]', 'Informatik', '-', 'oder', 'Physik', '##studium', '[SEP]']
['[CLS]', 'Tisch', '##ler', '-', 'oder', 'Kaufmann', '##sau', '##sb', '##ild', '##ung', '[SEP]']
```

## Maybe need our own de-hyphener?! Go through sentences
> Wirt­schafts­wissen­schaften/-infor­matik/-mathe­matik
> (Wirtschafts-)Informatik 

## How to label?

```bash
# start label studio
LOCAL_DIR="/mnt/sda1/projects/git/prototypes/202203_relation_extraction/data/label_studio"
CONTAINER_DIR="/label-studio/data"
IMAGE="heartexlabs/label-studio:1.4.0"
docker run -it -p 8080:8080 -v $LOCAL_DIR:$CONTAINER_DIR $IMAGE
```

setup relation extraction
https://labelstud.io/templates/relation_extraction.html#main

```xml
<View>
   <Relations>
    <Relation value="org:founded_by"/>
    <Relation value="org:founded"/>
  </Relations>
  <Labels name="label" toName="text">
    <Label value="Organization" background="orange"/>
    <Label value="Person" background="green"/>
    <Label value="Datetime" background="blue"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>
```


## Which model to use?

- German distillbert
  - https://huggingface.co/distilbert-base-german-cased


# Challenging sentences

## How to deal with "Ergänzungsstrich"?

- zum Maschinen- und Anlagenführer
- Regenmäntel und -schirme
- Ein- und Ausgänge

See also [Bachelorarbeit from Ruoff](file:///home/cgebbe/Desktop/Bachelorarbeit_Ruoff.pdf), page 5:
>  Es wird unterschieden zwischen Phrasen mit Erganzungsstrich, wie ,,Haupt- und Nebensatz“, und solchen mit Bindestrich, wie ¨in ,,Preis-Leistungs-Verhaltnis“. Erstere werden in mehrere Token unterteilt, zweitere bilden ein einzelnes.

This is part of "lexical analysis" or "Segmentierung"

## Is there a common German NLP preprocessing toolkit?

- https://www.google.com/search?q=german+nlp+preprocessing&oq=german+nlp+preprocessing&aqs=chrome..69i64j0i22i30.4679j1j7&sourceid=chrome&ie=UTF-8
- https://flavioclesio.com/a-small-journey-in-the-valley-of-natural-language-processing-and-text-pre-processing-for-german-language
- https://adrien.barbaresi.eu/blog/using-a-rule-based-tokenizer-for-german.html
- https://github.com/adbar/German-NLP
- see also my question from ~6months ago on data science: https://datascience.stackexchange.com/questions/100614/how-to-expand-lists
- https://richstone.github.io/blog/best-german-tokenizer/

List from https://github.com/adbar/German-NLP
- Tokenization
  - deep-eos = only for End Of Sentences
  - JTok = java-based tokenizer
  - nnsplit = supports compound splitting for German (mainly for sentence boundary detection)
  - SoMaJo = tokenizer from 2015 
  - syntok = **seems promising!**
  - waste = from 2013in C++
  - german-abbreviations = list of abbreviations

Asked question here:
https://datascience.stackexchange.com/questions/108829/how-to-deal-with-erg%c3%a4nzungsstrichen-and-bindestrichen-in-german-nlp


## How does spacy's "de_core_news_sm" tokenizer  work?

Try out https://spacy.io/models/de#de_core_news_sm
- tok2vec = token to vector (shouldn't this come before?!)
- tagger = POS tagging
- morphologizer = predicts morphological features and coarse-grained UPOS tags?!
- DependencyParser = jointly learns sentence segmentation AND dependendy parsing
- SentenceRecognizer = segments sentences
- attribute_ruler =  ?!
- lemmatizer = shortens tokens to base form
- NER = know this

Result: Big difference between "Haupt- und Nebensatz" and "Hauptsatz und Nebensatz".

### Use Spacy docker

#### version 2
https://hub.docker.com/r/jgontrum/spacyapi/tags
docker pull jgontrum/spacyapi:all_v2
docker run -p "127.0.0.1:8080:80" jgontrum/spacyapi:en_v2

#### version 3

> spaCy v3.0 features all new transformer-based pipelines

https://hub.docker.com/r/bbieniek/spacyapi/
only 600 pulls?!

```bash
IMAGE=bbieniek/spacyapi:all_v3
CONTAINER=cg_spacy3
docker run --rm -it --entrypoint bin/bash $IMAGE

# create container, start and attach (-t for pseudo-terminal, otherwise stops)
docker create -t --entrypoint "bin/bash" --name $CONTAINER $IMAGE

# -i for interactive (STDIN), -a for attach (STDOUT/STDERR)
docker start $CONTAINER  
docker attach $CONTAINER

cd /app
source env/bin/activate other
```

## How do other tokenizers (e.g. BERT) work on Ergänzungsstriche?

https://spacy.io/api/tokenizer
- tokenization is really only segmentation into parts
- spacy correctly creates all tokens (=individual elements)
- The issue is rather later at vectorization!

https://www.deepset.ai/german-bert
- This one works rather beautifully
- it splits words in half in an expected way
- it produces very similar embeddings for "Haupt- und Nebensatz" and "Hauptsatz und Nebensatz"
- see my answer at https://datascience.stackexchange.com/a/108862/68908


## TODO: How to do label word PARTS?

Example sentences: "Mathematik- oder Physikstudium"
- is split into tokens correctly using bert-base-german-cased
- desired outcome
  - classify tokens
    - Physik
    - Mathematik
    - Studium ("part of word!!!")
  - identify relation
    - Studium -> Mathematik
    - Studium -> Physik

Hmm... we don't need to split the sentences so far, but maybe later.

## How to deal with "oder vergleichbare..."?

> Abgeschlossenes Masterstudium der Fachrichtung Elektrotechnik, Mechatronik, Fahrzeugtechnik oder vergleichbare Ausbildung
desired outcome:
- Ausbildung IN Elektrotechnik, Mechatronik, Fahrzeugtechnik (doesn't exist, but would need to be mapped to typical Ausbildungsberufe)

> Eine erfolgreich abgeschlossene Ausbildung zum Fachinformatiker oder vergleichbare IT-Ausbildung bzw. Berufserfahrung in ähnlicher Funktion.
desired outcome:
- Ausbildung IN Fachinformatiker
- Ausbildung IN IT
- Berufserfahrung IN "ähnlicher Funktion" -> replace with name of job title
  - Hmm... maybe rather separate classifier which detects these sentences

> Ausbildung zum Fleischer bzw. Fleischermeister (m/w/d), Fachverkäufer im Lebensmittelhandwerk (m/w/d) mit Schwerpunkt Fleischerei, vergleichbare Ausbildung oder Erfahrung in diesem Bereich
outcome:
- Ausbildung IN Fleischer
- Ausbildung IN Fleischermeister
- Ausbildung IN Fachverkäufer im Lebensmittelhandwerk
- (second Ausbildung as well?!)
- Erfahrung IN "diesem Bereich" -> replace with name of job title

In general: Each line seems to consist only of ODER


Which labels?
- education type (Ausbildung, Studium, ...)
- education topic (Wirtschaftsinformatik, ...)
- education:in

> Wirt­schafts­wissen­schaften/-infor­matik/-mathe­matik
> (Wirtschafts-)Informatik 
- should ideally be Wirtschaftsinformatik, Informatik
- not sure how to deal with this. Maybe need hyphen replacer after all? shouldn't be too difficult using BERT or distillBERT.
- Ah, or simply regex expression! Need to collect a lot more samples though.
- Yep, perform regex first

**Existing de-hyphenizations**
from https://github.com/adbar/German-NLP
- https://github.com/pd3f/dehyphen = removes hyphens on line breaks
- https://github.com/spencermountain/compromise - seems interesting, but not for German
- https://github.com/msiemens/HypheNN-de - creates hyphens


## TODO later: Update docker

- Problem:
  - python libraries are installed using root
  - for vscode, HOME needs to be set differently (independent of user, though!)
  - files created as root cannot be easily accessed otherwise :/
- Solution  
  - add custom home directory
  - but still use root user
- TODO: Update utils/project-template
  - update Dockerfile - no USER
  - update .devcontainer.json - remove remoteUser
  - update launch.json - also accept breakpoints in system libraries code