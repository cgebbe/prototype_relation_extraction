# TODO

- [ ] start a new template with the huggingface docker container
- [ ] work in that container
- [ ] load "labeling_word_parts.json"

# Questions

## Can we deal with labeling word **parts**?
> Mathe­matik- oder Physikstudium

- We can label word parts in label studio
- export as ConLL reduces it to word labels, though.
- export as JSON works. (simply marks characters)
- how to load that as JSON labels?



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