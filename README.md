
![sbb-ner-demo example](.screenshots/sbb_ner_demo.png?raw=true)

***
# Preprocessing of NER ground-truth:


## compile_conll

Read CONLL 2003 ner ground truth files from directory and
write the outcome of the data parsing to some pandas DataFrame that is
stored as pickle.

### Usage

```
compile_conll --help
```

## compile_germ_eval

Read germ eval .tsv files from directory and write the
outcome of the data parsing to some pandas DataFrame that is stored as
pickle.

### Usage

```
compile_germ_eval --help
```

## compile_europeana_historic

Read europeana historic ner ground truth .bio files from directory 
and write the outcome of the data parsing to some pandas
DataFrame that is stored as pickle.

### Usage

```
compile_europeana_historic --help
```


## compile_wikiner

Read wikiner files from directory and write the outcome
of the data parsing to some pandas DataFrame that is stored as pickle.

### Usage

```
compile_wikiner --help
```

***
# Train BERT - NER model:

## bert-ner

Perform BERT for NER supervised training and test/cross-validation.

### Usage

```
bert-ner --help
```
