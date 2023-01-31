---
tags: 
- pytorch
- token-classification
- sequence-tagger-model
language: de
datasets:
- conll2003
- germeval_14
license: apache-2.0
---






# Model Card for sbb_ner

<!-- Provide a quick summary of what the model is/does. [Optional] -->
A BERT model trained on three German corpora containing contemporary and historical texts for named entity recognition tasks. It predicts the classes PER, LOC and ORG. 
Questions and comments about the model can be directed to Clemens Neudecker at clemens.neudecker@sbb.spk-berlin.de.




#  Table of Contents

- [Model Card for sbb_ner](#model-card-for--model_id-)
- [Table of Contents](#table-of-contents)
- [Table of Contents](#table-of-contents-1)
- [Model Details](#model-details)
  - [Model Description](#model-description)
- [Uses](#uses)
  - [Direct Use](#direct-use)
  - [Downstream Use [Optional]](#downstream-use-optional)
  - [Out-of-Scope Use](#out-of-scope-use)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
  - [Recommendations](#recommendations)
- [Training Details](#training-details)
  - [Training Data](#training-data)
  - [Training Procedure](#training-procedure)
    - [Preprocessing](#preprocessing)
    - [Speeds, Sizes, Times](#speeds-sizes-times)
- [Evaluation](#evaluation)
  - [Testing Data, Factors & Metrics](#testing-data-factors--metrics)
    - [Testing Data](#testing-data)
    - [Factors](#factors)
    - [Metrics](#metrics)
  - [Results](#results)
- [Model Examination](#model-examination)
- [Environmental Impact](#environmental-impact)
- [Technical Specifications [optional]](#technical-specifications-optional)
  - [Model Architecture and Objective](#model-architecture-and-objective)
  - [Compute Infrastructure](#compute-infrastructure)
    - [Hardware](#hardware)
    - [Software](#software)
- [Citation](#citation)
- [Glossary [optional]](#glossary-optional)
- [More Information [optional]](#more-information-optional)
- [Model Card Authors [optional]](#model-card-authors-optional)
- [Model Card Contact](#model-card-contact)
- [How to Get Started with the Model](#how-to-get-started-with-the-model)


# Model Details

## Model Description

<!-- Provide a longer summary of what this model is/does. -->
A BERT model trained on three German corpora containing contemporary and historical texts for named entity recognition tasks. 
It predicts the classes PER, LOC and ORG. 

- **Developed by:** [Kai Labusch](https://huggingface.co/labusch), [Clemens Neudecker](https://huggingface.co/cneud), David Zellhöfer
- **Shared by [Optional]:** [Staatsbibliothek zu Berlin / Berlin State Library] (https://huggingface.co/SBB)
- **Model type:** Language model
- **Language(s) (NLP):** de
- **License:** apache-2.0
- **Parent Model:** The BERT base multilingual cased model as provided by [Google] (https://huggingface.co/bert-base-multilingual-cased)
- **Resources for more information:** More information needed
    - [GitHub Repo](https://github.com/qurator-spk/sbb_ner)
    - [Associated Paper](https://konvens.org/proceedings/2019/papers/KONVENS2019_paper_4.pdf)

# Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

## Direct Use

The model can directly be used to perform NER on historical german texts obtained by OCR from digitized documents.
Supported entity types are PER, LOC and ORG. 

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->
<!-- If the user enters content, print that. If not, but they enter a task in the list, use that. If neither, say "more info needed." -->

## Downstream Use

The model has been pre-trained on 2.300.000 pages of OCR-text of the digitized collections of Berlin State Library.
Therefore it is adapted to OCR-error prone historical german texts and might be used for particular applications that involve such text material.

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->
<!-- If the user enters content, print that. If not, but they enter a task in the list, use that. If neither, say "more info needed." -->
 



## Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->
<!-- If the user enters content, print that. If not, but they enter a task in the list, use that. If neither, say "more info needed." -->




# Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The identification of named entities in historical and contemporary texts is a task contributing to knowledge creation aiming at enhancing scientific research and better discoverability of information in digitized historical texts. The aim of the development of this model was to improve this knowledge creation process, an endeavour that is not for profit. The results of the applied model are freely accessible for the users of the digital collections of the Berlin State Library. Against this backdrop, ethical challenges cannot be identified. As a limitation, it has to be noted that there is a lot of performance to gain for historical text by adding more historical ground-truth data. 


## Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

The general observation that historical texts often remain silent and avoid naming of subjects from the colonies and address them anonymously cannot be remedied by named entity recognition. Disambiguation of named entities proves to be challenging beyond the task of automatically identifying entities. The existence of broad variations in the spelling of person and place names because of non-normalized orthography and linguistic change as well as changes in the naming of places according to the context adds to this challenge. Historical texts, especially newspapers, contain narrative descriptions and visual representations of minorities and disadvantaged groups without naming them; de-anonymizing such persons and groups is a research task in itself, which has only been started to be tackled in the 2020&#39;s. 


# Training Details

## Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

Three datasets were used: 
1) CoNLL 2003 German Named Entity Recognition Ground Truth (Tjong Kim Sang and De Meulder, 2003)
2) GermEval Konvens 2014 Shared Task Data (Benikova et al., 2014)
3) DC-SBB Digital Collections of the Berlin State Library (Labusch and Zellhöfer, 2019)
4) Europeana Newspapers Historic German Datasets (Neudecker, 2016)


## Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

The BERT model is trained directly with respect to the NER by implementation of the same method that has been proposed by the BERT authors (Devlin et al., 2018). We applied unsupervised pre-training on 2,333,647 pages of unlabeled historical German text from the Berlin State Library digital collections, and supervised pre-training on two datasets with contemporary German text, conll2003 and germeval_14. Unsupervised pre-training on the DC-SBB data as well as supervised pre-training on contemporary NER ground truth were applied. Unsupervised and supervised pretraining are combined where unsupervised is done first and supervised second. Performance on different combinations of training and test sets was explored, and a 5-fold cross validation and comparison with state of the art approaches was conducted.

### Preprocessing

The model was pretrained on 2.300.000 pages of german texts from the digitized collections of the Berlin State Library.
The texts have been obtained by OCR from the page scans of the documents.

### Speeds, Sizes, Times

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

Since it is an incarnation of the original BERT-model published by Google, all the speed, size and time considerations of that original model hold.
 
# Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->
The model has been evaluated by 5-fold cross-validation on several german historical OCR ground truth datasets. 
See publication for detail.

## Testing Data, Factors & Metrics

### Testing Data

<!-- This should link to a Data Card if possible. -->

Two different test sets contained in the CoNLL 2003 German Named Entity Recognition Ground Truth, 
i.e. TEST-A and TEST-B, have been used for testing (DE-CoNLL-TEST).
Additionaly historical OCR-based ground truth datasets have been used for testing - see publication for details.


### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

The evaluation focuses on NER in historical germans documents, see publication for details.

### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

Performance metrics used in evaluation is precision, recall and F1-score.
See paper for actual results in terms of these metrics.

## Results 

See publication.

# Model Examination

See publication.

# Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** V100
- **Hours used:** Roughly 1-2 week(s) for pretraining. Roughly 1 hour for final NER-training.
- **Cloud Provider:** No cloud.
- **Compute Region:** Germany.
- **Carbon Emitted:** More information needed

# Technical Specifications [optional]

## Model Architecture and Objective

See original BERT publication.

## Compute Infrastructure

Training and pre-training has been performed on a single V100.

### Hardware

See above.

### Software

See published code on github.

# Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

@article{labusch_bert_2019,
	title = {{BERT} for {Named} {Entity} {Recognition} in {Contemporary} and {Historical} {German}},
	volume = {Conference on Natural Language Processing},
	url = {https://konvens.org/proceedings/2019/papers/KONVENS2019_paper_4.pdf},
	abstract = {We apply a pre-trained transformer based representational language model, i.e. BERT (Devlin et al., 2018), to named entity recognition (NER) in contemporary and historical German text and observe state of the art performance for both text categories. We further improve the recognition performance for historical German by unsupervised pre-training on a large corpus of historical German texts of the Berlin State Library and show that best performance for historical German is obtained by unsupervised pre-training on historical German plus supervised pre-training with contemporary NER ground-truth.},
	language = {en},
	author = {Labusch, Kai and Neudecker, Clemens and Zellhöfer, David},
	year = {2019},
	pages = {9},
}

**APA:**

(Labusch et al., 2019)

# Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

More information needed

# More Information [optional]

More information needed

# Model Card Authors [optional]

<!-- This section provides another layer of transparency and accountability. Whose views is this model card representing? How many voices were included in its construction? Etc. -->

Kai Labusch (kai.labusch@sbb.spk-berlin.de)
[Jörg Lehmann](https://huggingface.co/Jrglmn)

# Model Card Contact

Questions and comments about the model can be directed to Clemens Neudecker at clemens.neudecker@sbb.spk-berlin.de, 
questions and comments about the model card can be directed to Jörg Lehmann at joerg.lehmann@sbb.spk-berlin.de

# How to Get Started with the Model

Use the code below to get started with the model.

<details>
How to get started with this model is explained in the ReadMe file of the GitHub repository [over here] (https://github.com/qurator-spk/sbb_ner).
</details>