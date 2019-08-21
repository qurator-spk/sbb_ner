REPO_PATH ?=$(shell pwd)

BERT_BASE_PATH ?=$(REPO_PATH)/data/BERT

BERT_URL ?=https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
REL_BERT_PATH ?=multi_cased_L-12_H-768_A-12

#BERT_URL ?=https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
#REL_BERT_PATH ?=wwm_uncased_L-24_H-1024_A-16

BERT_MODEL_PATH ?=$(BERT_BASE_PATH)/$(REL_BERT_PATH)

DIGISAM_PATH ?=$(REPO_PATH)/data/digisam

REL_FINETUNED_PATH ?=data/digisam/BERT_de_finetuned
#REL_FINETUNED_PATH ?=data/digisam/BERT-large_de_finetuned
BERT_FINETUNED_PATH ?=$(REPO_PATH)/$(REL_FINETUNED_PATH)

NER_DATA_PATH ?=$(REPO_PATH)/data/NER

REL_BUILD_PATH ?=data/build
BUILD_PATH ?=$(REPO_PATH)/$(REL_BUILD_PATH)

EPOCHS ?=1
EPOCH_FILE ?=pytorch_model_ep$(EPOCHS).bin
MODEL_FILE ?=pytorch_model.bin
CROSS_VAL_FILE ?=cross_validation_results.pkl

WEIGHT_DECAY ?=0.03
WARMUP_PROPORTION ?=0.4

BATCH_SIZE ?=32
GRAD_ACC_STEPS ?=2

# BATCH_SIZE ?=128 # <===== unsupervised
# GRAD_ACC_STEPS ?=4 # <===== unsupervised

MAX_SEQ_LEN ?=128

# EXTRA_OPTIONS="--dry_run --no_cuda" <- Test if everything works.
EXTRA_OPTIONS ?=

DO_LOWER_CASE ?=

BERT_NER_OPTIONS ?=--task_name=ner --max_seq_length=$(MAX_SEQ_LEN) --num_train_epochs=$(EPOCHS) --warmup_proportion=$(WARMUP_PROPORTION) --gradient_accumulation_steps=$(GRAD_ACC_STEPS) --train_batch_size=$(BATCH_SIZE) --gt_file=$(BUILD_PATH)/gt.pkl --weight_decay=$(WEIGHT_DECAY) $(DO_LOWER_CASE) $(EXTRA_OPTIONS) 

BERT_NER_EVAL_OPTIONS ?=--eval_batch_size=8 --task_name=ner --gt_file=$(BUILD_PATH)/gt.pkl $(DO_LOWER_CASE) $(EXTRA_OPTIONS)

###############################################################################
# directories
#

$(BUILD_PATH):
	mkdir -p $(BUILD_PATH)

$(BERT_FINETUNED_PATH):
	mkdir -p $(BERT_FINETUNED_PATH)
	cp -L $(BERT_MODEL_PATH)/pytorch_model.bin $(BERT_FINETUNED_PATH)/pytorch_model.bin
	chmod u+rw $(BERT_FINETUNED_PATH)/pytorch_model.bin
	ln -sfn $(BERT_MODEL_PATH)/bert_config.json $(BERT_FINETUNED_PATH)/bert_config.json
	ln -sfn $(BERT_MODEL_PATH)/vocab.txt $(BERT_FINETUNED_PATH)/vocab.txt

dirs: $(BUILD_PATH) $(BERT_FINETUNED_PATH)

###############################################################################
# BERT unsupervised on "Digitale Sammlungen":
#

TEMP_PREFIX ?=/tmp/

$(BERT_MODEL_PATH)/bert_model.ckpt.index:
	wget -nc --directory-prefix=$(BERT_BASE_PATH) $(BERT_URL)
	unzip -d $(BERT_BASE_PATH) $(BERT_MODEL_PATH).zip

$(BERT_MODEL_PATH)/pytorch_model.bin:	$(BERT_MODEL_PATH)/bert_model.ckpt.index
	pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch $(BERT_MODEL_PATH)/bert_model.ckpt $(BERT_MODEL_PATH)/bert_config.json $(BERT_MODEL_PATH)/pytorch_model.bin

$(DIGISAM_PATH)/de_corpus.txt:
	altocsv2corpus $(DIGISAM_PATH)/xml2csv_alto.csv $(DIGISAM_PATH)/selection_de.pkl $(DIGISAM_PATH)/de_corpus.txt --chunksize=10000

$(BERT_MODEL_PATH)/epoch_0.json: $(DIGISAM_PATH)/de_corpus.txt $(BERT_MODEL_PATH)/pytorch_model.bin
	bert-pregenerate-trainingdata --train_corpus $(DIGISAM_PATH)/de_corpus.txt --output_dir $(BERT_MODEL_PATH) --bert_model $(BERT_MODEL_PATH) --reduce_memory --epochs $(EPOCHS)

bert-digisam-unsupervised: $(BERT_MODEL_PATH)/epoch_0.json
	bert-finetune --pregenerated_data $(BERT_MODEL_PATH) --output_dir $(BERT_FINETUNED_PATH) --bert_model $(BERT_MODEL_PATH) --reduce_memory --fp16 --gradient_accumulation_steps 4 --train_batch_size 32 --epochs $(EPOCHS) --temp_prefix $(TEMP_PREFIX)

bert-digisam-unsupervised-continued: $(BERT_FINETUNED_PATH) $(BERT_MODEL_PATH)/epoch_0.json
	bert-finetune --pregenerated_data $(BERT_MODEL_PATH) --output_dir $(BERT_FINETUNED_PATH) --bert_model $(BERT_FINETUNED_PATH) --reduce_memory --fp16 --gradient_accumulation_steps=$(GRAD_ACC_STEPS) --train_batch_size=$(BATCH_SIZE) --epochs $(EPOCHS) --temp_prefix $(TEMP_PREFIX)

get-bert: $(BERT_MODEL_PATH)/bert_model.ckpt.index

convert-bert: $(BERT_MODEL_PATH)/pytorch_model.bin

###############################################################################
#NER ground truth:

$(NER_DATA_PATH)/ner-corpora:
	git clone https://github.com/EuropeanaNewspapers/ner-corpora $(NER_DATA_PATH)/ner-corpora

$(BUILD_PATH)/europeana_historic.pkl: $(NER_DATA_PATH)/ner-corpora
	compile_europeana_historic $(NER_DATA_PATH)/ner-corpora $(BUILD_PATH)/europeana_historic.pkl

$(BUILD_PATH)/germ_eval.pkl: $(NER_DATA_PATH)/GermEval
	compile_germ_eval $(NER_DATA_PATH)/GermEval $(BUILD_PATH)/germ_eval.pkl

$(BUILD_PATH)/conll2003.pkl: $(NER_DATA_PATH)/conll2003
	compile_conll $(NER_DATA_PATH)/conll2003 $(BUILD_PATH)/conll2003.pkl

$(BUILD_PATH)/wikiner.pkl: $(NER_DATA_PATH)/wikiner
	compile_wikiner $(NER_DATA_PATH)/wikiner $(BUILD_PATH)/wikiner.pkl

$(BUILD_PATH)/gt.pkl:
	python qurator/ner/join_gt.py $(BUILD_PATH)/germ_eval.pkl $(BUILD_PATH)/europeana_historic.pkl $(BUILD_PATH)/conll2003.pkl $(BUILD_PATH)/wikiner.pkl $(BUILD_PATH)/gt.pkl

ner-ground-truth: dirs $(BUILD_PATH)/europeana_historic.pkl $(BUILD_PATH)/germ_eval.pkl $(BUILD_PATH)/conll2003.pkl $(BUILD_PATH)/wikiner.pkl $(BUILD_PATH)/gt.pkl

###############################################################################
#BERT NER training:

.PRECIOUS: %/vocab.txt %/bert_config.json %/$(MODEL_FILE)

%/vocab.txt:
	ln -sfnr $(BERT_FINETUNED_PATH)/vocab.txt $(@D)/vocab.txt

%/bert_config.json:
	ln -sfnr $(BERT_FINETUNED_PATH)/bert_config.json $(@D)/bert_config.json

%/$(MODEL_FILE):
	ln -sfnr $(@D)/$(EPOCH_FILE) $(@D)/$(MODEL_FILE)

########################################
# baseline

$(BUILD_PATH)/bert-conll2003-en-baseline/$(EPOCH_FILE):
	bert-ner --train_sets='EN-CONLL-TRAIN' --dev_sets='EN-CONLL-TESTA' --bert_model=bert-base-cased --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-conll2003-de-baseline/$(EPOCH_FILE):
	bert-ner --train_sets='DE-CONLL-TRAIN' --dev_sets='DE-CONLL-TESTA' --bert_model=bert-base-multilingual-cased --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-germ-eval-baseline/$(EPOCH_FILE):
	bert-ner --train_sets='GERM-EVAL-TRAIN' --dev_sets='GERM-EVAL-DEV' --bert_model=bert-base-multilingual-cased --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1 


$(BUILD_PATH)/bert-all-german-baseline/$(EPOCH_FILE):
	bert-ner --train_sets='GERM-EVAL-TRAIN|DE-CONLL-TRAIN' --dev_sets='GERM-EVAL-DEV|DE-CONLL-TESTA' --bert_model=bert-base-multilingual-cased --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1


$(BUILD_PATH)/bert-wikiner-baseline/$(EPOCH_FILE):
	bert-ner --train_sets='WIKINER-WP3' --dev_sets='GERM-EVAL-DEV|DE-CONLL-TESTA' --bert_model=bert-base-multilingual-cased --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1


$(BUILD_PATH)/bert-lft-baseline/$(EPOCH_FILE):
	bert-ner --train_sets='LFT' --bert_model=bert-base-multilingual-cased --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-onb-baseline/$(EPOCH_FILE):
	bert-ner --train_sets='ONB' --bert_model=bert-base-multilingual-cased --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-sbb-baseline/$(EPOCH_FILE):
	bert-ner --train_sets='SBB' --bert_model=bert-base-multilingual-cased --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-lft-sbb-baseline/$(EPOCH_FILE):
	bert-ner --train_sets='LFT|SBB' --bert_model=bert-base-multilingual-cased --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-onb-sbb-baseline/$(EPOCH_FILE):
	bert-ner --train_sets='ONB|SBB' --bert_model=bert-base-multilingual-cased --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-onb-lft-baseline/$(EPOCH_FILE):
	bert-ner --train_sets='ONB|LFT' --bert_model=bert-base-multilingual-cased --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1


bert-%-baseline: $(BUILD_PATH)/bert-%-baseline/$(EPOCH_FILE) $(BUILD_PATH)/bert-%-baseline/vocab.txt $(BUILD_PATH)/bert-%-baseline/bert_config.json $(BUILD_PATH)/bert-%-baseline/$(MODEL_FILE) ;

bert-baseline: dirs ner-ground-truth bert-conll2003-en-baseline bert-conll2003-de-baseline bert-germ-eval-baseline bert-all-german-baseline bert-wikiner-baseline bert-lft-baseline bert-onb-baseline bert-sbb-baseline bert-lft-sbb-baseline bert-onb-sbb-baseline bert-onb-lft-baseline


$(BUILD_PATH)/bert-lft-baseline/$(CROSS_VAL_FILE):
	bert-ner --do_cross_validation --train_sets='LFT' --bert_model=bert-base-multilingual-cased --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-onb-baseline/$(CROSS_VAL_FILE):
	bert-ner --do_cross_validation --train_sets='ONB' --bert_model=bert-base-multilingual-cased --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-sbb-baseline/$(CROSS_VAL_FILE):
	bert-ner --do_cross_validation --train_sets='SBB' --bert_model=bert-base-multilingual-cased --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

bert-cv-%-baseline: $(BUILD_PATH)/bert-%-baseline/$(CROSS_VAL_FILE) $(BUILD_PATH)/bert-%-baseline/vocab.txt $(BUILD_PATH)/bert-%-baseline/bert_config.json $(BUILD_PATH)/bert-%-baseline/$(MODEL_FILE) ;

bert-cv-baseline: bert-cv-lft-baseline bert-cv-onb-baseline bert-cv-sbb-baseline
	
########################################
#de-finetuned

$(BUILD_PATH)/bert-conll2003-de-finetuned/$(EPOCH_FILE): 
	bert-ner --train_sets='DE-CONLL-TRAIN' --dev_sets='DE-CONLL-TESTA' --bert_model=$(BERT_FINETUNED_PATH) --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-germ-eval-de-finetuned/$(EPOCH_FILE):
	bert-ner --train_sets='GERM-EVAL-TRAIN' --dev_sets='GERM-EVAL-DEV' --bert_model=$(BERT_FINETUNED_PATH) --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-all-german-de-finetuned/$(EPOCH_FILE):
	bert-ner --train_sets='GERM-EVAL-TRAIN|DE-CONLL-TRAIN' --dev_sets='GERM-EVAL-DEV|DE-CONLL-TESTA' --bert_model=$(BERT_FINETUNED_PATH) --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1


$(BUILD_PATH)/bert-wikiner-de-finetuned/$(EPOCH_FILE):
	bert-ner --train_sets='WIKINER-WP3' --dev_sets='GERM-EVAL-DEV|DE-CONLL-TESTA' --bert_model=$(BERT_FINETUNED_PATH) --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1


$(BUILD_PATH)/bert-lft-de-finetuned/$(EPOCH_FILE):
	bert-ner --train_sets='LFT' --bert_model=$(BERT_FINETUNED_PATH) --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-onb-de-finetuned/$(EPOCH_FILE):
	bert-ner --train_sets='ONB' --bert_model=$(BERT_FINETUNED_PATH) --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-sbb-de-finetuned/$(EPOCH_FILE):
	bert-ner --train_sets='SBB' --bert_model=$(BERT_FINETUNED_PATH) --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-lft-sbb-de-finetuned/$(EPOCH_FILE):
	bert-ner --train_sets='LFT|SBB' --bert_model=$(BERT_FINETUNED_PATH) --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-onb-sbb-de-finetuned/$(EPOCH_FILE):
	bert-ner --train_sets='ONB|SBB' --bert_model=$(BERT_FINETUNED_PATH) --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-onb-lft-de-finetuned/$(EPOCH_FILE):
	bert-ner --train_sets='ONB|LFT' --bert_model=$(BERT_FINETUNED_PATH) --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1


bert-%-de-finetuned: $(BUILD_PATH)/bert-%-de-finetuned/$(EPOCH_FILE) $(BUILD_PATH)/bert-%-de-finetuned/vocab.txt $(BUILD_PATH)/bert-%-de-finetuned/bert_config.json $(BUILD_PATH)/bert-%-de-finetuned/$(MODEL_FILE) ;

bert-finetuned: dirs ner-ground-truth bert-conll2003-de-finetuned bert-germ-eval-de-finetuned bert-all-german-de-finetuned bert-wikiner-de-finetuned bert-lft-de-finetuned bert-onb-de-finetuned bert-sbb-de-finetuned bert-lft-sbb-de-finetuned bert-onb-sbb-de-finetuned bert-onb-lft-de-finetuned


$(BUILD_PATH)/bert-lft-de-finetuned/$(CROSS_VAL_FILE):
	bert-ner --do_cross_validation --train_sets='LFT' --bert_model=$(BERT_FINETUNED_PATH) --output_dir=$(@D) $(BERT_NER_OPTIONS) # >> $(@D).log 2<&1

$(BUILD_PATH)/bert-onb-de-finetuned/$(CROSS_VAL_FILE):
	bert-ner --do_cross_validation --train_sets='ONB' --bert_model=$(BERT_FINETUNED_PATH) --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-sbb-de-finetuned/$(CROSS_VAL_FILE): 
	bert-ner --do_cross_validation --train_sets='SBB' --bert_model=$(BERT_FINETUNED_PATH) --output_dir=$(@D) $(BERT_NER_OPTIONS) >> $(@D).log 2<&1

bert-cv-%-de-finetuned: $(BUILD_PATH)/bert-%-de-finetuned/$(CROSS_VAL_FILE) $(BUILD_PATH)/bert-%-de-finetuned/vocab.txt $(BUILD_PATH)/bert-%-de-finetuned/bert_config.json $(BUILD_PATH)/bert-%-de-finetuned/$(MODEL_FILE) ;

bert-cv-de-finetuned: bert-cv-lft-de-finetuned bert-cv-onb-de-finetuned bert-cv-sbb-de-finetuned

bert-cv: bert-cv-de-finetuned bert-cv-baseline

bert-train: bert-finetuned bert-baseline

###############################################################################
# Evaluation
#

$(BUILD_PATH)/bert-conll2003-de-baseline/eval_results-DE-CONLL-TESTA.pkl:
	bert-ner --dev_sets='DE-CONLL-TESTA' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-conll2003-de-baseline/eval_results-DE-CONLL-TESTB.pkl:
	bert-ner --dev_sets='DE-CONLL-TESTB' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-conll2003-de-baseline/eval_results-LFT.pkl:
	bert-ner --dev_sets='LFT' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-conll2003-de-baseline/eval_results-SBB.pkl:
	bert-ner --dev_sets='SBB' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-conll2003-de-baseline/eval_results-ONB.pkl:
	bert-ner --dev_sets='ONB' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

#


$(BUILD_PATH)/bert-germ-eval-baseline/eval_results-GERM-EVAL-TEST.pkl:
	bert-ner --dev_sets='GERM-EVAL-TEST' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-all-german-baseline/eval_results-GERM-EVAL-TEST.pkl:
	bert-ner --dev_sets='GERM-EVAL-TEST' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-wikiner-baseline/eval_results-GERM-EVAL-TEST.pkl: 
	bert-ner --dev_sets='GERM-EVAL-TEST' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-germ-eval-baseline/eval_results-LFT.pkl:
	bert-ner --dev_sets='LFT' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-germ-eval-baseline/eval_results-SBB.pkl:
	bert-ner --dev_sets='SBB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-germ-eval-baseline/eval_results-ONB.pkl:
	bert-ner --dev_sets='ONB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

#

$(BUILD_PATH)/bert-all-german-baseline/eval_results-DE-CONLL-TESTA.pkl:
	bert-ner --dev_sets='DE-CONLL-TESTA' --output_dir=$(@D) --num_train_epochs $(EPOCHS)   $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-wikiner-baseline/eval_results-DE-CONLL-TESTA.pkl: 
	bert-ner --dev_sets='DE-CONLL-TESTA' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

#

$(BUILD_PATH)/bert-all-german-baseline/eval_results-DE-CONLL-TESTB.pkl:
	bert-ner --dev_sets='DE-CONLL-TESTB' --output_dir=$(@D) --num_train_epochs $(EPOCHS)   $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-wikiner-baseline/eval_results-DE-CONLL-TESTB.pkl: 
	bert-ner --dev_sets='DE-CONLL-TESTB' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

#

$(BUILD_PATH)/bert-all-german-baseline/eval_results-LFT.pkl:
	bert-ner --dev_sets='LFT' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-all-german-baseline/eval_results-SBB.pkl:
	bert-ner --dev_sets='SBB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-all-german-baseline/eval_results-ONB.pkl:
	bert-ner --dev_sets='ONB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

#

$(BUILD_PATH)/bert-wikiner-baseline/eval_results-LFT.pkl:
	bert-ner --dev_sets='LFT' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-wikiner-baseline/eval_results-SBB.pkl:
	bert-ner --dev_sets='SBB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-wikiner-baseline/eval_results-ONB.pkl:
	bert-ner --dev_sets='ONB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1


$(BUILD_PATH)/bert-lft-baseline/eval_results-ONB.pkl:
	bert-ner --dev_sets='ONB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-lft-baseline/eval_results-SBB.pkl:
	bert-ner --dev_sets='SBB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1


$(BUILD_PATH)/bert-onb-baseline/eval_results-LFT.pkl:
	bert-ner --dev_sets='LFT' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-onb-baseline/eval_results-SBB.pkl:
	bert-ner --dev_sets='SBB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1


$(BUILD_PATH)/bert-sbb-baseline/eval_results-LFT.pkl:
	bert-ner --dev_sets='LFT' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-sbb-baseline/eval_results-ONB.pkl:
	bert-ner --dev_sets='ONB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1


$(BUILD_PATH)/bert-lft-sbb-baseline/eval_results-ONB.pkl:
	bert-ner --dev_sets='ONB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-onb-sbb-baseline/eval_results-LFT.pkl:
	bert-ner --dev_sets='LFT' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-onb-lft-baseline/eval_results-SBB.pkl:
	bert-ner --dev_sets='SBB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1


bert-ner-evaluation-baseline: dirs $(BUILD_PATH)/bert-all-german-baseline/eval_results-LFT.pkl $(BUILD_PATH)/bert-all-german-baseline/eval_results-SBB.pkl $(BUILD_PATH)/bert-all-german-baseline/eval_results-ONB.pkl $(BUILD_PATH)/bert-wikiner-baseline/eval_results-LFT.pkl $(BUILD_PATH)/bert-wikiner-baseline/eval_results-SBB.pkl $(BUILD_PATH)/bert-wikiner-baseline/eval_results-ONB.pkl $(BUILD_PATH)/bert-all-german-baseline/eval_results-GERM-EVAL-TEST.pkl $(BUILD_PATH)/bert-wikiner-baseline/eval_results-GERM-EVAL-TEST.pkl $(BUILD_PATH)/bert-all-german-baseline/eval_results-DE-CONLL-TESTA.pkl $(BUILD_PATH)/bert-wikiner-baseline/eval_results-DE-CONLL-TESTA.pkl $(BUILD_PATH)/bert-all-german-baseline/eval_results-DE-CONLL-TESTB.pkl $(BUILD_PATH)/bert-wikiner-baseline/eval_results-DE-CONLL-TESTB.pkl $(BUILD_PATH)/bert-germ-eval-baseline/eval_results-GERM-EVAL-TEST.pkl $(BUILD_PATH)/bert-conll2003-de-baseline/eval_results-DE-CONLL-TESTA.pkl $(BUILD_PATH)/bert-conll2003-de-baseline/eval_results-DE-CONLL-TESTB.pkl $(BUILD_PATH)/bert-lft-baseline/eval_results-ONB.pkl $(BUILD_PATH)/bert-lft-baseline/eval_results-SBB.pkl $(BUILD_PATH)/bert-onb-baseline/eval_results-LFT.pkl $(BUILD_PATH)/bert-onb-baseline/eval_results-SBB.pkl $(BUILD_PATH)/bert-sbb-baseline/eval_results-LFT.pkl $(BUILD_PATH)/bert-sbb-baseline/eval_results-ONB.pkl $(BUILD_PATH)/bert-lft-sbb-baseline/eval_results-ONB.pkl $(BUILD_PATH)/bert-germ-eval-baseline/eval_results-LFT.pkl $(BUILD_PATH)/bert-germ-eval-baseline/eval_results-SBB.pkl $(BUILD_PATH)/bert-germ-eval-baseline/eval_results-ONB.pkl $(BUILD_PATH)/bert-onb-sbb-baseline/eval_results-LFT.pkl $(BUILD_PATH)/bert-onb-lft-baseline/eval_results-SBB.pkl $(BUILD_PATH)/bert-conll2003-de-baseline/eval_results-LFT.pkl $(BUILD_PATH)/bert-conll2003-de-baseline/eval_results-SBB.pkl $(BUILD_PATH)/bert-conll2003-de-baseline/eval_results-ONB.pkl

#######################################

$(BUILD_PATH)/bert-conll2003-de-finetuned/eval_results-DE-CONLL-TESTA.pkl:
	bert-ner --dev_sets='DE-CONLL-TESTA' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-conll2003-de-finetuned/eval_results-DE-CONLL-TESTB.pkl:
	bert-ner --dev_sets='DE-CONLL-TESTB' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-conll2003-de-finetuned/eval_results-LFT.pkl:
	bert-ner --dev_sets='LFT' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-conll2003-de-finetuned/eval_results-SBB.pkl:
	bert-ner --dev_sets='SBB' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-conll2003-de-finetuned/eval_results-ONB.pkl:
	bert-ner --dev_sets='ONB' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1


$(BUILD_PATH)/bert-germ-eval-de-finetuned/eval_results-GERM-EVAL-TEST.pkl:
	bert-ner --dev_sets='GERM-EVAL-TEST' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-all-german-de-finetuned/eval_results-GERM-EVAL-TEST.pkl:
	bert-ner --dev_sets='GERM-EVAL-TEST' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-wikiner-de-finetuned/eval_results-GERM-EVAL-TEST.pkl: 
	bert-ner --dev_sets='GERM-EVAL-TEST' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1
#


$(BUILD_PATH)/bert-germ-eval-de-finetuned/eval_results-LFT.pkl:
	bert-ner --dev_sets='LFT' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-germ-eval-de-finetuned/eval_results-SBB.pkl:
	bert-ner --dev_sets='SBB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-germ-eval-de-finetuned/eval_results-ONB.pkl:
	bert-ner --dev_sets='ONB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

#

$(BUILD_PATH)/bert-all-german-de-finetuned/eval_results-DE-CONLL-TESTA.pkl:
	bert-ner --dev_sets='DE-CONLL-TESTA' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-wikiner-de-finetuned/eval_results-DE-CONLL-TESTA.pkl: 
	bert-ner --dev_sets='DE-CONLL-TESTA' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

#

$(BUILD_PATH)/bert-all-german-de-finetuned/eval_results-DE-CONLL-TESTB.pkl:
	bert-ner --dev_sets='DE-CONLL-TESTB' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-wikiner-de-finetuned/eval_results-DE-CONLL-TESTB.pkl: 
	bert-ner --dev_sets='DE-CONLL-TESTB' --output_dir=$(@D) --num_train_epochs $(EPOCHS)  $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

#

$(BUILD_PATH)/bert-all-german-de-finetuned/eval_results-LFT.pkl:
	bert-ner --dev_sets='LFT' --output_dir=$(BUILD_PATH)/bert-all-german-de-finetuned --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-all-german-de-finetuned/eval_results-SBB.pkl:
	bert-ner --dev_sets='SBB' --output_dir=$(BUILD_PATH)/bert-all-german-de-finetuned --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-all-german-de-finetuned/eval_results-ONB.pkl:
	bert-ner --dev_sets='ONB' --output_dir=$(BUILD_PATH)/bert-all-german-de-finetuned --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

#

$(BUILD_PATH)/bert-wikiner-de-finetuned/eval_results-LFT.pkl:
	bert-ner --dev_sets='LFT' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-wikiner-de-finetuned/eval_results-SBB.pkl:
	bert-ner --dev_sets='SBB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-wikiner-de-finetuned/eval_results-ONB.pkl:
	bert-ner --dev_sets='ONB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1


$(BUILD_PATH)/bert-lft-de-finetuned/eval_results-ONB.pkl:
	bert-ner --dev_sets='ONB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-lft-de-finetuned/eval_results-SBB.pkl:
	bert-ner --dev_sets='SBB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1


$(BUILD_PATH)/bert-onb-de-finetuned/eval_results-LFT.pkl:
	bert-ner --dev_sets='LFT' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-onb-de-finetuned/eval_results-SBB.pkl:
	bert-ner --dev_sets='SBB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1


$(BUILD_PATH)/bert-sbb-de-finetuned/eval_results-LFT.pkl:
	bert-ner --dev_sets='LFT' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-sbb-de-finetuned/eval_results-ONB.pkl:
	bert-ner --dev_sets='ONB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1


$(BUILD_PATH)/bert-lft-sbb-de-finetuned/eval_results-ONB.pkl:
	bert-ner --dev_sets='ONB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-onb-sbb-de-finetuned/eval_results-LFT.pkl:
	bert-ner --dev_sets='LFT' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1

$(BUILD_PATH)/bert-onb-lft-de-finetuned/eval_results-SBB.pkl:
	bert-ner --dev_sets='SBB' --output_dir=$(@D) --num_train_epochs $(EPOCHS) $(BERT_NER_EVAL_OPTIONS) >> $(@D).log 2<&1


bert-ner-evaluation-de-finetuned: dirs $(BUILD_PATH)/bert-all-german-de-finetuned/eval_results-LFT.pkl $(BUILD_PATH)/bert-all-german-de-finetuned/eval_results-SBB.pkl $(BUILD_PATH)/bert-all-german-de-finetuned/eval_results-ONB.pkl $(BUILD_PATH)/bert-wikiner-de-finetuned/eval_results-LFT.pkl $(BUILD_PATH)/bert-wikiner-de-finetuned/eval_results-SBB.pkl $(BUILD_PATH)/bert-wikiner-de-finetuned/eval_results-ONB.pkl $(BUILD_PATH)/bert-all-german-de-finetuned/eval_results-GERM-EVAL-TEST.pkl $(BUILD_PATH)/bert-wikiner-de-finetuned/eval_results-GERM-EVAL-TEST.pkl $(BUILD_PATH)/bert-all-german-de-finetuned/eval_results-DE-CONLL-TESTA.pkl $(BUILD_PATH)/bert-wikiner-de-finetuned/eval_results-DE-CONLL-TESTA.pkl $(BUILD_PATH)/bert-all-german-de-finetuned/eval_results-DE-CONLL-TESTB.pkl $(BUILD_PATH)/bert-wikiner-de-finetuned/eval_results-DE-CONLL-TESTB.pkl $(BUILD_PATH)/bert-germ-eval-de-finetuned/eval_results-GERM-EVAL-TEST.pkl $(BUILD_PATH)/bert-conll2003-de-finetuned/eval_results-DE-CONLL-TESTA.pkl $(BUILD_PATH)/bert-conll2003-de-finetuned/eval_results-DE-CONLL-TESTB.pkl $(BUILD_PATH)/bert-lft-de-finetuned/eval_results-ONB.pkl $(BUILD_PATH)/bert-lft-de-finetuned/eval_results-SBB.pkl $(BUILD_PATH)/bert-onb-de-finetuned/eval_results-LFT.pkl $(BUILD_PATH)/bert-onb-de-finetuned/eval_results-SBB.pkl $(BUILD_PATH)/bert-sbb-de-finetuned/eval_results-LFT.pkl $(BUILD_PATH)/bert-sbb-de-finetuned/eval_results-ONB.pkl $(BUILD_PATH)/bert-lft-sbb-de-finetuned/eval_results-ONB.pkl $(BUILD_PATH)/bert-germ-eval-de-finetuned/eval_results-LFT.pkl $(BUILD_PATH)/bert-germ-eval-de-finetuned/eval_results-SBB.pkl $(BUILD_PATH)/bert-germ-eval-de-finetuned/eval_results-ONB.pkl $(BUILD_PATH)/bert-onb-sbb-de-finetuned/eval_results-LFT.pkl $(BUILD_PATH)/bert-onb-lft-de-finetuned/eval_results-SBB.pkl $(BUILD_PATH)/bert-conll2003-de-finetuned/eval_results-LFT.pkl $(BUILD_PATH)/bert-conll2003-de-finetuned/eval_results-SBB.pkl $(BUILD_PATH)/bert-conll2003-de-finetuned/eval_results-ONB.pkl

bert-evaluation: bert-ner-evaluation-baseline bert-ner-evaluation-de-finetuned

###############################################################################
#wikipedia

WIKI_DATA_DIR=data/wikipedia
WP_EPOCH_SIZE=100000

wikipedia-ner-baseline-train: $(WIKI_DATA_DIR)/wikipedia-tagged.csv
	bert-ner --gt_file=$(WIKI_DATA_DIR)/wikipedia-tagged.csv --train_sets=$(WIKI_DATA_DIR)/ner-train-index.pkl --dev_sets=$(WIKI_DATA_DIR)/ner-dev-index.pkl --bert_model=bert-base-multilingual-cased --task_name=wikipedia-ner --output_dir=$(BUILD_PATH)/wikipedia-baseline --num_train_epochs $(EPOCHS) --num_data_epochs=$(EPOCHS) --epoch_size=$(WP_EPOCH_SIZE) $(BERT_NER_OPTIONS)

wikipedia-ner-de-finetuned-train: $(WIKI_DATA_DIR)/wikipedia-tagged.csv
	bert-ner --gt_file=$(WIKI_DATA_DIR)/wikipedia-tagged.csv --train_sets=$(WIKI_DATA_DIR)/ner-train-index.pkl --dev_sets=$(WIKI_DATA_DIR)/ner-dev-index.pkl --bert_model=$(DIGISAM_PATH)/BERT_de_finetuned --task_name=wikipedia-ner --output_dir=$(BUILD_PATH)/wikipedia-de-finetuned --num_train_epochs $(EPOCHS) --num_data_epochs=$(EPOCHS) --epoch_size=$(WP_EPOCH_SIZE) $(BERT_NER_OPTIONS)

########################

$(BUILD_PATH)/wikipedia-baseline/eval_results-LFT.pkl:
	bert-ner --dev_sets='LFT' --task_name=ner --output_dir=$(@D) $(BERT_NER_EVAL_OPTIONS) 

$(BUILD_PATH)/wikipedia-baseline/eval_results-SBB.pkl:
	bert-ner --dev_sets='SBB' --task_name=ner --output_dir=$(@D) $(BERT_NER_EVAL_OPTIONS)  

$(BUILD_PATH)/wikipedia-baseline/eval_results-ONB.pkl:
	bert-ner --dev_sets='ONB' --task_name=ner --output_dir=$(@D) $(BERT_NER_EVAL_OPTIONS) 

$(BUILD_PATH)/wikipedia-baseline/eval_results-DE-CONLL-TESTA.pkl:
	bert-ner --dev_sets='DE-CONLL-TESTA' --task_name=ner --output_dir=$(@D) $(BERT_NER_EVAL_OPTIONS) 


$(BUILD_PATH)/wikipedia-de-finetuned/eval_results-LFT.pkl:
	bert-ner --dev_sets='LFT' --task_name=ner --output_dir=$(@D) $(BERT_NER_EVAL_OPTIONS)  

$(BUILD_PATH)/wikipedia-de-finetuned/eval_results-SBB.pkl:
	bert-ner --dev_sets='SBB' --task_name=ner --output_dir=$(@D) $(BERT_NER_EVAL_OPTIONS) 

$(BUILD_PATH)/wikipedia-de-finetuned/eval_results-DE-CONLL-TESTA.pkl:
	bert-ner --dev_sets='DE-CONLL-TESTA' --task_name=ner --output_dir=$(@D) $(BERT_NER_EVAL_OPTIONS)  

wikipedia-baseline-evaluation: $(BUILD_PATH)/wikipedia-baseline/eval_results-SBB.pkl $(BUILD_PATH)/wikipedia-baseline/eval_results-LFT.pkl $(BUILD_PATH)/wikipedia-baseline/eval_results-ONB.pkl $(BUILD_PATH)/wikipedia-baseline/eval_results-DE-CONLL-TESTA.pkl

wikipedia-evaluation: $(BUILD_PATH)/wikipedia-de-finetuned/eval_results-LFT.pkl
wikipedia-evaluation2: $(BUILD_PATH)/wikipedia-de-finetuned/eval_results-SBB.pkl
wikipedia-evaluation3: $(BUILD_PATH)/wikipedia-de-finetuned/eval_results-DE-CONLL-TESTA.pkl

###############################

model_archive:
	tar --exclude='*ep[1-6]*' --exclude='*eval*' --exclude='pytorch_model.bin' --exclude='*.pkl' -chzf models.tar.gz data/konvens2019/build-wd_0.03/bert-all-german-de-finetuned data/konvens2019/build-on-all-german-de-finetuned/bert-sbb-de-finetuned data/konvens2019/build-wd_0.03/bert-sbb-de-finetuned data/konvens2019/build-wd_0.03/bert-all-german-baseline
