#!/bin/bash

#my_reference=../datasets/resources/LARGE_BPE38000_CLEAN150/wmt/tok/test.de
reference=~/.sacrebleu/wmt14/full/en-de.de
MOSES_SCRIPTS=~/utils/mosesdecoder/scripts
MULTI_BLEU=$MOSES_SCRIPTS/generic/multi-bleu.perl
DETOKENIZER=$MOSES_SCRIPTS/tokenizer/detokenizer.perl
RESULT_DIR=./results

EX_NAME=test
GPU_ID=0
LOAD_NUM=0
while getopts e:g:l: OPT
do
    case $OPT in
    e ) EX_NAME=$OPTARG
        ;;
    
    g ) GPU_ID=$OPTARG
        ;;

    l ) LOAD_NUM=$OPTARG
        ;;
    esac
done

# --- Prepare result text ---
python evaluate.py -e $EX_NAME -g $GPU_ID -l $LOAD_NUM --beam

result_txt=$RESULT_DIR/$EX_NAME/result.txt

detokenized_result_txt=$RESULT_DIR/$EX_NAME/detokenized_result.txt
cat $result_txt | perl $DETOKENIZER -l de > $detokenized_result_txt

ATAT_result_txt=$RESULT_DIR/$EX_NAME/ATAT_result.txt
cat $detokenized_result_txt | perl -ple 's{(\S)-(\S)}{$1 - $2}g' > $ATAT_result_txt

ATAT_reference=$RESULT_DIR/$EX_NAME/ATAT_reference.de
cat $reference | perl -ple 's{(\S)-(\S)}{$1 - $2}g' > $ATAT_reference

cat $ATAT_result_txt | sacrebleu $ATAT_reference -lc


#cat $result_txt | perl -ple 's{(\S)-(\S)}{$1 - $2}g' > $ATAT_result_txt
#cat $my_reference | perl -ple 's{(\S)-(\S)}{$1 - $2}g' > $ATAT_reference
#perl $MULTI_BLEU $ATAT_reference < $ATAT_result_txt
