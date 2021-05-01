#!/bin/bash

BPE_TOKENS=7000
EX_NAME=test
GPU_ID=0

while getopts b:e:g: OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    e ) EX_NAME=$OPTARG
        ;;
    g ) GPU_ID=$OPTARG
        ;;
    esac
done

NMT_train=NMT/train.py
NMT_translate=NMT/translate.py
MNMT_train=MNMT/train.py
DAMSM_train=T2I/DAMSM/pretrain_DAMSM.py
T2I_train=T2I/image_generator/train_init_model.py

# --- Train NMT from parallel text data ---
python $NMT_train -b 128 --max_epoch 40 -e $EX_NAME -g $GPU_ID --bpe $BPE_TOKENS --overwrite --dropout 0.3 --use_beam

# --- Train MNMT ---
python $MNMT_train -b 128 --max_epoch 40 -e $EX_NAME -g $GPU_ID --bpe $BPE_TOKENS --overwrite --dropout 0.3 --use_beam

# --- Generate init triplet by NMT ---
python $NMT_translate -e $EX_NAME -g $GPU_ID --use_beam

# --- Train init T2I from init triplet ---
python $DAMSM_train -e en_${EX_NAME} -g $GPU_ID --bpe $BPE_TOKENS --overwrite --use_amp
python $DAMSM_train -e de_${EX_NAME} -g $GPU_ID -l de --bpe $BPE_TOKENS --overwrite --use_amp
python $T2I_train -b 32 -e ${EX_NAME} -g $GPU_ID --DAMSM en_${EX_NAME},de_${EX_NAME} --bpe $BPE_TOKENS --bilingual --overwrite --use_amp

# --- Train init MNMT from init triplet ---
python $MNMT_train -b 128 --max_epoch 25 -e pre_${EX_NAME} -g $GPU_ID -p --bpe $BPE_TOKENS --overwrite --use_amp --use_beam --dropout 0.3

# --- ReTrain MNMT and T2I ---
python train.py -Mb 128 -Tb 32 -e ${EX_NAME} -g $GPU_ID --MNMT pre_${EX_NAME},25 --T2I ${EX_NAME},100 \
                --bpe $BPE_TOKENS --overwrite --use_amp --use_beam --tgt_rnn_fine_tuning