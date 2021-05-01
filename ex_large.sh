#!/bin/bash

BPE_TOKENS=38000
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

MT_train=MT/train.py
MT_translate=MT/translate.py
MMT_train=MMT/train.py
DAMSM_train=GAN/DAMSM/pretrain_DAMSM.py
T2I_train=GAN/image_generator/train_init_model.py

# --- Train NMT from parallel text data ---
python $MT_train -t 12500 --max_step 100000 -d large -e $EX_NAME -g $GPU_ID --apex --bpe $BPE_TOKENS --overwrite \
                 --check_point_average 5 --grad_accumulation 2 --dropout 0.1 --shared_embedding --share_dec_input_output_embed --init_weight

# --- Generate init triplet by NMT ---
python $MT_translate -e $EX_NAME -g $GPU_ID -l 100000 --beam

# --- Train init T2I from init triplet ---
python $DAMSM_train -m 80 -d large -e en_${EX_NAME} -g $GPU_ID --bpe $BPE_TOKENS --grad_accumulation 2 --overwrite
python $DAMSM_train -m 80 -d large -e de_${EX_NAME} -g $GPU_ID -l de --bpe $BPE_TOKENS --grad_accumulation 2 --overwrite
trained_de_DAMSM_dir=GAN/DAMSM/results/de_${EX_NAME}/trained_models
cp $trained_de_DAMSM_dir/epoch_80.pth $trained_de_DAMSM_dir/best.pth
python $T2I_train -b 64 -m 80 -d large -e $EX_NAME -g $GPU_ID --DAMSM en_${EX_NAME},de_${EX_NAME} --apex --bpe $BPE_TOKENS --overwrite --save_freq 5

# --- Train init MNMT from init triplet ---
python $MMT_train -b 64 --max_epoch 20 -d large -e pre_${EX_NAME} -g $GPU_ID -p --apex --bpe $BPE_TOKENS --overwrite \
                  --grad_accumulation 4 --dropout 0.3

# --- Adapt init MNMT from multi30k ---
python $MMT_train -b 128 --max_epoch 15 -d large -e adapt_init_${EX_NAME} -g $GPU_ID --adapt_init_MMT pre_${EX_NAME},20 --apex \
                  --bpe $BPE_TOKENS --overwrite --dropout 0.3

# --- ReTrain MNMT and T2I ---
python train.py -Mb 32 -Tb 64 -d large -e ${EX_NAME} -g $GPU_ID --MMT pre_${EX_NAME},20 --T2I ${EX_NAME},60 \
                --apex --bpe $BPE_TOKENS --MMT_grad_accumulation 8 --tgt_rnn_fine_tuning --overwrite --dropout 0.1
 