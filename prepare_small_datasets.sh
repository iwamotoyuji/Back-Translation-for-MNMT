#!/bin/bash

BPE_TOKENS=7000
CLEAN_TOKENS=150
UTILS_DIR=./datasets/utils
RESOURCES_DIR=./datasets/resources
src=en
tgt=de
while getopts b:c:u:r:s:t: OPT
do
    case $OPT in
    b ) BPE_TOKENS=$OPTARG
        ;;
    c ) CLEAN_TOKENS=$OPTARG
        ;;
    u ) UTILS_DIR=$OPTARG
        ;;
    r ) RESOURCES_DIR=$OPTARG
        ;;
    s ) src=$OPTARG
        ;;
    t ) tgt=$OPTARG
        ;;
    esac
done

WORKSPACE_DIR=./datasets/resources/SMALL_BPE${BPE_TOKENS}
rm -rf $WORKSPACE_DIR
coco_py=./datasets/preprocess/prepare_mscoco_data.py
pickle_py=./datasets/preprocess/make_unsuperMMT_data.py

MOSES_SCRIPTS=${RESOURCES_DIR}/multi30k/scripts/moses-3a0631a/tokenizer
LOWERCASE=${MOSES_SCRIPTS}/lowercase.perl
NORM_PUNC=${MOSES_SCRIPTS}/normalize-punctuation.perl
REM_NON_PRINT_CHAR=${MOSES_SCRIPTS}/remove-non-printing-char.perl
TOKENIZER=${MOSES_SCRIPTS}/tokenizer.perl
FASTBPE=${UTILS_DIR}/fastBPE/fast
NUM_WORKERS=${NUM_WORKERS:-16}

# --- Download utils ---
mkdir -p $UTILS_DIR
pushd $UTILS_DIR
if [ -d ./fastBPE ]; then
    echo "[Info] fastBPE already exists, skipping download"
else
    echo "[Info] Cloning fastBPE repository (for BPE pre-processing)..."
    git clone https://github.com/glample/fastBPE.git
fi
if [ -f ./fastBPE/fast ]; then
    echo "[Info] fastBPE already exists, skipping install"
else
    cd ./fastBPE
    g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
    cd ..
    if ! [[ -f ./fastBPE/fast ]]; then
        echo "[Error] fastBPE not successfully installed, abort."
        exit -1
    fi
fi
popd


# --- Download multi30k dataset ---
multi30k_orig=${RESOURCES_DIR}/multi30k/data/task1
multi30k_img_splits=${WORKSPACE_DIR}/multi30k/image_splits
multi30k_tok=${WORKSPACE_DIR}/multi30k/tok
multi30k_bpe=${WORKSPACE_DIR}/multi30k/bpe
mkdir -p $multi30k_img_splits $multi30k_tok $multi30k_bpe

pushd $RESOURCES_DIR
if [ -d multi30k ]; then
    echo "[Info] multi30k dataset already exists, skipping download"
else
    echo "[Info] Cloning multi30k dataset github repository ..."
    git clone https://github.com/multi30k/dataset.git multi30k
fi
popd

for l in $src $tgt; do
    cp -f ${multi30k_orig}/tok/train.lc.norm.tok.$l ${multi30k_tok}/train.$l
    cp -f ${multi30k_orig}/tok/val.lc.norm.tok.$l ${multi30k_tok}/valid.$l
    cp -f ${multi30k_orig}/tok/test_2016_flickr.lc.norm.tok.$l ${multi30k_tok}/test.$l
done
cp -f ${multi30k_orig}/image_splits/train.txt ${multi30k_img_splits}/train.img
cp -f ${multi30k_orig}/image_splits/val.txt ${multi30k_img_splits}/valid.img
cp -f ${multi30k_orig}/image_splits/test_2016_flickr.txt ${multi30k_img_splits}/test.img


# --- Download MSCOCO dataset ---
coco_dir=${RESOURCES_DIR}/COCO
coco_orig=${RESOURCES_DIR}/COCO/orig
coco_img_splits=${WORKSPACE_DIR}/mscoco/image_splits
mkdir -p $coco_dir $coco_orig $coco_img_splits
URLS=(
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    "http://images.cocodataset.org/zips/train2014.zip"
    "http://images.cocodataset.org/zips/val2014.zip"
)
FILES=(
    "annotations_trainval2014.zip"
    "train2014.zip"
    "val2014.zip"
)
DIRS=(
    "annotations"
    "train2014"
    "val2014"
)

pushd $coco_dir
for ((i=0;i<${#URLS[@]};++i)); do
    dir=${DIRS[i]}
    if [ -d $dir ]; then
        echo "[Info] $dir already exists, skipping download"
    else
        url=${URLS[i]}
        file=${FILES[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "[Info] $url successfully downloaded."
        else
            echo "[Error] $url not successfully downloaded."
            exit -1
        fi
        unzip $file
        if [ $file = "val2014.zip" ]; then
            convert ./val2014/COCO_val2014_000000320612.jpg ./val2014/COCO_val2014_000000320612.jpg
        fi
    fi
done
popd

if [ -f ${coco_orig}/train.$src ] && [ -f ${coco_orig}/train.img ] && \
   [ -f ${coco_orig}/valid.$src ] && [ -f ${coco_orig}/valid.img ]; then
    echo "[Info] MSCOCO data already exists, skipping extrant"
else
    echo "[Info] Extracting MSCOCO data..."
    python $coco_py -i $coco_dir/annotations -o $coco_orig
    mv -f ${coco_orig}/val.$src ${coco_orig}/valid.$src
    mv -f ${coco_orig}/val.img ${coco_orig}/valid.img
fi

cp -f ${coco_orig}/train.img ${coco_img_splits}/train.img
cp -f ${coco_orig}/valid.img ${coco_img_splits}/valid.img


# --- Pre-Processing MSCOCO ---
coco_tok=${WORKSPACE_DIR}/mscoco/tok
coco_bpe=${WORKSPACE_DIR}/mscoco/bpe
mkdir -p $coco_tok $coco_bpe

echo "[Info] Pre-processing MSCOCO data..."
for f in train valid; do
    rm -f ${coco_tok}/$f.$src
    cat ${coco_orig}/$f.$src | \
        perl $LOWERCASE | \
        perl $NORM_PUNC $src | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads $NUM_WORKERS -a -l $src >> ${coco_tok}/$f.$src
done


# --- BPE ---
bpe_tmp=${WORKSPACE_DIR}/bpe_tmp
mkdir -p $bpe_tmp

# --- Learn codes ---
BPE_CODE=${bpe_tmp}/codes
cat ${multi30k_tok}/train.$src ${coco_tok}/train.$src > ${bpe_tmp}/multi30k_coco_merge.$src
$FASTBPE learnbpe $BPE_TOKENS ${bpe_tmp}/multi30k_coco_merge.$src ${multi30k_tok}/train.$tgt > $BPE_CODE

# --- Apply codes to train ---
$FASTBPE applybpe ${multi30k_bpe}/train.$src ${multi30k_tok}/train.$src $BPE_CODE
$FASTBPE applybpe ${multi30k_bpe}/train.$tgt ${multi30k_tok}/train.$tgt $BPE_CODE
$FASTBPE applybpe ${coco_bpe}/train.$src ${coco_tok}/train.$src $BPE_CODE

# --- Get train vocabulary ---
cat ${multi30k_bpe}/train.$src ${coco_bpe}/train.$src > ${bpe_tmp}/multi30k_coco_merge_bpe.$src
$FASTBPE getvocab ${bpe_tmp}/multi30k_coco_merge_bpe.$src > ${bpe_tmp}/multi30k_coco_vocab.$src.$BPE_TOKENS
$FASTBPE getvocab ${multi30k_bpe}/train.$tgt > ${bpe_tmp}/vocab.$tgt.$BPE_TOKENS

# --- Apply codes to valid and test ---
for f in valid test; do
    $FASTBPE applybpe ${multi30k_bpe}/$f.$src ${multi30k_tok}/$f.$src $BPE_CODE ${bpe_tmp}/multi30k_coco_vocab.$src.$BPE_TOKENS
    $FASTBPE applybpe ${multi30k_bpe}/$f.$tgt ${multi30k_tok}/$f.$tgt $BPE_CODE ${bpe_tmp}/vocab.$tgt.$BPE_TOKENS
done
$FASTBPE applybpe ${coco_bpe}/valid.$src ${coco_tok}/valid.$src $BPE_CODE ${bpe_tmp}/multi30k_coco_vocab.$src.$BPE_TOKENS


# --- convert to pickle ---
python $pickle_py $WORKSPACE_DIR -d small --bpe $BPE_TOKENS