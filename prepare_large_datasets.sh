#!/bin/bash

BPE_TOKENS=37000
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

WORKSPACE_DIR=./datasets/resources/LARGE_BPE${BPE_TOKENS}_CLEAN${CLEAN_TOKENS}
rm -rf $WORKSPACE_DIR
gn_py=./datasets/preprocess/prepare_goodnews_data.py
pickle_py=./datasets/preprocess/make_unsuperMMT_data.py
CLEAN=./datasets/preprocess/my_clean-corpus-n.perl

MOSES_SCRIPTS=${UTILS_DIR}/mosesdecoder/scripts/tokenizer
LOWERCASE=${MOSES_SCRIPTS}/lowercase.perl
NORM_PUNC=${MOSES_SCRIPTS}/normalize-punctuation.perl
REM_NON_PRINT_CHAR=${MOSES_SCRIPTS}/remove-non-printing-char.perl
TOKENIZER=${MOSES_SCRIPTS}/tokenizer.perl
FASTBPE=${UTILS_DIR}/fastBPE/fast
NUM_WORKERS=${NUM_WORKERS:-16}


# --- Download utils ---
mkdir -p $UTILS_DIR
pushd $UTILS_DIR
if [ -d ./mosesdecoder ]; then
    echo "[Info] mosesdecoder already exists, skipping download"
else
    echo "[Info] Cloning Moses github repository (for tokenization scripts)..."
    git clone https://github.com/moses-smt/mosesdecoder.git    
fi

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


# --- Download wmt14 dataset ---
wmt_orig=${RESOURCES_DIR}/WMT14/orig
mkdir -p $wmt_orig
URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://statmt.org/wmt14/training-parallel-nc-v9.tgz"
    "http://data.statmt.org/wmt17/translation-task/dev.tgz"
    "http://statmt.org/wmt14/test-full.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v9.tgz"
    "dev.tgz"
    "test-full.tgz"
)

pushd $wmt_orig
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "[Info] $file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "[Info] $url successfully downloaded."
        else
            echo "[Error] $url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        fi
    fi
done
popd


# --- Download goodnews dataset ---
gn_orig=${RESOURCES_DIR}/GoodNews/orig
gn_imgs=${RESOURCES_DIR}/GoodNews/images
mkdir -p $gn_orig $gn_imgs

pushd $gn_orig
file="captioning_dataset.json"
if [ -f $file ]; then
    echo "[Info] $file already exists, skipping download"
else
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1rswGdNNfl4HoP9trslP0RUrcmSbg1_RD" > /dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1rswGdNNfl4HoP9trslP0RUrcmSbg1_RD" -o $file
    if [ -f $file ]; then
        echo "[Info] $url successfully downloaded."
    else
        echo "[Error] $url not successfully downloaded."
        exit -1
    fi
fi
popd

pushd $gn_imgs
file="resized.tar.gz"
if [ -f $file ]; then
    echo "[Info] $file already exists, skipping download"
else
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1RF-XlPTNHwwh_XcXE2a1D6Am3fBPvQNy" > /dev/null
    CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1RF-XlPTNHwwh_XcXE2a1D6Am3fBPvQNy" -o $file
    if [ -f $file ]; then
        echo "[Info] $url successfully downloaded."
    else
        echo "[Error] $url not successfully downloaded."
        exit -1
    fi
    tar zxvf $file
fi
popd

if [ -f ${gn_orig}/train_unclean.$src ] && [ -f ${gn_orig}/train_unclean.img ] && \
   [ -f ${gn_orig}/valid_unclean.$src ] && [ -f ${gn_orig}/valid_unclean.img ] && \
   [ -f ${gn_orig}/test_unclean.$src ]  && [ -f ${gn_orig}/test_unclean.img ]; then
    echo "[Info] goodnews data already exists, skipping extrant"
else
    echo "[Info] Extracting GoodNews data..."
    python $gn_py -i ${gn_orig}/captioning_dataset.json -o $gn_orig --images_dir ${gn_imgs}/resized
fi


# --- Download multi30k dataset ---
multi30k_orig=${RESOURCES_DIR}/multi30k/data/task1
multi30k_img_splits=${WORKSPACE_DIR}/multi30k/image_splits
multi30k_tok=${WORKSPACE_DIR}/multi30k/tok
multi30k_bpe=${WORKSPACE_DIR}/multi30k/bpe
mkdir -p $multi30k_img_splits $multi30k_tok $multi30k_bpe

pushd $RESOURCES_DIR
if [ -d ./multi30k ]; then
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
    python $coco_py -i ${coco_dir}/annotations -o $coco_orig
    mv -f ${coco_orig}/val.$src ${coco_orig}/valid.$src
    mv -f ${coco_orig}/val.img ${coco_orig}/valid.img
fi

cp -f ${coco_orig}/train.img ${coco_img_splits}/train.img
cp -f ${coco_orig}/valid.img ${coco_img_splits}/valid.img


# --- Pre-Processing WMT14 ---
wmt_tok=${WORKSPACE_DIR}/wmt/tok
wmt_bpe=${WORKSPACE_DIR}/wmt/bpe
mkdir -p $wmt_tok $wmt_bpe
CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training/news-commentary-v9.de-en"
)

echo "[Info] Pre-processing WMT train data..."
for l in $src $tgt; do
    rm -f ${wmt_tok}/train_unclean.$l
    for f in "${CORPORA[@]}"; do
        cat ${wmt_orig}/$f.$l | \
            perl $LOWERCASE | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads $NUM_WORKERS -a -l $l >> ${wmt_tok}/train_unclean.$l
    done
done

echo "[Info] Pre-processing WMT valid data"
for l in $src $tgt; do
    rm -f ${wmt_tok}/valid.$l
    cat ${wmt_orig}/dev/newstest2013.$l | \
        perl $LOWERCASE | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads $NUM_WORKERS -a -l $l >> ${wmt_tok}/valid.$l
done

echo "[Info] Pre-processing WMT test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' ${wmt_orig}/test-full/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
        perl $LOWERCASE | \
        perl $TOKENIZER -threads $NUM_WORKERS -a -l $l > ${wmt_tok}/test.$l
    echo ""
done


# --- Pre-Processing goodnews data ---
gn_img_splits=${WORKSPACE_DIR}/goodnews/image_splits
gn_tok=${WORKSPACE_DIR}/goodnews/tok
gn_bpe=${WORKSPACE_DIR}/goodnews/bpe
mkdir -p $gn_img_splits $gn_tok $gn_bpe

echo "[Info] Pre-processing GoodNews data..."
for f in train valid test; do
    rm -f ${gn_tok}/${f}_unclean.$src
    cat ${gn_orig}/${f}_unclean.$src | \
        perl $LOWERCASE | \
        perl $NORM_PUNC $src | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads $NUM_WORKERS -a -l $src >> ${gn_tok}/${f}_unclean.$src
done


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
cat ${wmt_tok}/train_unclean.$src ${gn_tok}/train_unclean.$src ${multi30k_tok}/train.$src > ${bpe_tmp}/wmt_gn_multi30k_merge.$src
cat ${wmt_tok}/train_unclean.$tgt ${multi30k_tok}/train.$tgt > ${bpe_tmp}/wmt_multi30k_merge.$tgt
$FASTBPE learnbpe $BPE_TOKENS ${bpe_tmp}/wmt_gn_multi30k_merge.$src ${bpe_tmp}/wmt_multi30k_merge.$tgt > $BPE_CODE

# --- Apply codes to train ---
$FASTBPE applybpe ${wmt_bpe}/train_unclean.$src ${wmt_tok}/train_unclean.$src $BPE_CODE
$FASTBPE applybpe ${wmt_bpe}/train_unclean.$tgt ${wmt_tok}/train_unclean.$tgt $BPE_CODE
$FASTBPE applybpe ${gn_bpe}/train_unclean.$src ${gn_tok}/train_unclean.$src $BPE_CODE
$FASTBPE applybpe ${multi30k_bpe}/train.$src ${multi30k_tok}/train.$src $BPE_CODE
$FASTBPE applybpe ${multi30k_bpe}/train.$tgt ${multi30k_tok}/train.$tgt $BPE_CODE

# --- Get train vocabulary ---
$FASTBPE getvocab ${wmt_bpe}/train_unclean.$src > ${bpe_tmp}/wmt_vocab.$src.$BPE_TOKENS
$FASTBPE getvocab ${wmt_bpe}/train_unclean.$tgt > ${bpe_tmp}/wmt_vocab.$tgt.$BPE_TOKENS
cat ${wmt_bpe}/train_unclean.$src ${gn_bpe}/train_unclean.$src ${multi30k_bpe}/train.$src > ${bpe_tmp}/wmt_gn_multi30k_merge_bpe.$src
cat ${wmt_bpe}/train_unclean.$tgt ${multi30k_bpe}/train.$tgt > ${bpe_tmp}/wmt_multi30k_merge_bpe.$tgt
$FASTBPE getvocab ${bpe_tmp}/wmt_gn_multi30k_merge_bpe.$src > ${bpe_tmp}/wmt_gn_multi30k_vocab.$src.$BPE_TOKENS
$FASTBPE getvocab ${bpe_tmp}/wmt_multi30k_merge_bpe.$tgt > ${bpe_tmp}/wmt_multi30k_vocab.$tgt.$BPE_TOKENS

# --- Apply codes to valid and test ---
for f in valid test; do
    $FASTBPE applybpe ${wmt_bpe}/$f.$src ${wmt_tok}/$f.$src $BPE_CODE ${bpe_tmp}/wmt_vocab.$src.$BPE_TOKENS
    $FASTBPE applybpe ${wmt_bpe}/$f.$tgt ${wmt_tok}/$f.$tgt $BPE_CODE ${bpe_tmp}/wmt_vocab.$tgt.$BPE_TOKENS
    $FASTBPE applybpe ${gn_bpe}/${f}_unclean.$src ${gn_tok}/${f}_unclean.$src $BPE_CODE ${bpe_tmp}/wmt_vocab.$src.$BPE_TOKENS
    $FASTBPE applybpe ${multi30k_bpe}/$f.$src ${multi30k_tok}/$f.$src $BPE_CODE ${bpe_tmp}/wmt_gn_multi30k_vocab.$src.$BPE_TOKENS
    $FASTBPE applybpe ${multi30k_bpe}/$f.$tgt ${multi30k_tok}/$f.$tgt $BPE_CODE ${bpe_tmp}/wmt_multi30k_vocab.$tgt.$BPE_TOKENS
done

# --- Apply codes to MSCOCO ---
$FASTBPE applybpe ${coco_bpe}/train.$src ${coco_tok}/train.$src $BPE_CODE ${bpe_tmp}/wmt_gn_multi30k_vocab.$src.$BPE_TOKENS
$FASTBPE applybpe ${coco_bpe}/valid.$src ${coco_tok}/valid.$src $BPE_CODE ${bpe_tmp}/wmt_gn_multi30k_vocab.$src.$BPE_TOKENS

# --- Clean Corpus ---
echo "[Info] Clean train data..."
perl $CLEAN -ratio 1.5 $wmt_bpe $wmt_tok train_unclean train $src $tgt 2 $CLEAN_TOKENS

for f in train valid test; do
    cp -f ${gn_orig}/${f}_unclean.img ${gn_tok}/${f}_unclean.img
    cp -f ${gn_orig}/${f}_unclean.img ${gn_bpe}/${f}_unclean.img
    perl $CLEAN -ignore-ratio -tgt_eq_img $gn_bpe $gn_tok ${f}_unclean $f $src img 2 $CLEAN_TOKENS
    rm -f ${gn_tok}/${f}_unclean.img
    rm -f ${gn_bpe}/${f}_unclean.img
    mv -f ${gn_tok}/${f}.img $gn_img_splits/${f}.img
    rm -f ${gn_bpe}/${f}.img
done


# --- convert to pickle ---
python $pickle_py $WORKSPACE_DIR -d large -en_i 3 -de_i 3 --bpe $BPE_TOKENS --adapt_t2i_dict_name train_small_data_bpe7000.pickle