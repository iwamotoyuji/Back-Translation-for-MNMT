import sys
import argparse
import pickle
import random
import numpy as np
from pathlib import Path
pardir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(pardir))

## [Self-Module] ########################################################################
import my_utils.Constants as C
from my_utils.general_utils import sort_by_length
#########################################################################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help="Root directory of preprocessed data")
    parser.add_argument('-d', '--d_scale', default="small", help="small or large")
    parser.add_argument('-en_i', '--en_ignore_cnt', type=int, default=1)
    parser.add_argument('-de_i', '--de_ignore_cnt', type=int, default=1)
    parser.add_argument('-r', '--random_seed', type=int, default=42, help="None is not fixed")
    parser.add_argument('--bpe', default=None)
    parser.add_argument('--adapt_t2i_dict_name', default=None)

    args = parser.parse_args()
    if args.d_scale == "large":
        assert args.adapt_t2i_dict_name is not None

    return args


def read_instances(lang1_file_path, lang2_file_path=None, image_file_path=None):
    lang1_word_insts = None 
    lang2_word_insts = None
    image_name_insts = None

    lang1_word_insts = []
    with open(lang1_file_path) as lang1_file:
        for lang1_sent in lang1_file:
            lang1_words = lang1_sent.strip().split(' ')
            if lang1_words:
                lang1_word_insts.append(lang1_words)
    print(f"[Info] {lang1_file_path} : {len(lang1_word_insts)}")

    if lang2_file_path is not None:
        lang2_word_insts = []
        with open(lang2_file_path) as lang2_file:
            for lang2_sent in lang2_file:
                lang2_words = lang2_sent.strip().split(' ')
                if lang2_words:
                    lang2_word_insts.append(lang2_words)
        print(f"[Info] {lang2_file_path} : {len(lang2_word_insts)}")
 
    if image_file_path is not None:
        image_name_insts = []
        with open(image_file_path) as image_file:
            for image_name in image_file:
                image_name = image_name.strip()
                if image_name:
                    image_name_insts.append(image_name)
        print(f"[Info] {image_file_path} : {len(image_name_insts)}")
    
    print()
    return lang1_word_insts, lang2_word_insts, image_name_insts


def main():
    opt = parse_args()

    # -- Random Seed setting --
    if opt.random_seed is None:
        opt.random_seed = random.randint(1, 10000)
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)

    para_text = C.D_SCALE[opt.d_scale]['para_text']
    img_cap = C.D_SCALE[opt.d_scale]['img_cap']
    para_text_dir = f"{opt.data_dir}/{para_text}"
    img_cap_dir = f"{opt.data_dir}/{img_cap}"


    # --- define parallel text data path ---
    para_text_train_en_path = para_text_dir + "/tok/train.en"
    para_text_train_de_path = para_text_dir + "/tok/train.de"
    para_text_valid_de_path = para_text_dir + "/tok/valid.de"
    para_text_eval_de_path = para_text_dir + "/tok/test.de"
    if opt.bpe is None:
        para_text_valid_en_path = para_text_dir + "/tok/valid.en"
        para_text_eval_en_path = para_text_dir + "/tok/test.en"
    else:
        para_text_bpe_train_en_path = para_text_dir + "/bpe/train.en"
        para_text_bpe_train_de_path = para_text_dir + "/bpe/train.de"
        para_text_bpe_valid_en_path = para_text_dir + "/bpe/valid.en"
        para_text_bpe_eval_en_path = para_text_dir + "/bpe/test.en"


    # --- define image-caption data path ---
    img_cap_train_img_path = img_cap_dir + "/image_splits/train.img"
    img_cap_train_en_path = img_cap_dir + "/tok/train.en"
    img_cap_valid_img_path = img_cap_dir + "/image_splits/valid.img"
    img_cap_valid_en_path = img_cap_dir + "/tok/valid.en"
    if opt.bpe is not None:
        img_cap_bpe_train_en_path = img_cap_dir + "/bpe/train.en"
        img_cap_bpe_valid_en_path = img_cap_dir + "/bpe/valid.en"


    # --- define parallel text data's image path ---
    if para_text == "multi30k":
        para_text_train_img_path = para_text_dir + "/image_splits/train.img"
        para_text_valid_img_path = para_text_dir + "/image_splits/valid.img"
        para_text_eval_img_path = para_text_dir + "/image_splits/test.img"
    else:
        para_text_train_img_path = None
        para_text_valid_img_path = None
        para_text_eval_img_path = None


    # --- define adaptive data path ---
    if opt.d_scale == "large":
        adapt_para_text_dir = f"{opt.data_dir}/multi30k"
        adapt_img_cap_dir = f"{opt.data_dir}/mscoco"

        adapt_para_text_train_img_path = adapt_para_text_dir + "/image_splits/train.img"
        adapt_para_text_train_en_path = adapt_para_text_dir + "/tok/train.en"
        adapt_para_text_train_de_path = adapt_para_text_dir + "/tok/train.de"   
        adapt_para_text_valid_img_path = adapt_para_text_dir + "/image_splits/valid.img"
        adapt_para_text_valid_de_path = adapt_para_text_dir + "/tok/valid.de"     
        adapt_para_text_eval_img_path = adapt_para_text_dir + "/image_splits/test.img"
        adapt_para_text_eval_de_path = adapt_para_text_dir + "/tok/test.de"

        adapt_img_cap_train_img_path = adapt_img_cap_dir + "/image_splits/train.img"
        adapt_img_cap_train_en_path = adapt_img_cap_dir + "/tok/train.en"

        if opt.bpe is None:
            adapt_para_text_valid_en_path = adapt_para_text_dir + "/tok/valid.en"
            adapt_para_text_eval_en_path = adapt_para_text_dir + "/tok/test.en"
        else:
            adapt_para_text_bpe_train_en_path = adapt_para_text_dir + "/bpe/train.en"
            adapt_para_text_bpe_train_de_path = adapt_para_text_dir + "/bpe/train.de"
            adapt_para_text_bpe_valid_en_path = adapt_para_text_dir + "/bpe/valid.en"
            adapt_para_text_bpe_eval_en_path = adapt_para_text_dir + "/bpe/test.en"
            adapt_img_cap_bpe_train_en_path = adapt_img_cap_dir + "/bpe/train.en"


    # --- Load parallel text data ---
    print("[Info] Loading data...")
    para_text_train_en_words, para_text_train_de_words, para_text_train_imgs = read_instances(
        lang1_file_path=para_text_train_en_path,
        lang2_file_path=para_text_train_de_path,
        image_file_path=para_text_train_img_path
    )
    if opt.bpe is None:
        para_text_valid_en_words, para_text_valid_de_words, para_text_valid_imgs = read_instances(
            lang1_file_path=para_text_valid_en_path,
            lang2_file_path=para_text_valid_de_path,
            image_file_path=para_text_valid_img_path
        )
        para_text_eval_en_words, para_text_eval_de_words, para_text_eval_imgs = read_instances(
            lang1_file_path=para_text_eval_en_path,
            lang2_file_path=para_text_eval_de_path,
            image_file_path=para_text_eval_img_path
        )
    else:
        para_text_bpe_train_en_words, para_text_bpe_train_de_words, _ = read_instances(
            lang1_file_path=para_text_bpe_train_en_path,
            lang2_file_path=para_text_bpe_train_de_path
        )
        para_text_bpe_valid_en_words, para_text_valid_de_words, para_text_valid_imgs = read_instances(
            lang1_file_path=para_text_bpe_valid_en_path,
            lang2_file_path=para_text_valid_de_path,
            image_file_path=para_text_valid_img_path
        )
        para_text_bpe_eval_en_words, para_text_eval_de_words, para_text_eval_imgs = read_instances(
            lang1_file_path=para_text_bpe_eval_en_path,
            lang2_file_path=para_text_eval_de_path,
            image_file_path=para_text_eval_img_path
        )


    # --- Load image-caption data ---
    img_cap_train_en_words, _, img_cap_train_imgs = read_instances(
        lang1_file_path=img_cap_train_en_path,
        image_file_path=img_cap_train_img_path
    )
    img_cap_valid_en_words, _, img_cap_valid_imgs = read_instances(
        lang1_file_path=img_cap_valid_en_path,
        image_file_path=img_cap_valid_img_path
    )
    if opt.bpe is not None:
        img_cap_bpe_train_en_words, _, _ = read_instances(img_cap_bpe_train_en_path)
        img_cap_bpe_valid_en_words, _, _ = read_instances(img_cap_bpe_valid_en_path)


    # --- Load adaptive data ---
    if opt.d_scale == "large":
        adapt_para_text_train_en_words, adapt_para_text_train_de_words, adapt_para_text_train_imgs = read_instances(
            lang1_file_path=adapt_para_text_train_en_path,
            lang2_file_path=adapt_para_text_train_de_path,
            image_file_path=adapt_para_text_train_img_path
        )    
        if opt.bpe is None:
            adapt_para_text_valid_en_words, adapt_para_text_valid_de_words, adapt_para_text_valid_imgs = read_instances(
                lang1_file_path=adapt_para_text_valid_en_path,
                lang2_file_path=adapt_para_text_valid_de_path,
                image_file_path=adapt_para_text_valid_img_path
            )
            adapt_para_text_eval_en_words, adapt_para_text_eval_de_words, adapt_para_text_eval_imgs = read_instances(
                lang1_file_path=adapt_para_text_eval_en_path,
                lang2_file_path=adapt_para_text_eval_de_path,
                image_file_path=adapt_para_text_eval_img_path
            )
        else:
            adapt_para_text_bpe_train_en_words, adapt_para_text_bpe_train_de_words, _ = read_instances(
                lang1_file_path=adapt_para_text_bpe_train_en_path,
                lang2_file_path=adapt_para_text_bpe_train_de_path
            )
            adapt_para_text_bpe_valid_en_words, adapt_para_text_valid_de_words, adapt_para_text_valid_imgs = read_instances(
                lang1_file_path=adapt_para_text_bpe_valid_en_path,
                lang2_file_path=adapt_para_text_valid_de_path,
                image_file_path=adapt_para_text_valid_img_path
            )
            adapt_para_text_bpe_eval_en_words, adapt_para_text_eval_de_words, adapt_para_text_eval_imgs = read_instances(
                lang1_file_path=adapt_para_text_bpe_eval_en_path,
                lang2_file_path=adapt_para_text_eval_de_path,
                image_file_path=adapt_para_text_eval_img_path
            )
        adapt_img_cap_train_en_words, _, adapt_img_cap_train_imgs = read_instances(
            lang1_file_path=adapt_img_cap_train_en_path,
            image_file_path=adapt_img_cap_train_img_path
        )
        if opt.bpe is not None:
            adapt_img_cap_bpe_train_en_words, _, _ = read_instances(adapt_img_cap_bpe_train_en_path)


    # --- Create dictionaly ---
    print("[Info] Creating dictionaly...")
    en_lang = C.Lang('en')
    for sent in para_text_train_en_words:
        en_lang.make_word2count_from_sent(sent)
    for sent in img_cap_train_en_words:
        en_lang.make_word2count_from_sent(sent)

    de_lang = C.Lang('de')
    for sent in para_text_train_de_words:
        de_lang.make_word2count_from_sent(sent)

    if opt.bpe is not None:
        bpe_lang = C.Lang('bpe')
        for sent in para_text_bpe_train_en_words:
            bpe_lang.make_word2count_from_sent(sent)
        for sent in img_cap_bpe_train_en_words:
            bpe_lang.make_word2count_from_sent(sent)
        for sent in para_text_bpe_train_de_words:
            bpe_lang.make_word2count_from_sent(sent)
    
    if opt.d_scale == "large":
        if opt.bpe is None:
            for sent in adapt_para_text_train_en_words:
                en_lang.make_word2count_from_sent(sent)
            for sent in adapt_para_text_train_de_words:
                de_lang.make_word2count_from_sent(sent)
        else:
            for sent in adapt_para_text_bpe_train_en_words:
                bpe_lang.make_word2count_from_sent(sent)
            for sent in adapt_para_text_bpe_train_de_words:
                bpe_lang.make_word2count_from_sent(sent)

        adapt_t2i_dict_path = pardir / f"datasets/resources/{opt.adapt_t2i_dict_name}"
        with open(adapt_t2i_dict_path, 'rb') as f:
            x = pickle.load(f)
        t2i_en_index2word = x['index2word']['en']
        t2i_de_index2word = x['index2word']['de']
        t2i_en_word2index = x['word2index']['en']
        t2i_de_word2index = x['word2index']['de']
        del x

    en_lang.make_dicts_from_word2count(opt.en_ignore_cnt)
    de_lang.make_dicts_from_word2count(opt.de_ignore_cnt)
    if opt.bpe is not None:
        bpe_lang.make_dicts_from_word2count(0)        


    # --- Sort by length ---
    if opt.d_scale == "large":
        print("[Info] Sorting data by length...")
        if opt.bpe is None:
            para_text_train_de_words, para_text_train_en_words, para_text_train_imgs = sort_by_length(
                first_key_list=para_text_train_de_words,
                second_key_list=para_text_train_en_words,
                other_apply_lists=para_text_train_imgs
            )
            para_text_valid_en_words, para_text_valid_de_words, para_text_valid_imgs = sort_by_length(
                first_key_list=para_text_valid_en_words,
                second_key_list=para_text_valid_de_words,
                other_apply_lists=para_text_valid_imgs
            )
            """
            para_text_eval_en_words, para_text_eval_de_words, para_text_eval_imgs = sort_by_length(
                first_key_list=para_text_eval_en_words,
                second_key_list=para_text_eval_de_words,
                other_apply_lists=para_text_eval_imgs
            )
            """

            if C.CAP_PER_IMG[img_cap] == 1:
                img_cap_train_en_words, _, img_cap_train_imgs = sort_by_length(
                    first_key_list=img_cap_train_en_words,
                    second_key_list=None,
                    other_apply_lists=img_cap_train_imgs
                )
                img_cap_valid_en_words, _, img_cap_valid_imgs = sort_by_length(
                    first_key_list=img_cap_valid_en_words,
                    second_key_list=None,
                    other_apply_lists=img_cap_valid_imgs
                )
        else:
            para_text_bpe_train_de_words, para_text_bpe_train_en_words, \
            (para_text_train_de_words, para_text_train_en_words, para_text_train_imgs) = sort_by_length(
                first_key_list=para_text_bpe_train_de_words,
                second_key_list=para_text_bpe_train_en_words,
                other_apply_lists=(para_text_train_de_words, para_text_train_en_words, para_text_train_imgs)
            )
            para_text_bpe_valid_en_words, para_text_valid_de_words, para_text_valid_imgs = sort_by_length(
                first_key_list=para_text_bpe_valid_en_words,
                second_key_list=para_text_valid_de_words,
                other_apply_lists=para_text_valid_imgs
            )
            """
            para_text_bpe_eval_en_words, para_text_eval_de_words, para_text_eval_imgs = sort_by_length(
                first_key_list=para_text_bpe_eval_en_words,
                second_key_list=para_text_eval_de_words,
                other_apply_lists=para_text_eval_imgs
            )
            """

            if C.CAP_PER_IMG[img_cap] == 1:
                img_cap_bpe_train_en_words, _, \
                (img_cap_train_en_words, img_cap_train_imgs) = sort_by_length(
                    first_key_list=img_cap_bpe_train_en_words,
                    second_key_list=None,
                    other_apply_lists=(img_cap_train_en_words, img_cap_train_imgs)
                )
                img_cap_valid_en_words, _, img_cap_valid_imgs = sort_by_length(
                    first_key_list=img_cap_valid_en_words,
                    second_key_list=None,
                    other_apply_lists=img_cap_valid_imgs
                )


    # --- Word to Index ---
    num_para_text_train_imgs = None if para_text_train_imgs is None else len(para_text_train_imgs)
    num_para_text_valid_imgs = None if para_text_valid_imgs is None else len(para_text_valid_imgs)
    num_para_text_eval_imgs = None if para_text_eval_imgs is None else len(para_text_eval_imgs)

    print("[Info] Converting word to index...")
    para_text_train_en_ids = [en_lang.sent2ids(sent) for sent in para_text_train_en_words]
    para_text_train_de_ids = [de_lang.sent2ids(sent) for sent in para_text_train_de_words]
    img_cap_train_en_ids = [en_lang.sent2ids(sent) for sent in img_cap_train_en_words]
    img_cap_valid_en_ids = [en_lang.sent2ids(sent) for sent in img_cap_valid_en_words]

    if opt.bpe is None:
        para_text_valid_en_ids = [en_lang.sent2ids(sent) for sent in para_text_valid_en_words]
        para_text_eval_en_ids = [en_lang.sent2ids(sent) for sent in para_text_eval_en_words]
        print(f"[Info] {para_text} images for train : {num_para_text_train_imgs}")
        print(f"[Info] {para_text} en data for train : {len(para_text_train_en_ids)}")
        print(f"[Info] {para_text} de data for train : {len(para_text_train_de_ids)}")
        print(f"[Info] {para_text} images for valid : {num_para_text_valid_imgs}")
        print(f"[Info] {para_text} en data for valid : {len(para_text_valid_en_ids)}")
        print(f"[Info] {para_text} de data for valid : {len(para_text_valid_de_words)}")
        print(f"[Info] {para_text} images for eval : {num_para_text_eval_imgs}")
        print(f"[Info] {para_text} en data for eval : {len(para_text_eval_en_ids)}")
        print(f"[Info] {para_text} de data for eval : {len(para_text_eval_de_words)}")
        print(f"[Info] {img_cap} images for train : {len(img_cap_train_imgs)}")
        print(f"[Info] {img_cap} caption data for train : {len(img_cap_train_en_ids)}")
        print(f"[Info] {img_cap} image for valid : {len(img_cap_valid_imgs)}")
        print(f"[Info] {img_cap} caption data for valid : {len(img_cap_valid_en_ids)}")
    else:
        para_text_bpe_train_en_ids = [bpe_lang.sent2ids(sent) for sent in para_text_bpe_train_en_words]
        para_text_bpe_train_de_ids = [bpe_lang.sent2ids(sent) for sent in para_text_bpe_train_de_words]
        para_text_bpe_valid_en_ids = [bpe_lang.sent2ids(sent) for sent in para_text_bpe_valid_en_words]
        para_text_bpe_eval_en_ids = [bpe_lang.sent2ids(sent) for sent in para_text_bpe_eval_en_words]
        img_cap_bpe_train_en_ids = [bpe_lang.sent2ids(sent) for sent in img_cap_bpe_train_en_words]
        img_cap_bpe_valid_en_ids = [bpe_lang.sent2ids(sent) for sent in img_cap_bpe_valid_en_words]
        print(f"[Info] {para_text} images for train : {num_para_text_train_imgs}")
        print(f"[Info] {para_text} en data for train : {len(para_text_bpe_train_en_ids)}")
        print(f"[Info] {para_text} de data for train : {len(para_text_bpe_train_de_ids)}")
        print(f"[Info] {para_text} images for valid : {num_para_text_valid_imgs}")
        print(f"[Info] {para_text} en data for valid : {len(para_text_bpe_valid_en_ids)}")
        print(f"[Info] {para_text} de data for valid : {len(para_text_valid_de_words)}")
        print(f"[Info] {para_text} images for eval : {num_para_text_eval_imgs}")
        print(f"[Info] {para_text} en data for eval : {len(para_text_bpe_eval_en_ids)}")
        print(f"[Info] {para_text} de data for eval : {len(para_text_eval_de_words)}")
        print(f"[Info] {img_cap} images for train : {len(img_cap_train_imgs)}")
        print(f"[Info] {img_cap} caption data for train : {len(img_cap_bpe_train_en_ids)}")
        print(f"[Info] {img_cap} image for valid : {len(img_cap_valid_imgs)}")
        print(f"[Info] {img_cap} caption data for valid : {len(img_cap_bpe_valid_en_ids)}")

    if opt.d_scale == "large":
        adapt_para_text_t2i_train_en_ids = [[t2i_en_word2index.get(word, C.UNK) for word in sent] for sent in adapt_para_text_train_en_words]
        adapt_para_text_t2i_train_de_ids = [[t2i_de_word2index.get(word, C.UNK) for word in sent] for sent in adapt_para_text_train_de_words]
        adapt_img_cap_t2i_train_en_ids = [[t2i_en_word2index.get(word, C.UNK) for word in sent] for sent in adapt_img_cap_train_en_words]

        if opt.bpe is None:
            adapt_para_text_mmt_train_en_ids = [en_lang.sent2ids(sent) for sent in adapt_para_text_train_en_words]
            adapt_para_text_mmt_train_de_ids = [de_lang.sent2ids(sent) for sent in adapt_para_text_train_de_words]
            adapt_para_text_mmt_valid_en_ids = [en_lang.sent2ids(sent) for sent in adapt_para_text_valid_en_words]
            adapt_para_text_mmt_eval_en_ids = [en_lang.sent2ids(sent) for sent in adapt_para_text_eval_en_words]
            adapt_img_cap_mmt_train_en_ids = [en_lang.sent2ids(sent) for sent in adapt_img_cap_train_en_words]
            print(f"[Info] multi30k images for adaptive train : {len(adapt_para_text_train_imgs)}")
            print(f"[Info] multi30k en data for adaptive train : {len(adapt_para_text_mmt_train_en_ids)}")
            print(f"[Info] multi30k de data for adaptive train : {len(adapt_para_text_mmt_train_de_ids)}")
            print(f"[Info] multi30k images for adaptive valid : {len(adapt_para_text_valid_imgs)}")
            print(f"[Info] multi30k en data for adaptive valid : {len(adapt_para_text_mmt_valid_en_ids)}")
            print(f"[Info] multi30k de data for adaptive valid : {len(adapt_para_text_valid_de_words)}")
            print(f"[Info] multi30k images for adaptive eval : {len(adapt_para_text_eval_imgs)}")
            print(f"[Info] multi30k en data for adaptive eval : {len(adapt_para_text_mmt_eval_en_ids)}")
            print(f"[Info] multi30k de data for adaptive eval : {len(adapt_para_text_eval_de_words)}")
            print(f"[Info] mscoco images for train : {len(adapt_img_cap_train_imgs)}")
            print(f"[Info] mscoco caption data for train : {len(adapt_img_cap_mmt_train_en_ids)}")
        else:
            adapt_para_text_bpe_train_en_ids = [bpe_lang.sent2ids(sent) for sent in adapt_para_text_bpe_train_en_words]
            adapt_para_text_bpe_train_de_ids = [bpe_lang.sent2ids(sent) for sent in adapt_para_text_bpe_train_de_words]
            adapt_para_text_bpe_valid_en_ids = [bpe_lang.sent2ids(sent) for sent in adapt_para_text_bpe_valid_en_words]
            adapt_para_text_bpe_eval_en_ids = [bpe_lang.sent2ids(sent) for sent in adapt_para_text_bpe_eval_en_words]
            adapt_img_cap_bpe_train_en_ids = [bpe_lang.sent2ids(sent) for sent in adapt_img_cap_bpe_train_en_words]
            print(f"[Info] multi30k images for adaptive train : {len(adapt_para_text_train_imgs)}")
            print(f"[Info] multi30k en data for adaptive train : {len(adapt_para_text_bpe_train_en_ids)}")
            print(f"[Info] multi30k de data for adaptive train : {len(adapt_para_text_bpe_train_de_ids)}")
            print(f"[Info] multi30k images for adaptive valid : {len(adapt_para_text_valid_imgs)}")
            print(f"[Info] multi30k en data for adaptive valid : {len(adapt_para_text_bpe_valid_en_ids)}")
            print(f"[Info] multi30k de data for adaptive valid : {len(adapt_para_text_valid_de_words)}")
            print(f"[Info] multi30k images for adaptive eval : {len(adapt_para_text_eval_imgs)}")
            print(f"[Info] multi30k en data for adaptive eval : {len(adapt_para_text_bpe_eval_en_ids)}")
            print(f"[Info] multi30k de data for adaptive eval : {len(adapt_para_text_eval_de_words)}")
            print(f"[Info] mscoco images for train : {len(adapt_img_cap_train_imgs)}")
            print(f"[Info] mscoco caption data for train : {len(adapt_img_cap_bpe_train_en_ids)}")
    print()


    train_save_dict = {
        para_text: {
            'img': para_text_train_imgs,
            'en': {
                'id': para_text_train_en_ids},
            'de': {
                'id': para_text_train_de_ids}},
        img_cap: {
            'img': img_cap_train_imgs,
            'en': {
                'id': img_cap_train_en_ids},
            'de': {}},
        'index2word': {
            'en': en_lang.index2word,
            'de': de_lang.index2word},
        'word2index': {
            'en': en_lang.word2index,
            'de': de_lang.word2index}}

    valid_save_dict = {
        para_text: {
            'img': para_text_valid_imgs,
            'en': {},
            'de': {
                'word': para_text_valid_de_words}},
        img_cap: {
            'img': img_cap_valid_imgs,
            'en': {
                'id': img_cap_valid_en_ids}},
        'index2word': {
            'en': en_lang.index2word,
            'de': de_lang.index2word},
        'word2index': {
            'en': en_lang.word2index,
            'de': de_lang.word2index}}

    eval_save_dict = {
        para_text: {
            'img': para_text_eval_imgs,
            'en':{},
            'de': {
                'word': para_text_eval_de_words}},
        'index2word': {
            'en': en_lang.index2word,
            'de': de_lang.index2word},
        'word2index': {
            'en': en_lang.word2index,
            'de': de_lang.word2index}}

    if opt.bpe is None:
        valid_save_dict[para_text]['en']['id'] = para_text_valid_en_ids
        eval_save_dict[para_text]['en']['id'] = para_text_eval_en_ids
    else:
        train_save_dict[para_text]['en']['bpe'] = para_text_bpe_train_en_ids
        train_save_dict[para_text]['de']['bpe'] = para_text_bpe_train_de_ids
        valid_save_dict[para_text]['en']['bpe'] = para_text_bpe_valid_en_ids
        eval_save_dict[para_text]['en']['bpe'] = para_text_bpe_eval_en_ids
        train_save_dict[img_cap]['en']['bpe'] = img_cap_bpe_train_en_ids
        valid_save_dict[img_cap]['en']['bpe'] = img_cap_bpe_valid_en_ids
        train_save_dict['index2word']['bpe'] = bpe_lang.index2word
        train_save_dict['word2index']['bpe'] = bpe_lang.word2index
        valid_save_dict['index2word']['bpe'] = bpe_lang.index2word
        valid_save_dict['word2index']['bpe'] = bpe_lang.word2index
        eval_save_dict['index2word']['bpe'] = bpe_lang.index2word
        eval_save_dict['word2index']['bpe'] = bpe_lang.word2index


    if opt.d_scale == "large":
        adapt_train_save_dict = {
            'multi30k': {
                'img': adapt_para_text_train_imgs,
                'en': {
                    't2i_id': adapt_para_text_t2i_train_en_ids},
                'de': {
                    't2i_id': adapt_para_text_t2i_train_de_ids}},
            'mscoco': {
                'img': adapt_img_cap_train_imgs,
                'en': {
                    't2i_id': adapt_img_cap_t2i_train_en_ids}},
            'index2word': {
                't2i_en': t2i_en_index2word,
                't2i_de': t2i_de_index2word},
            'word2index': {
                't2i_en': t2i_en_word2index,
                't2i_de': t2i_de_word2index}}

        adapt_valid_save_dict = {
            'multi30k': {
                'img': adapt_para_text_valid_imgs,
                'en': {},
                'de': {
                    'word': adapt_para_text_valid_de_words}},
            'index2word': {
                't2i_en': t2i_en_index2word,
                't2i_de': t2i_de_index2word},
            'word2index': {
                't2i_en': t2i_en_word2index,
                't2i_de': t2i_de_word2index}}

        adapt_eval_save_dict = {
            'multi30k': {
                'img': adapt_para_text_eval_imgs,
                'en': {},
                'de': {
                    'word': adapt_para_text_eval_de_words}},
            'index2word': {
                't2i_en': t2i_en_index2word,
                't2i_de': t2i_de_index2word},
            'word2index': {
                't2i_en': t2i_en_word2index,
                't2i_de': t2i_de_word2index}}

        if opt.bpe is None:
            adapt_train_save_dict['multi30k']['en']['mmt_id'] = adapt_para_text_mmt_train_en_ids
            adapt_train_save_dict['multi30k']['de']['mmt_id'] = adapt_para_text_mmt_train_de_ids
            adapt_train_save_dict['mscoco']['en']['mmt_id'] = adapt_img_cap_mmt_train_en_ids
            adapt_train_save_dict['index2word']['mmt_en'] = en_lang.index2word
            adapt_train_save_dict['index2word']['mmt_de'] = de_lang.index2word
            adapt_train_save_dict['word2index']['mmt_en'] = en_lang.word2index
            adapt_train_save_dict['word2index']['mmt_de'] = de_lang.word2index
            adapt_valid_save_dict['multi30k']['en']['mmt_id'] = adapt_para_text_mmt_valid_en_ids
            adapt_valid_save_dict['index2word']['mmt_en'] = en_lang.index2word
            adapt_valid_save_dict['index2word']['mmt_de'] = de_lang.index2word
            adapt_valid_save_dict['word2index']['mmt_en'] = en_lang.word2index
            adapt_valid_save_dict['word2index']['mmt_de'] = de_lang.word2index
            adapt_eval_save_dict['multi30k']['en']['mmt_id'] = adapt_para_text_mmt_eval_en_ids
            adapt_eval_save_dict['index2word']['mmt_en'] = en_lang.index2word
            adapt_eval_save_dict['index2word']['mmt_de'] = de_lang.index2word
            adapt_eval_save_dict['word2index']['mmt_en'] = en_lang.word2index
            adapt_eval_save_dict['word2index']['mmt_de'] = de_lang.word2index
        else:
            adapt_train_save_dict['multi30k']['en']['bpe'] = adapt_para_text_bpe_train_en_ids
            adapt_train_save_dict['multi30k']['de']['bpe'] = adapt_para_text_bpe_train_de_ids
            adapt_train_save_dict['mscoco']['en']['bpe'] = adapt_img_cap_bpe_train_en_ids
            adapt_train_save_dict['index2word']['bpe'] = bpe_lang.index2word
            adapt_train_save_dict['word2index']['bpe'] = bpe_lang.word2index
            adapt_valid_save_dict['multi30k']['en']['bpe'] = adapt_para_text_bpe_valid_en_ids
            adapt_valid_save_dict['index2word']['bpe'] = bpe_lang.index2word
            adapt_valid_save_dict['word2index']['bpe'] = bpe_lang.word2index
            adapt_eval_save_dict['multi30k']['en']['bpe'] = adapt_para_text_bpe_eval_en_ids
            adapt_eval_save_dict['index2word']['bpe'] = bpe_lang.index2word
            adapt_eval_save_dict['word2index']['bpe'] = bpe_lang.word2index


    save_file_name = f"{opt.d_scale}_data_bpe{opt.bpe}.pickle"
    save_dir = pardir / "datasets/resources"
    save_path = save_dir / ("train_" + save_file_name)
    with open(save_path, mode='wb') as f:
        pickle.dump(train_save_dict, f)
        print(f"[Info] Dumped data to {save_path}")
    save_path = save_dir / ("valid_" + save_file_name)
    with open(save_path, mode='wb') as f:
        pickle.dump(valid_save_dict, f)
        print(f"[Info] Dumped data to {save_path}")
    save_path = save_dir / ("eval_" + save_file_name)
    with open(save_path, mode='wb') as f:
        pickle.dump(eval_save_dict, f)
        print(f"[Info] Dumped data to {save_path}")
    
    if opt.d_scale == "large":
        save_path = save_dir / ("adapt_train_" + save_file_name)
        with open(save_path, mode='wb') as f:
            pickle.dump(adapt_train_save_dict, f)
            print(f"[Info] Dumped data to {save_path}")
        save_path = save_dir / ("adapt_valid_" + save_file_name)
        with open(save_path, mode='wb') as f:
            pickle.dump(adapt_valid_save_dict, f)
            print(f"[Info] Dumped data to {save_path}")
        save_path = save_dir / ("adapt_eval_" + save_file_name)
        with open(save_path, mode='wb') as f:
            pickle.dump(adapt_eval_save_dict, f)
            print(f"[Info] Dumped data to {save_path}")

if __name__ == "__main__":
    main()