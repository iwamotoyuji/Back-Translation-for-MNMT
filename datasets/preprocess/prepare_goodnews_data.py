import os
import argparse
import json
from nltk.tokenize import RegexpTokenizer, word_tokenize
#import spacy
import numpy as np
from tqdm import tqdm
#import unidecode
from bs4 import BeautifulSoup
import re
import unicodedata
#from itertools import groupby


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', default="../resources/GoodNews/orig/captioning_dataset.json")
    parser.add_argument('-o', '--output_dir', default="../resources/GoodNews/orig")
    parser.add_argument('--images_dir', default="../resources/GoodNews/images/resized")
    parser.add_argument('--min_len', type=int, default=2) 

    args = parser.parse_args()
    return args

"""
unicodedata.normalize('NFKC', sent)

"""

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


def preprocess_sentence(sen):
    sen = sen.strip()
    sen = denoise_text(sen)
    sen = sen.replace("\n", " ")
    sen = sen.replace("\ufffd\ufffd", " ")
    tokenizer = RegexpTokenizer(r'\w+')
    sen = tokenizer.tokenize(sen.lower())
    #sen = word_tokenize(sen)
    sen = remove_non_ascii(sen)
    return sen
    

def get_split():
    rand = np.random.uniform()
    if rand > 0.95:
        split = 'test'
    elif rand > 0.91 and rand < 0.95:
        split = 'valid'
    else:
        split = 'train'
    return split


def main():
    opt = parse_args()
    np.random.seed(42)
    #print('Loading spacy modules.')
    #nlp = spacy.load('en', disable=['parser', 'tagger'])

    print('Loading the json.')
    with open(opt.input_file, "rb") as f:
        captioning_dataset = json.load(f)
    
    captions = []
    valid_captions = []
    test_captions = []
    filenames = []
    valid_filenames = []
    test_filenames = []
    total_cnt = 0
    no_img_cnt = 0
    no_cap_cnt = 0
    pbar = tqdm(captioning_dataset.items(), ascii=True, mininterval=0.5, ncols=90)
    for k, anns in pbar:
        for ix, cap in anns['images'].items():
            total_cnt += 1

            split = get_split()
            filename = k + '_' + ix + '.jpg'
            cap = preprocess_sentence(cap)
            if len(cap) > opt.min_len:
                if os.path.isfile(opt.images_dir + '/' + filename):
                    cap = ' '.join(cap)
                    captions.append(cap)
                    filenames.append(filename)
                    if split == 'valid':
                        valid_captions.append(cap)
                        valid_filenames.append(filename)
                    elif split == 'test':
                        test_captions.append(cap)
                        test_filenames.append(filename)
                else:
                    no_img_cnt += 1
            else:
                no_cap_cnt += 1

    if len(captions) != len(filenames):
        raise ValueError("The number of training data is not equal")
    if len(valid_captions) != len(valid_filenames):
        raise ValueError("The number of validation data is not equal")
    if len(test_captions) != len(test_filenames):
        raise ValueError("The number of test data is not equal")

    with open(opt.output_dir + '/train_unclean.img', mode='w') as f:
        f.write('\n'.join(filenames))
        f.write('\n')
    with open(opt.output_dir + '/train_unclean.en', mode='w') as f:
        f.write('\n'.join(captions))
        f.write('\n')
    with open(opt.output_dir + '/valid_unclean.img', mode='w') as f:
        f.write('\n'.join(valid_filenames))
        f.write('\n')
    with open(opt.output_dir + '/valid_unclean.en', mode='w') as f:
        f.write('\n'.join(valid_captions))
        f.write('\n')
    with open(opt.output_dir + '/test_unclean.img', mode='w') as f:
        f.write('\n'.join(test_filenames))
        f.write('\n')
    with open(opt.output_dir + '/test_unclean.en', mode='w') as f:
        f.write('\n'.join(test_captions))
        f.write('\n')

    print("total: %d" % total_cnt)
    print("no_cap_cnt: %d" % no_cap_cnt)
    print("no_img_cnt: %d" % no_img_cnt)
    print("train: %d" % len(captions))
    print("valid: %d" % len(valid_captions))
    print("test: %d" % len(test_captions))


if __name__ == '__main__':
    main()
