import argparse
import json
from nltk.tokenize import RegexpTokenizer, word_tokenize
import unicodedata
import pickle
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', default="../resources/COCO/annotations")
    parser.add_argument('-o', '--output_dir', default="../resources/COCO/orig")
    
    args = parser.parse_args()

    return args


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        #new_word = word.encode('ascii', 'ignore').decode('ascii')
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def preprocess_sentence(sen):
    sen = sen.strip()
    sen = sen.replace("\n", " ")
    sen = sen.replace("\ufffd\ufffd", " ")
    tokenizer = RegexpTokenizer(r'\w+')
    sen = tokenizer.tokenize(sen.lower())
    #sen = word_tokenize(sen)
    sen = remove_non_ascii(sen)
    sen = ' '.join(sen)
    return sen


def main():
    opt = parse_args()

    modes = ["train", "val"]
    for mode in modes:
        captions_source_path = opt.input_dir + f"/captions_{mode}2014.json"
        instances_source_path = opt.input_dir + f"/instances_{mode}2014.json"
    
        with open(captions_source_path) as f:
            x = json.load(f)
            all_caption_data = x['annotations']
            del x
        with open(instances_source_path) as f:
            x = json.load(f)
            all_image_data = x['images']
            del x

        all_images = []
        all_captions = []

        pbar = tqdm(all_image_data, ascii=True, mininterval=0.5, ncols=90)

        for image_data in pbar:
            image_id = image_data['id']
            all_images.append(image_data['file_name'])

            captions = [caption_data['caption'] for caption_data in all_caption_data if caption_data['image_id'] == image_id]
            captions = [preprocess_sentence(caption) for caption in captions]
            assert len(captions) >= 5
            all_captions += captions[:5]

        print(f"all_images : {len(all_images)}")
        print(f"all_captions : {len(all_captions)}")

        save_path = opt.output_dir + f"/{mode}.img"
        with open(save_path, mode='w') as f:
            f.write('\n'.join(all_images))
            f.write('\n')
        print(f"[Info] Writing is complete in {save_path}")

        save_path = opt.output_dir + f"/{mode}.en"
        with open(save_path, mode='w') as f:
            f.write('\n'.join(all_captions))
            f.write('\n')
        print(f"[Info] Writing is complete in {save_path}")

        
if __name__ == '__main__':
    main()
