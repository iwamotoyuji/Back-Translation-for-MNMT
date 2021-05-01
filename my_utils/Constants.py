PAD = 0
BOS = 1
EOS = 2
UNK = 3

PAD_WORD = "<pad>"
BOS_WORD = "<bos>"
EOS_WORD = "<eos>"
UNK_WORD = "<unk>"

D_SCALE = {
    "small": {
        "para_text": "multi30k", 
        "img_cap" : "mscoco"},
    "large": {
        "para_text": "wmt",
        "img_cap" : "goodnews"}}

IMAGE_DIRS = {
    "multi30k": {
        "train": "./datasets/resources/multi30k/data/image/flickr30k_images/",
        "valid": "./datasets/resources/multi30k/data/image/flickr30k_images/",
        "eval": "./datasets/resources/multi30k/data/image/flickr30k_images/"},
    "mscoco": {
        "train": "./datasets/resources/COCO/train2014/",
        "valid": "./datasets/resources/COCO/val2014/"},
    "goodnews": {
        "train": "./datasets/resources/GoodNews/images/resized/",
        "valid": "./datasets/resources/GoodNews/images/resized/"}}

CAP_PER_IMG = {
    "multi30k": 1,
    "mscoco": 5,
    "goodnews": 1}


# -- for make init language dictionary --
def init_index2word():
    index2word = {
        PAD: PAD_WORD,
        BOS: BOS_WORD,
        EOS: EOS_WORD,
        UNK: UNK_WORD,
    }
    return index2word

def init_word2index():
    word2index = {
        PAD_WORD: PAD,
        BOS_WORD: BOS,
        EOS_WORD: EOS,
        UNK_WORD: UNK,
    }
    return word2index

def init_word2count():
    word2count = {
        PAD_WORD: 0, 
        BOS_WORD: 0, 
        EOS_WORD: 0, 
        UNK_WORD: 0,
    }
    return word2count

init_n_words = 4


class Lang:
    def __init__(self, name):
        self.name = name
        self.index2word = init_index2word()
        self.word2index = init_word2index()
        self.word2count = init_word2count()
        self.n_words = init_n_words

    def make_dicts_from_word(self, word):
        if not word in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
        
    def make_dicts_from_sent(self, sent):
        for word in sent:
            self.make_dicts_from_word(word)


    def make_dicts_from_word2count(self, ignore_cnt):
        ignored_word_count = 0
        for word, count in self.word2count.items():
            if word not in self.word2index:
                if count > ignore_cnt:
                    self.word2index[word] = self.n_words
                    self.index2word[self.n_words] = word
                    self.n_words += 1
                else:
                    ignored_word_count += 1

        print("[Info] Dictionary for {}".format(self.name))
        print("[Info] Original Vocabulary size = {}".format(len(self.word2count)))
        print("[Info] Trimmed vocabulary size = {}".format(len(self.word2index)))
        print("[Info] Ignored word count = {}".format(ignored_word_count))
        print()
    

    def make_word2count_from_word(self, word):
        if word in self.word2count:
            self.word2count[word] += 1
        else:
            self.word2count[word] = 1

    def make_word2count_from_sent(self, sent):
        for word in sent:
            self.make_word2count_from_word(word)
            
    def increace_BEOS_count(self):
        self.word2count[BOS_WORD] += 1
        self.word2count[EOS_WORD] += 1

    def sent2ids(self, sent):
        return [self.word2index.get(word, UNK) for word in sent]
