import numpy as np
import random
import pickle

## [Pytorch] ############################################################################
import torch
from torch.utils.data import DataLoader
#########################################################################################

## [Self-Module] ########################################################################
from datasets.base_dataset import BaseT2IDataset
import my_utils.Constants as Constants
#########################################################################################


def get_train_loader(opt, base_size=64, trans_norm=None, drop_last=True):
    def worker_init_fn(worker_id):
        random.seed(worker_id + opt.random_seed)
        np.random.seed(worker_id + opt.random_seed)

    train_dataset = DAMSMDataset(
        data_path=opt.data_path,
        words_limit=opt.words_limit,
        base_size=base_size,
        stage_num=opt.stage_num,
        trans_norm=trans_norm,
        lang=opt.lang,
        mode="train",
        scale=opt.d_scale,
        use_memo=opt.use_memo,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size,
        shuffle=True, drop_last=drop_last,
        num_workers=opt.workers, worker_init_fn=worker_init_fn,
    )
    """
        train_batch_sampler = MyBatchSampler(
            sorted_insts=train_dataset.text_insts, batch_size=opt.batch_size,
            shuffle=True, drop_last=drop_last,
        )
        train_loader = DataLoader(
            train_dataset, batch_sampler=train_batch_sampler,
            num_workers=opt.workers, worker_init_fn=worker_init_fn
        )
    """

    return train_loader


def get_valid_loader(opt, base_size=64, trans_norm=None, shuffle=False, drop_last=True):
    def worker_init_fn(worker_id):
        random.seed(worker_id + opt.random_seed)
        np.random.seed(worker_id + opt.random_seed)

    if opt.lang == 'en' or Constants.D_SCALE[opt.d_scale]['para_text'] == "multi30k":
        valid_dataset = DAMSMDataset(
            data_path=opt.data_path,
            words_limit=opt.words_limit,
            base_size=base_size,
            stage_num=opt.stage_num,
            trans_norm=trans_norm,
            lang=opt.lang,
            mode="valid",
            scale=opt.d_scale,
            use_memo=False
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=opt.batch_size,
            shuffle=shuffle, drop_last=drop_last,
            num_workers=opt.workers, worker_init_fn=worker_init_fn
        )
    else:
        valid_loader = None
    
    return valid_loader
    

class DAMSMDataset(BaseT2IDataset):
    def __init__(self, data_path, words_limit, base_size=64, stage_num=3, trans_norm=None,
                 lang='en', mode="train", scale="small", use_memo=False):
        super().__init__(words_limit, base_size=base_size, stage_num=stage_num, trans_norm=trans_norm,
                         mode=mode, use_acc=True)
        self.use_memo = use_memo

        # --- Load each data ---
        self.load_data(data_path, lang, scale)
        self.class_id = np.arange(len(self.img_insts))

        # --- Make a note (may run out of memory) ---
        if use_memo:
            self.image_memory = self.load_image_memory(data_path, base_size)

        self.num_example = len(self.img_insts)

        if self.num_example * self.cap_per_img != len(self.text_insts):
            error_info = f"img_insts:{self.num_example} text_insts:{len(self.text_insts)}"
            raise ValueError(f"[Warning] Number of images or text does not match.\n{error_info}")

        print(f"[Info] Dataset is ready for {self.mode}. # data:{self.num_example}")


    def load_data(self, data_path, lang, scale):
        img_cap = Constants.D_SCALE[scale]['img_cap']

        data_path = data_path.format(mode=self.mode)
        print(f"[Info] Loading data from {data_path}")
        with open(data_path, 'rb') as f:
            x = pickle.load(f)

        self.index2word = x['index2word'][lang]
        self.word2index = x['word2index'][lang]
        
        if self.mode == "train":
            self.img_insts = x[img_cap]['img']
            self.text_insts = x[img_cap][lang]['id']
            self.imgs_dir = Constants.IMAGE_DIRS[img_cap][self.mode]
            self.cap_per_img = Constants.CAP_PER_IMG[img_cap]
        else:
            if lang == 'en':
                self.img_insts = x[img_cap]['img']
                self.text_insts = x[img_cap]['en']['id']
                self.imgs_dir = Constants.IMAGE_DIRS[img_cap][self.mode]
                self.cap_per_img = Constants.CAP_PER_IMG[img_cap]
            else:
                self.img_insts = x['multi30k']['img']
                text_words = x['multi30k'][lang]['word']
                self.text_insts = [[self.word2index.get(word, Constants.UNK) for word in words] for words in text_words]
                self.imgs_dir = Constants.IMAGE_DIRS['multi30k'][self.mode]
                self.cap_per_img = Constants.CAP_PER_IMG['multi30k']

        self.vocab_size = len(self.index2word)


    def get_image(self, index):
        if self.use_memo:
            image = self.image_memory[index]
        else:
            img_name = self.img_insts[index]
            image = self.img_name2img(img_name)
            image = self.first_resize(image)

        image = self.trans_random(image)
        image = self.trans_norm(image)

        return image

        
    def __len__(self):
        return self.num_example

    def __getitem__(self, index):
        image = self.get_image(index)
        
        error_cnt = 0
        while True:
            text_id = np.random.randint(0, self.cap_per_img)
            text_id = index * self.cap_per_img + text_id
            text = self.text_insts[text_id]
            num_words = len(text)
            if num_words > 1:
                break
            error_cnt += 1
            if error_cnt > 50:
                raise ValueError("Sentences are only length of 1")
        text, text_len = self.text_sampling(text)
        class_id = self.class_id[index]
        filename = self.img_insts[index]
        return image, text, text_len, class_id, filename
