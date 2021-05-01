import numpy as np
import random
import pickle

## [Pytroch] ############################################################################
import torch
from torch.utils.data import DataLoader
#########################################################################################

## [Self-Module] ########################################################################
from datasets.base_dataset import BaseMNMTDataset, MyBatchSampler
import my_utils.Constants as Constants
#########################################################################################


def get_train_val_loader(opt):
    def worker_init_fn(worker_id):
        random.seed(worker_id + opt.random_seed)
        np.random.seed(worker_id + opt.random_seed)

    adapt = opt.adapt_init_MNMT or opt.adapt_prop_MNMT
    train_dataset = BT_MNMTDataset(
        data_path=opt.data_path,
        words_limit=opt.eval_words_limit,
        src_lang=opt.src_lang,
        tgt_lang=opt.tgt_lang,
        mode="train",
        scale=opt.d_scale,
        bpe=opt.bpe,
        adapt=adapt,
    )

    if opt.d_scale == "small" or adapt:
        train_loader = DataLoader(
            train_dataset, batch_size=opt.MNMT_batch_size,
            shuffle=True, num_workers=opt.workers,
            collate_fn=train_dataset.train_collate_fn, worker_init_fn=worker_init_fn,
        )
        valid_dataset = BT_MNMTDataset(
            data_path=opt.data_path,
            words_limit=opt.eval_words_limit,
            src_lang=opt.src_lang,
            tgt_lang=opt.tgt_lang,
            mode="valid",
            scale=opt.d_scale,
            bpe=opt.bpe,
            adapt=adapt,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=opt.MNMT_batch_size, 
            shuffle=False, num_workers=opt.workers,
            collate_fn=valid_dataset.val_eval_collate_fn, worker_init_fn=worker_init_fn,
        )
    else:
        train_batch_sampler = MyBatchSampler(
            sorted_insts=train_dataset.MNMT_tgt_insts, batch_size=opt.MNMT_batch_size,
            shuffle=True, drop_last=False,
        )
        train_loader = DataLoader(
            train_dataset, batch_sampler=train_batch_sampler, num_workers=opt.workers,
            collate_fn=train_dataset.train_collate_fn, worker_init_fn=worker_init_fn,
        )
        valid_loader = None

    return train_loader, valid_loader


def get_eval_loader(opt):
    adapt = opt.adapt_init_MNMT or opt.adapt_prop_MNMT
    eval_dataset = BT_MNMTDataset(
        data_path=opt.data_path,
        words_limit=None,
        src_lang=opt.src_lang,
        tgt_lang=opt.tgt_lang,
        mode="eval",
        scale=opt.d_scale,
        bpe=opt.bpe,
        adapt=adapt,
    )

    eval_loader = DataLoader(
        eval_dataset, batch_size=opt.batch_size,
        shuffle=False, num_workers=opt.workers,
        collate_fn=eval_dataset.val_eval_collate_fn
    )

    return eval_loader


class BT_MNMTDataset(BaseMNMTDataset):
    def __init__(self, data_path, words_limit, src_lang='en', tgt_lang='de',
                 mode='train', scale="small", bpe=None, adapt=False):
        super().__init__(mode=mode)
        self.words_limit = words_limit

        # -- Load each data --
        self.load_data(data_path, src_lang, tgt_lang, scale, bpe, adapt)
        self.num_example = len(self.MNMT_src_insts)

        if len(self.MNMT_src_insts) != len(self.MNMT_tgt_insts):
            error_info = f"src_insts:{len(self.MNMT_src_insts)} tgt_insts:{len(self.MNMT_tgt_insts)}"
            raise ValueError(f"[Warning] Number of texts does not match.\n{error_info}")

        print(f"[Info] Dataset is ready for {self.mode}. # data:{self.num_example}")
        
        
    def load_data(self, data_path, src_lang, tgt_lang, scale, bpe=None, adapt=False):
        mode = "adapt_" + self.mode if adapt else self.mode
        scale = "small" if adapt else scale
        if adapt:
            t2i_d_type = 't2i_id'
            MNMT_d_type = 'MNMT_id' if bpe is None else 'bpe'
            src_dict_type = 'MNMT_' + src_lang if bpe is None else 'bpe'
            tgt_dict_type = 'MNMT_' + tgt_lang if bpe is None else 'bpe'
        else:
            t2i_d_type = 'id'
            MNMT_d_type = 'id' if bpe is None else 'bpe'
            src_dict_type = src_lang if bpe is None else 'bpe'
            tgt_dict_type = tgt_lang if bpe is None else 'bpe'

        para_text = Constants.D_SCALE[scale]['para_text']    

        print(f"[Info] Loading data from {data_path}")
        with open(data_path.format(mode=mode), 'rb') as f:
            x = pickle.load(f)

        if self.mode == 'train':
            self.MNMT_src_insts = x[para_text][src_lang][MNMT_d_type]
            self.MNMT_tgt_insts = x[para_text][tgt_lang][MNMT_d_type]
            self.T2I_src_insts = x[para_text][src_lang][t2i_d_type]
            self.T2I_tgt_insts = x[para_text][tgt_lang][t2i_d_type]
            assert len(self.T2I_src_insts) == len(self.MNMT_src_insts)
        else:
            assert para_text == "multi30k"
            self.img_insts = x[para_text]['img']
            self.MNMT_src_insts = x[para_text][src_lang][MNMT_d_type]
            references = x[para_text][tgt_lang]['word']
            self.MNMT_tgt_insts = [[sent] for sent in references]
            self.imgs_dir = Constants.IMAGE_DIRS[para_text][self.mode]

            # --- input same image ---
            #np.random.seed(100)
            #num_img = len(self.img_insts)
            #self.img_insts = [self.img_insts[np.random.randint(0,num_img)]] * num_img

        self.tgt_index2word = x['index2word'][tgt_dict_type]
        self.src_vocab_size = len(x['index2word'][src_dict_type])
        self.tgt_vocab_size = len(self.tgt_index2word)
    

    def text_sampling(self, text):
        text = np.array(text).astype('int64')

        # -- Padding and keeping within words limits --
        new_text = np.zeros((self.words_limit), dtype='int64')
        num_words = len(text)

        if num_words <= self.words_limit:
            new_text[:num_words] = text
            text_len = num_words
        else:
            ids = list(np.arange(num_words))
            np.random.shuffle(ids)
            ids = ids[:self.words_limit]
            ids = np.sort(ids)
            new_text = text[ids]
            text_len = self.words_limit

        return new_text, text_len    


    def __len__(self):
        return self.num_example

    def __getitem__(self, idx):
        if self.mode == 'train':
            MNMT_src_text = self.MNMT_src_insts[idx]
            MNMT_tgt_text = self.MNMT_tgt_insts[idx] + [self.EOS_token]
            T2I_src_text, T2I_src_text_len = self.text_sampling(self.T2I_src_insts[idx])
            T2I_tgt_text, T2I_tgt_text_len = self.text_sampling(self.T2I_tgt_insts[idx])
            return T2I_src_text, T2I_src_text_len, T2I_tgt_text, T2I_tgt_text_len, MNMT_src_text, MNMT_tgt_text
        else:
            return self.img_insts[idx], self.MNMT_src_insts[idx]


    def train_collate_fn(self, insts):
        T2I_src_text, T2I_src_text_len, T2I_tgt_text, T2I_tgt_text_len, MNMT_src_text, MNMT_tgt_text = list(zip(*insts))
        T2I_src_text = torch.LongTensor(T2I_src_text)
        T2I_src_text_len = torch.LongTensor(T2I_src_text_len)
        T2I_tgt_text = torch.LongTensor(T2I_tgt_text)
        T2I_tgt_text_len = torch.LongTensor(T2I_tgt_text_len)
        MNMT_src_text, MNMT_src_text_pos = self.make_text_pos(MNMT_src_text)
        MNMT_tgt_text, MNMT_tgt_text_pos = self.make_text_pos(MNMT_tgt_text)
        return T2I_src_text, T2I_src_text_len, T2I_tgt_text, T2I_tgt_text_len, MNMT_src_text, MNMT_src_text_pos, MNMT_tgt_text, MNMT_tgt_text_pos

    def val_eval_collate_fn(self, insts):
        img_names, MNMT_src_text = list(zip(*insts))
        images = [self.img_name2img(img_name) for img_name in img_names]
        images = torch.stack(images, dim=0)
        MNMT_src_text, MNMT_src_pos = self.make_text_pos(MNMT_src_text)
        return images, MNMT_src_text, MNMT_src_pos
