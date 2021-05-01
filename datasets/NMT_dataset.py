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

    adapt = opt.adapt_NMT is not None
    train_dataset = NMTDataset(
        data_path=opt.data_path,
        src_lang=opt.src_lang,
        tgt_lang=opt.tgt_lang,
        mode="train",
        scale=opt.d_scale,
        bpe=opt.bpe,
        adapt=adapt,
    )
    valid_dataset = NMTDataset(
        data_path=opt.data_path,
        src_lang=opt.src_lang,
        tgt_lang=opt.tgt_lang,
        mode="valid",
        scale=opt.d_scale,
        bpe=opt.bpe,
        adapt=adapt,
    )

    if opt.d_scale == "small" or adapt:
        train_loader = DataLoader(
            train_dataset, batch_size=opt.batch_size, 
            shuffle=True, num_workers=opt.workers, 
            collate_fn=train_dataset.train_collate_fn,
            worker_init_fn=worker_init_fn,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=opt.batch_size, 
            shuffle=False, num_workers=opt.workers, 
            collate_fn=valid_dataset.make_text_pos,
        )
    else:     
        train_batch_sampler = MyBatchSampler(
            sorted_insts=train_dataset.tgt_insts,
            batch_size=opt.batch_size, token_size=opt.token_size,
            shuffle=True, drop_last=False,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler, num_workers=opt.workers,
            collate_fn=train_dataset.train_collate_fn, worker_init_fn=worker_init_fn,
        )
        valid_batch_sampler = MyBatchSampler(
            sorted_insts=valid_dataset.tgt_insts,
            batch_size=opt.batch_size, token_size=opt.token_size,
            shuffle=False, drop_last=False,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_sampler=valid_batch_sampler, num_workers=opt.workers,
            collate_fn=valid_dataset.make_text_pos,
        )
    
    return train_loader, valid_loader


def get_eval_loader(opt):
    adapt = opt.adapt_NMT is not None
    eval_dataset = NMTDataset(
        data_path=opt.data_path,
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
        collate_fn=eval_dataset.make_text_pos,
    )

    return eval_loader


def get_trans_loader(opt):
    trans_dataset = NMTDataset(
        data_path=opt.data_path,
        src_lang=opt.src_lang,
        tgt_lang=opt.tgt_lang,
        mode=opt.tgt_mode,
        scale=opt.d_scale,
        bpe=opt.bpe,
        adapt=False,
        translate=True,
    )
    trans_loader = DataLoader(
        trans_dataset, batch_size=opt.batch_size,
        shuffle=False, num_workers=opt.workers,
        collate_fn=trans_dataset.make_text_pos
    )
        
    return trans_loader


class NMTDataset(BaseMNMTDataset):
    def __init__(self, data_path, src_lang='en', tgt_lang='de',
                 mode="train", scale="small", bpe=None, adapt=False, translate=False):
        super().__init__(mode=mode)

        self.load_data(data_path, src_lang, tgt_lang, scale, bpe, adapt, translate)
        self.num_example = len(self.src_insts)

        if self.tgt_insts is not None and len(self.src_insts) != len(self.tgt_insts):
            error_info = f"src_insts:{len(self.src_insts)} tgt_insts:{len(self.tgt_insts)}"
            raise ValueError(f"[Warning] Number of texts does not match.\n{error_info}")

        print(f"[Info] Dataset is ready for {self.mode}. # data:{self.num_example}")


    def load_data(self, data_path, src_lang, tgt_lang, scale, bpe=None, adapt=False, translate=False):
        mode = "adapt_" + self.mode if adapt else self.mode
        scale = "small" if adapt else scale
        if adapt:
            d_type = 'mmt_id' if bpe is None else 'bpe'
            src_dict_type = 'mmt_' + src_lang if bpe is None else 'bpe'
            tgt_dict_type = 'mmt_' + tgt_lang if bpe is None else 'bpe'
        else:
            d_type = 'id' if bpe is None else 'bpe'
            src_dict_type = src_lang if bpe is None else 'bpe'
            tgt_dict_type = tgt_lang if bpe is None else 'bpe'

        para_text = Constants.D_SCALE[scale]['para_text']
        img_cap = Constants.D_SCALE[scale]['img_cap']

        data_path = data_path.format(mode=mode)
        print(f"[Info] Loading data from {data_path}")
        with open(data_path, 'rb') as f:
            x = pickle.load(f)

        if translate:
            self.mode = "translate"
            self.src_insts = x[img_cap][src_lang][d_type]
            self.tgt_insts = None
            if bpe is not None:
                self.no_bpe_word2id = x['word2index'][tgt_lang]
        elif self.mode == "train":
            self.src_insts = x[para_text][src_lang][d_type]
            self.tgt_insts = x[para_text][tgt_lang][d_type]
        else:
            self.src_insts = x[para_text][src_lang][d_type]
            references = x[para_text][tgt_lang]['word']
            self.tgt_insts = [[sent] for sent in references]    

        self.tgt_index2word = x['index2word'][tgt_dict_type]
        self.src_vocab_size = len(x['index2word'][src_dict_type])
        self.tgt_vocab_size = len(self.tgt_index2word)
        
    def __len__(self):
        return self.num_example

    def __getitem__(self, idx):
        src_text = self.src_insts[idx]
        if self.mode == "train":
            tgt_text = self.tgt_insts[idx] + [self.EOS_token]
            return src_text, tgt_text
        return src_text

    def train_collate_fn(self, insts):
        src_insts, tgt_insts = list(zip(*insts))
        src_insts = self.make_text_pos(src_insts)
        tgt_insts = self.make_text_pos(tgt_insts)
        return (*src_insts, *tgt_insts)
