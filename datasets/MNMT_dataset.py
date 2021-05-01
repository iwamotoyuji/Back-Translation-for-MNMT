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

    adapt = opt.adapt_init_MNMT is not None or opt.adapt_prop_MNMT is not None
    train_dataset = MNMTDataset(
        data_path=opt.data_path,
        src_lang=opt.src_lang,
        tgt_lang=opt.tgt_lang,
        mode="train",
        scale=opt.d_scale,
        bpe=opt.bpe,
        pretrain=opt.pretrain,
        adapt=adapt,
    )

    if opt.d_scale == "small" or adapt:
        train_loader = DataLoader(
            train_dataset, batch_size=opt.batch_size, 
            shuffle=True, num_workers=opt.workers, 
            collate_fn=train_dataset.train_collate_fn,
            worker_init_fn=worker_init_fn,
        )
        valid_dataset = MNMTDataset(
            data_path=opt.data_path,
            src_lang=opt.src_lang,
            tgt_lang=opt.tgt_lang,
            mode="valid",
            scale=opt.d_scale,
            bpe=opt.bpe,
            adapt=adapt,
        )  
        valid_loader = DataLoader(
            valid_dataset, batch_size=opt.batch_size, 
            shuffle=False, num_workers=opt.workers,
            collate_fn=valid_dataset.val_eval_collate_fn,
        )
    else:
        train_batch_sampler = MyBatchSampler(
            sorted_insts=train_dataset.tgt_insts,
            batch_size=opt.batch_size, token_size=opt.token_size, 
            shuffle=True, drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler, num_workers=opt.workers,
            collate_fn=train_dataset.train_collate_fn, worker_init_fn=worker_init_fn,
        )
        valid_loader = None

    return train_loader, valid_loader


def get_eval_loader(opt):
    adapt = opt.adapt_init_MNMT is not None or opt.adapt_prop_MNMT is not None
    eval_dataset = MNMTDataset(
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
        collate_fn=eval_dataset.val_eval_collate_fn
    )

    return eval_loader


class MNMTDataset(BaseMNMTDataset):
    def __init__(self, data_path, src_lang='en', tgt_lang='de',
                 mode="train", scale="small", bpe=None, pretrain=False, adapt=False):
        super().__init__(mode=mode)

        self.load_data(data_path, src_lang, tgt_lang, scale, bpe, pretrain, adapt)        
        self.num_example = len(self.img_names)

        if (self.num_example * self.cap_per_img) != len(self.src_insts) or len(self.src_insts) != len(self.tgt_insts):
            error_info = f"img_names:{self.num_example} src_insts:{len(self.src_insts)} tgt_insts:{len(self.tgt_insts)}"
            raise ValueError(f"[Warning] Number of images or text does not match\n{error_info}.")

        print(f"[Info] Dataset is ready for {self.mode}. # data:{self.num_example}")


    def load_data(self, data_path, src_lang, tgt_lang, scale, bpe=None, pretrain=False, adapt=False):
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

        print(f"[Info] Loading data from {data_path}")
        with open(data_path.format(mode=mode), 'rb') as f:
            x = pickle.load(f)

        if pretrain:
            self.img_names = x[img_cap]['img']
            self.src_insts = x[img_cap][src_lang][d_type]
            self.tgt_insts = x[img_cap][tgt_lang][d_type]
            self.imgs_dir = Constants.IMAGE_DIRS[img_cap][self.mode]
            self.cap_per_img = Constants.CAP_PER_IMG[img_cap]            
        elif self.mode == "train":
            assert para_text == "multi30k"
            self.img_names = x[para_text]['img']
            self.src_insts = x[para_text][src_lang][d_type]
            self.tgt_insts = x[para_text][tgt_lang][d_type]
            self.imgs_dir = Constants.IMAGE_DIRS[para_text][self.mode]
            self.cap_per_img = Constants.CAP_PER_IMG[para_text]        
        else:
            assert para_text == "multi30k"
            self.img_names = x[para_text]['img']
            self.src_insts = x[para_text][src_lang][d_type]
            references = x[para_text][tgt_lang]['word']
            self.tgt_insts = [[sent] for sent in references]
            self.imgs_dir = Constants.IMAGE_DIRS[para_text][self.mode]
            self.cap_per_img = Constants.CAP_PER_IMG[para_text]

            # --- input same image ---
            #np.random.seed(100)
            #num_img = len(self.img_names)
            #self.img_names = [self.img_names[np.random.randint(0,num_img)]] * num_img

        self.tgt_index2word = x['index2word'][tgt_dict_type]
        self.src_vocab_size = len(x['index2word'][src_dict_type])
        self.tgt_vocab_size = len(self.tgt_index2word)
        
    def __len__(self):
        return self.num_example

    def __getitem__(self, idx):
        text_id = np.random.randint(0, self.cap_per_img)
        text_id = idx * self.cap_per_img + text_id
        img_name = self.img_names[idx]
        src_text = self.src_insts[text_id]
        if self.mode == "train":
            tgt_text = self.tgt_insts[text_id] + [self.EOS_token]
            return img_name, src_text, tgt_text
        return img_name, src_text

    def train_collate_fn(self, insts):
        img_names, src_insts, tgt_insts = list(zip(*insts))
        img_insts = [self.img_name2img(img_name) for img_name in img_names]
        img_insts = torch.stack(img_insts, dim=0)
        src_insts = self.make_text_pos(src_insts)
        tgt_insts = self.make_text_pos(tgt_insts)
        return (img_insts, *src_insts, *tgt_insts)

    def val_eval_collate_fn(self, insts):
        img_names, src_insts = list(zip(*insts))
        img_insts = [self.img_name2img(img_name) for img_name in img_names]
        img_insts = torch.stack(img_insts, dim=0)
        src_insts = self.make_text_pos(src_insts)
        return (img_insts, *src_insts)