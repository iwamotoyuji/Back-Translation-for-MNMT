import numpy as np
import random
import pickle
from itertools import takewhile
from re import compile, sub

## [Pytorch] ############################################################################
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
#########################################################################################

## [Self-Module] ########################################################################
from datasets.base_dataset import BaseT2IDataset, MyBatchSampler
import my_utils.Constants as Constants
#########################################################################################


class ModelConnector:
    def __init__(self, T2I_batch_size, T2I_words_limit, MMT_id2word=None, T2I_word2id=None, bpe=None):
        self.T2I_words_limit = T2I_words_limit
        self.MMT_id2word = MMT_id2word
        self.T2I_word2id = T2I_word2id
        self.bpe = bpe

        self.T2I_inputs_zeros = np.zeros((T2I_batch_size, T2I_words_limit), dtype='int64')
        self.T2I_inputs_len_zeros = np.zeros((T2I_batch_size), dtype='int64')
        if bpe is not None:
            self.bpe_re = compile("@@ |@@ ?$")

    def make_T2I_tgt(self, MMT_outputs):
        batch_size = len(MMT_outputs)
        T2I_inputs = self.T2I_inputs_zeros[:batch_size]
        T2I_inputs_len = self.T2I_inputs_len_zeros[:batch_size]
        for i in range(batch_size):
            text_ids = [index.item() for index in MMT_outputs[i]]
            text_ids = list(takewhile(lambda index: index != Constants.EOS, text_ids))

            if self.bpe is not None:
                text_words = [self.MMT_id2word[index] for index in text_ids]
                text = ' '.join(text_words)
                text = self.bpe_re.sub('', text)
                text_words = text.split(' ')
                text_ids = [self.T2I_word2id.get(index, Constants.UNK) for index in text_words]
            
            text_ids = np.array(text_ids)
            num_words = len(text_ids)

            if num_words <= self.T2I_words_limit:
                text_len = num_words
                T2I_inputs[i][:num_words] = text_ids
            else:
                indices = np.arange(num_words)
                np.random.shuffle(indices)
                indices = indices[:self.T2I_words_limit]
                indices = np.sort(indices)
                text_ids = text_ids[indices]
                text_len = self.T2I_words_limit
                T2I_inputs[i] = text_ids
            T2I_inputs_len[i] = text_len
        T2I_inputs = torch.LongTensor(T2I_inputs)
        T2I_inputs_len = torch.LongTensor(T2I_inputs_len)

        return T2I_inputs, T2I_inputs_len


def get_train_loader(opt, base_size=64, trans_norm=None, drop_last=True):
    def worker_init_fn(worker_id):
        random.seed(worker_id + opt.random_seed)
        np.random.seed(worker_id + opt.random_seed)

    train_dataset = T2IDataset(
        data_path=opt.data_path,
        words_limit=opt.words_limit,
        base_size=base_size,
        stage_num=opt.stage_num,
        trans_norm=trans_norm,
        src_lang=opt.src_lang,
        tgt_lang=opt.tgt_lang,
        mode='train',
        scale=opt.d_scale,
        use_memo=opt.use_memo,
        bilingual=opt.bilingual,
    )

    if opt.d_scale == "small":
        train_loader = DataLoader(
            train_dataset, batch_size=opt.batch_size,
            shuffle=True, num_workers=opt.workers,
            drop_last=drop_last, worker_init_fn=worker_init_fn,
        )
    else:
        train_batch_sampler = MyBatchSampler(
            sorted_insts=train_dataset.src_insts, batch_size=opt.batch_size,
            shuffle=True, drop_last=drop_last,
        )
        train_loader = DataLoader(
            train_dataset, batch_sampler=train_batch_sampler,
            num_workers=opt.workers, worker_init_fn=worker_init_fn,
        )

    return train_loader


def get_test_loader(opt, base_size=64, trans_norm=None, drop_last=False):
    def worker_init_fn(worker_id):
        random.seed(worker_id + opt.random_seed)
        np.random.seed(worker_id + opt.random_seed)

    eval_dataset = T2IDataset(
        data_path=opt.data_path,
        words_limit=opt.words_limit,
        base_size=base_size,
        stage_num=opt.stage_num,
        trans_norm=trans_norm,
        src_lang=opt.src_lang,
        tgt_lang=opt.tgt_lang,
        mode='valid',
        scale=opt.d_scale,
        use_memo=False,
    )

    eval_loader = DataLoader(
        eval_dataset, batch_size=opt.batch_size,
        shuffle=False, num_workers=opt.workers,
        drop_last=drop_last, worker_init_fn=worker_init_fn,
        collate_fn=eval_dataset.valid_collate_fn,
    )
    
    return eval_loader
    

class T2IDataset(BaseT2IDataset):
    def __init__(self, data_path, words_limit, base_size=64, stage_num=3, trans_norm=None,
                 src_lang='en', tgt_lang='de', mode='train', scale="small", use_memo=False,
                 bilingual=True):
        super().__init__(words_limit, base_size=base_size, stage_num=stage_num, trans_norm=trans_norm,
                         mode=mode, use_acc=False)
        self.use_memo = use_memo
        self.bilingual = bilingual

        # --- Load each data ---
        self.load_data(data_path, src_lang, tgt_lang, scale)
        self.class_id = np.arange(len(self.img_insts))

        # --- Set the process to be applied to the images ---
        second_size = base_size
        self.second_resizes = []
        for _ in range(stage_num - 1):
            self.second_resizes.append(transforms.Resize(second_size))
            second_size *= 2

        # -- Make a note (may run out of memory) --
        if use_memo:
            self.image_memory = self.load_image_memory(data_path, base_size)

        self.num_example = len(self.img_insts)

        if self.num_example * self.cap_per_img != len(self.src_insts):
            error_info = f"img_insts:{self.num_example} text_insts:{len(self.src_insts)}"
            raise ValueError(f"[Warning] Number of images or text does not match.\n{error_info}")

        print("[Info] Dataset is ready for {}. # data:{}".format(self.mode, self.num_example))


    def load_data(self, data_path, src_lang, tgt_lang, scale):
        img_cap = Constants.D_SCALE[scale]['img_cap']

        print(f"[Info] Loading data from {data_path}")
        with open(data_path.format(mode=self.mode), 'rb') as f:
            x = pickle.load(f)
        
        if self.mode == 'train':
            self.img_insts = x[img_cap]['img']
            self.src_insts = x[img_cap][src_lang]['id']
            if self.bilingual:
                self.tgt_insts = x[img_cap][tgt_lang]['id']
            self.imgs_dir = Constants.IMAGE_DIRS[img_cap][self.mode]
            self.cap_per_img = Constants.CAP_PER_IMG[img_cap]
        else:
            self.img_insts = x["mscoco"]['img']
            self.T2I_src_insts = x["mscoco"][src_lang]['id']
            #self.tgt_insts = x["mscoco"][tgt_lang]['id']
            self.MMT_src_insts = x["mscoco"][src_lang]['bpe']
            self.imgs_dir = Constants.IMAGE_DIRS["mscoco"][self.mode]
            self.cap_per_img = Constants.CAP_PER_IMG["mscoco"]
            self.MMT_index2word = x['index2word']['bpe']

        self.tgt_word2index = x['word2index'][tgt_lang]
        self.src_vocab_size = len(x['index2word'][src_lang])
        self.tgt_vocab_size = len(self.tgt_word2index)


    def get_image(self, index):
        if self.use_memo:
            image = self.image_memory[index]
        else:
            img_name = self.img_insts[index]
            image = self.img_name2img(img_name)
            image = self.first_resize(image)
        
        image = self.trans_random(image)
        images = [self.trans_norm(resize(image)) for resize in self.second_resizes]
        images.append(self.trans_norm(image))

        return images
        
        
    def __len__(self):
        return self.num_example

    def __getitem__(self, index):
        text_id = np.random.randint(0, self.cap_per_img)
        text_id = index * self.cap_per_img + text_id
        filenames = self.img_insts[index]
        
        if self.mode == "train":
            images = self.get_image(index)
            class_id = self.class_id[index]

            if self.bilingual:
                while True:
                    src_text, src_text_len = self.text_sampling(self.src_insts[text_id])
                    tgt_text, tgt_text_len = self.text_sampling(self.tgt_insts[text_id])
                    if tgt_text_len > 1:
                        break
                    text_id = np.random.randint(0, self.cap_per_img)
                    text_id = index * self.cap_per_img + text_id
                return images, src_text, src_text_len, tgt_text, tgt_text_len, class_id, filenames
            else:
                src_text, src_text_len = self.text_sampling(self.src_insts[text_id])
                return images, src_text, src_text_len, class_id, filenames
        else:
            MMT_src_text = self.MMT_src_insts[text_id]
            T2I_src_text, T2I_src_text_len = self.text_sampling(self.T2I_src_insts[text_id])
            return index, MMT_src_text, T2I_src_text, T2I_src_text_len, filenames


    def valid_collate_fn(self, insts):
        indices, MMT_src_insts, T2I_src_insts, T2I_src_insts_len, filenames = list(zip(*insts))
        images = []
        for index in indices:
            img_name = self.img_insts[index]
            image = self.img_name2img(img_name)
            image = self.trans_random(image)
            image = self.trans_norm(image)
            images.append(image)
        images = [torch.stack(images,dim=0)]

        MMT_src_insts, MMT_src_insts_pos = MT_collate_fn(MMT_src_insts)
        T2I_src_insts = torch.LongTensor(T2I_src_insts)
        T2I_src_insts_len = torch.LongTensor(T2I_src_insts_len)
        filenames = np.array(filenames)

        return images, MMT_src_insts, MMT_src_insts_pos, T2I_src_insts, T2I_src_insts_len, filenames


def MT_collate_fn(insts):
    max_seq_len = max(len(inst) for inst in insts)

    batch_seq = np.array([inst + [Constants.PAD] * (max_seq_len - len(inst)) for inst in insts])
    batch_pos = np.array([[pos+1 if w != Constants.PAD else 0 for pos, w in enumerate(inst)] for inst in batch_seq])

    batch_seq = torch.LongTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos