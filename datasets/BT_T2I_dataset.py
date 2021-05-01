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
    def __init__(self, T2I_batch_size, T2I_words_limit, MNMT_id2word=None, T2I_word2id=None, bpe=None):
        self.T2I_words_limit = T2I_words_limit
        self.MNMT_id2word = MNMT_id2word
        self.T2I_word2id = T2I_word2id
        self.bpe = bpe

        self.T2I_inputs_zeros = np.zeros((T2I_batch_size, T2I_words_limit), dtype='int64')
        self.T2I_inputs_len_zeros = np.zeros((T2I_batch_size), dtype='int64')
        if bpe is not None:
            self.bpe_re = compile("@@ |@@ ?$")

    def make_T2I_tgt(self, MNMT_outputs):
        batch_size = len(MNMT_outputs)
        T2I_inputs = self.T2I_inputs_zeros[:batch_size]
        T2I_inputs_len = self.T2I_inputs_len_zeros[:batch_size]
        for i in range(batch_size):
            text_ids = [index.item() for index in MNMT_outputs[i]]
            text_ids = list(takewhile(lambda index: index != Constants.EOS, text_ids))

            if self.bpe is not None:
                text_words = [self.MNMT_id2word[index] for index in text_ids]
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

    adapt = opt.adapt_prop_MNMT or opt.adapt_init_MNMT
    train_dataset = T2IDataset(
        data_path=opt.data_path,
        words_limit=opt.train_words_limit,
        base_size=base_size,
        stage_num=opt.stage_num,
        trans_norm=trans_norm,
        src_lang=opt.src_lang,
        tgt_lang=opt.tgt_lang,
        mode="train",
        scale=opt.d_scale,
        use_memo=opt.use_memo,
        bpe=opt.bpe,
        adapt=adapt,
    )

    if opt.d_scale == "small" or adapt:
        train_loader = DataLoader(
            train_dataset, batch_size=opt.T2I_batch_size,
            shuffle=True, num_workers=opt.workers,
            drop_last=drop_last, worker_init_fn=worker_init_fn,
            collate_fn=train_dataset.train_collate_fn,
        )
    else:
        train_batch_sampler = MyBatchSampler(
            sorted_insts=train_dataset.MNMT_src_insts, batch_size=opt.T2I_batch_size,
            shuffle=True, drop_last=drop_last,
        )
        train_loader = DataLoader(
            train_dataset, batch_sampler=train_batch_sampler,
            num_workers=opt.workers, worker_init_fn=worker_init_fn,
            collate_fn=train_dataset.train_collate_fn,
        )

    return train_loader


def get_eval_loader(opt, base_size=64, trans_norm=None, drop_last=False):
    def worker_init_fn(worker_id):
        random.seed(worker_id + opt.random_seed)
        np.random.seed(worker_id + opt.random_seed)

    eval_dataset = T2IDataset(
        data_path=opt.data_path,
        words_limit=opt.eval_words_limit,
        base_size=base_size,
        stage_num=opt.stage_num,
        trans_norm=trans_norm,
        src_lang=opt.src_lang,
        tgt_lang=opt.tgt_lang,
        mode=opt.mode,
        scale=opt.d_scale,
        use_memo=False,
        bpe=opt.bpe,
        adapt=False,
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
                 src_lang='en', tgt_lang='de', mode="train", scale="small", use_memo=False,
                 bpe=None, adapt=False):
        super().__init__(words_limit, base_size=base_size, stage_num=stage_num, trans_norm=trans_norm,
                         mode=mode, use_acc=False)
        self.use_memo = use_memo

        # -- Load each data --
        self.load_data(data_path, src_lang, tgt_lang, scale, bpe, adapt)
        self.class_id = np.arange(len(self.img_insts))

        # -- Set the process to be applied to the images --
        second_size = base_size
        self.second_resizes = []
        for _ in range(stage_num -1):
            self.second_resizes.append(transforms.Resize(second_size))
            second_size *= 2

        # -- Make a note (may run out of memory) --
        if use_memo:
            self.image_memory = self.load_image_memory(data_path, base_size)

        self.PAD_token = Constants.PAD
        self.EOS_token = Constants.EOS
        self.num_example = len(self.img_insts)

        if self.num_example * self.cap_per_img != len(self.T2I_src_insts):
            error_info = "img_insts:{} src_insts:{}".format(self.num_example, len(self.T2I_src_insts))
            raise ValueError("[Warning] Number of images or text does not match\n{}".format(error_info))

        print("[Info] Dataset is ready for {}. # data:{}".format(self.mode, self.num_example))


    def load_data(self, data_path, src_lang, tgt_lang, scale, bpe=None, adapt=False):
        mode = "adapt_" + self.mode if adapt else self.mode
        scale = "small" if adapt else scale    
        if adapt:
            t2i_d_type = 't2i_id'
            MNMT_d_type = 'MNMT_id' if bpe is None else 'bpe'
            src_dict_type = 't2i_' + src_lang
            tgt_dict_type = 't2i_' + tgt_lang
        else:
            t2i_d_type = 'id'
            MNMT_d_type = 'id' if bpe is None else 'bpe'
            src_dict_type = src_lang
            tgt_dict_type = tgt_lang

        img_cap = Constants.D_SCALE[scale]['img_cap']

        print(f"[Info] Loading data from {data_path}")
        with open(data_path.format(mode=mode), 'rb') as f:
            x = pickle.load(f)

        if self.mode == 'train':
            self.img_insts = x[img_cap]['img']
            self.T2I_src_insts = x[img_cap][src_lang][t2i_d_type]
            self.T2I_tgt_insts = None
            self.MNMT_src_insts = x[img_cap][src_lang][MNMT_d_type]
            self.imgs_dir = Constants.IMAGE_DIRS[img_cap][self.mode]
            self.cap_per_img = Constants.CAP_PER_IMG[img_cap]
            assert len(self.T2I_src_insts) == len(self.MNMT_src_insts)
        else:
            """
            self.img_insts = x["multi30k"]['img']
            self.T2I_src_insts = x["multi30k"][src_lang][MNMT_d_type]
            self.T2I_tgt_insts = x["multi30k"][tgt_lang]['word']
            if bpe is None:
                self.bpe_index2word = None
            else:
                self.bpe_index2word = x['index2word']['bpe']
                self.bpe_re = compile("@@ |@@ ?$")
                self.src_word2index = x['word2index'][src_lang]
            self.cap_per_img = Constants.CAP_PER_IMG["multi30k"]
            """
            self.img_insts = x["mscoco"]['img']
            self.T2I_src_insts = x["mscoco"][src_lang]['id']
            self.MNMT_src_insts = x["mscoco"][src_lang]['bpe']
            self.imgs_dir = Constants.IMAGE_DIRS["mscoco"][self.mode]
            self.cap_per_img = Constants.CAP_PER_IMG["mscoco"]
            self.MNMT_index2word = x['index2word']['bpe']
            
        self.tgt_word2index = x['word2index'][tgt_dict_type]
        self.src_vocab_size = len(x['index2word'][src_dict_type])
        self.tgt_vocab_size = len(self.tgt_word2index)


    def get_T2I_text(self, text):
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


    def __getitem__(self, index):
        text_id = np.random.randint(0, self.cap_per_img)
        text_id = index * self.cap_per_img + text_id
        if self.mode == "train":
            MNMT_src_text = self.MNMT_src_insts[text_id]
            T2I_src_text, T2I_src_text_len = self.text_sampling(self.T2I_src_insts[text_id])
            class_id = self.class_id[index]
            return index, MNMT_src_text, T2I_src_text, T2I_src_text_len, class_id
        else:
            """
            T2I_src_text = self.T2I_src_insts[text_id]
            if self.bpe_index2word is not None:
                T2I_src_text = [self.bpe_index2word[id] for id in T2I_src_text]
                T2I_src_text = ' '.join(T2I_src_text)
                T2I_src_text = self.bpe_re.sub('', T2I_src_text)
                T2I_src_text = T2I_src_text.split(' ')
                T2I_src_text = [self.src_word2index.get(word, Constants.UNK) for word in T2I_src_text]
            T2I_tgt_text = self.T2I_tgt_insts[text_id]
            T2I_tgt_text = [self.tgt_word2index.get(word, Constants.UNK) for word in T2I_tgt_text]
            T2I_src_text, T2I_src_text_len = self.get_T2I_text(T2I_src_text)
            T2I_tgt_text, T2I_tgt_text_len = self.get_T2I_text(T2I_tgt_text)
            filename = self.img_insts[index]
            return T2I_src_text, T2I_src_text_len, T2I_tgt_text, T2I_tgt_text_len, filename
            """
            filenames = self.img_insts[index]
            MNMT_src_text = self.MNMT_src_insts[text_id]
            T2I_src_text, T2I_src_text_len = self.get_T2I_text(self.T2I_src_insts[text_id])
            return index, MNMT_src_text, T2I_src_text, T2I_src_text_len, filenames


    def make_text_pos(self, text_insts):
        max_text_len = max(len(text) for text in text_insts)
        batch_text = np.array([text + [self.PAD_token] * (max_text_len - len(text)) for text in text_insts])
        batch_pos = np.array([[pos+1 if w != self.PAD_token else 0 for pos, w in enumerate(text)] for text in batch_text])
        batch_text = torch.LongTensor(batch_text)
        batch_pos = torch.LongTensor(batch_pos)
        return batch_text, batch_pos


    def train_collate_fn(self, insts):
        indices, MNMT_src_insts, T2I_src_insts, T2I_src_insts_len, class_ids = list(zip(*insts))
        all_images = [[] for i in range(self.stage_num)]
        for index in indices:
            if self.use_memo:
                image = self.image_memory[index]
            else:
                img_name = self.img_insts[index]
                image = self.img_name2img(img_name)
                image = self.first_resize(image)
            image = self.trans_random(image)
            images = [self.trans_norm(resize(image)) for resize in self.second_resizes]
            images.append(self.trans_norm(image))

            for i in range(self.stage_num):
                all_images[i].append(images[i])

        all_images = [torch.stack(images) for images in all_images]
        
        MNMT_src_insts, MNMT_src_insts_pos = self.make_text_pos(MNMT_src_insts)
        T2I_src_insts = torch.LongTensor(T2I_src_insts)
        T2I_src_insts_len = torch.LongTensor(T2I_src_insts_len)
        class_ids = torch.LongTensor(class_ids)
        
        return all_images, MNMT_src_insts, MNMT_src_insts_pos, T2I_src_insts, T2I_src_insts_len, class_ids


    def valid_collate_fn(self, insts):
        indices, MNMT_src_insts, T2I_src_insts, T2I_src_insts_len, filenames = list(zip(*insts))
        images = []
        for index in indices:
            img_name = self.img_insts[index]
            image = self.img_name2image(img_name)
            image = self.trans_random(image)
            image = self.trans_norm(image)
            images.append(image)
        images = [torch.stack(images,dim=0)]

        MNMT_src_insts, MNMT_src_insts_pos = self.make_text_pos(MNMT_src_insts)
        T2I_src_insts = torch.LongTensor(T2I_src_insts)
        T2I_src_insts_len = torch.LongTensor(T2I_src_insts_len)
        filenames = np.array(filenames)

        return images, MNMT_src_insts, MNMT_src_insts_pos, T2I_src_insts, T2I_src_insts_len, filenames
