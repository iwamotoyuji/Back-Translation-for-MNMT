import numpy as np
from random import shuffle
from tqdm import tqdm
from accimage import Image
from PIL import Image as PIL_Image

## [Pytorch] ############################################################################
import torch
from torchvision import transforms, set_image_backend
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
#########################################################################################

## [Self-Module] ########################################################################
import my_utils.Constants as Constants
#########################################################################################


class BaseMNMTDataset(Dataset):
    def __init__(self, mode="train", transform=None):
        self.mode = mode
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.CenterCrop((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
            ])
        else:
            self.transform = transform

        self.PAD_token = Constants.PAD
        self.EOS_token = Constants.EOS


    def load_data(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

    def img_name2img(self, img_name):
        image = Image(self.imgs_dir + img_name)
        image = self.transform(image)
        return image

    def make_text_pos(self, text_insts):
        max_text_len = max(len(text) for text in text_insts)
        batch_text = np.array([text + [self.PAD_token] * (max_text_len - len(text)) for text in text_insts])
        batch_pos = np.array([[pos+1 if w != self.PAD_token else 0 for pos, w in enumerate(text)] for text in batch_text])
        batch_text = torch.LongTensor(batch_text)
        batch_pos = torch.LongTensor(batch_pos)
        return batch_text, batch_pos


class BaseT2IDataset(Dataset):
    def __init__(self, words_limit, base_size=64, stage_num=3, trans_norm=None, mode="train", use_acc=False):
        self.words_limit = words_limit
        self.stage_num = stage_num
        self.mode = mode
        self.use_acc = use_acc
        self.img_insts = None

        if use_acc:
            set_image_backend('accimage')
        img_size = base_size * (2 * (stage_num - 1))
        first_size = int(img_size * 76 / 64)
        self.first_resize = transforms.Resize(first_size)
        self.trans_random = transforms.Compose([
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
        ])
        if trans_norm is None:
            trans_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.trans_norm = transforms.Compose([
            transforms.ToTensor(),
            trans_norm,
        ])


    def load_data(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

    def img_name2img(self, img_name):
        img_path = self.imgs_dir + img_name
        if self.use_acc:
            image = Image(img_path)
        else:
            image = PIL_Image.open(img_path).convert('RGB')
        return image

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

    def load_image_memory(self, data_path, base_size):
        img_size = base_size * (2 * (self.stage_num - 1))
        memory_path = Path(data_path).parent / "memory" / f"{self.mode}_resized_images{img_size}.pickle"
        if memory_path.exists():
            print(f"Loading memory for {self.mode}...")
            with open(memory_path, 'rb') as f:
                image_memory = pickle.load(f)
        else:
            if self.img_insts is None:
                raise ValueError("Please load the img_name data first")
            print(f"Creating memory for {self.mode}...")
            self.use_acc = False
            image_memory = []
            pbar = tqdm(self.img_insts, ascii=True, mininterval=0.5, ncols=90)
            for img_name in pbar:
                image = self.img_name2img(img_name)
                image = self.first_resize(image)
                image_memory.append(image)
            with open(memory_path, 'wb') as f:
                pickle.dump(image_memory, f)
                print(f"Dumped note to {memory_path}")

        return image_memory


class MyBatchSampler(BatchSampler):
    def __init__(self, sorted_insts, batch_size=None, token_size=None, shuffle=True, drop_last=False):
        if (batch_size is None and token_size is None) or \
           (batch_size is not None and token_size is not None):
            raise ValueError("Please specify either batch_size or token_size")

        self.batch_size = batch_size
        self.token_size = token_size
        self.shuffle = shuffle
        self.drop_last = drop_last
            
        self.batches = []
        if self.token_size is None:
            self.make_batches_by_batch_size(sorted_insts)
        else:
            self.make_batches_by_tokens_size(sorted_insts)
        if len(self.batches[-1]) == self.batch_size:
            self.drop_last = False

        self.num_batch = len(self.batches)
        if self.drop_last:
            self.num_batch -= 1
        self.batch_ids = [i for i in range(self.num_batch)]
        

    def make_batches_by_batch_size(self, sorted_insts):
        self.batches = []
        num_data = len(sorted_insts)
        data_id = 0
        data_ids = [i for i in range(num_data)]
        while data_id < num_data:
            self.batches.append(data_ids[data_id:data_id + self.batch_size])
            data_id += self.batch_size

    def make_batches_by_tokens_size(self, sorted_insts):
        self.batches = []
        num_data = len(sorted_insts)
        data_id = 0
        while data_id < num_data:
            batch = []
            tokens = len(sorted_insts[data_id])
            sum_tokens = tokens
            while sum_tokens < self.token_size:
                batch.append(data_id)
                data_id += 1
                if data_id == num_data:
                    break
                sum_tokens += tokens
            self.batches.append(batch)

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        if self.shuffle:
            shuffle(self.batch_ids)
        for batch_id in self.batch_ids:
            yield self.batches[batch_id]