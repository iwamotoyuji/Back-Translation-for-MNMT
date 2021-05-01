
## [Pytorch] ############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#########################################################################################

## [Self-Module] ########################################################################
import my_utils.Constants as Constants
#########################################################################################


class DAMSMImageEncoder(nn.Module):
    def __init__(self, out_feat_size=256):
        super().__init__()
        #self.interpolate = F.interpolate

        inception_v3 = models.inception_v3(init_weights=False)

        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        inception_v3.load_state_dict(model_zoo.load_url(url))
        for param in inception_v3.parameters():
            param.requires_grad = False
        
        self.Conv2d_1a_3x3 = inception_v3.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception_v3.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception_v3.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception_v3.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception_v3.Conv2d_4a_3x3
        self.Mixed_5b = inception_v3.Mixed_5b
        self.Mixed_5c = inception_v3.Mixed_5c
        self.Mixed_5d = inception_v3.Mixed_5d
        self.Mixed_6a = inception_v3.Mixed_6a
        self.Mixed_6b = inception_v3.Mixed_6b
        self.Mixed_6c = inception_v3.Mixed_6c
        self.Mixed_6d = inception_v3.Mixed_6d
        self.Mixed_6e = inception_v3.Mixed_6e
        self.Mixed_7a = inception_v3.Mixed_7a
        self.Mixed_7b = inception_v3.Mixed_7b
        self.Mixed_7c = inception_v3.Mixed_7c

        self.emb_features = nn.Conv2d(768, out_feat_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.emb_cnn_code = nn.Linear(2048, out_feat_size)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.emb_features.weight.data.uniform_(-init_range, init_range)
        self.emb_cnn_code.weight.data.uniform_(-init_range, init_range)
        
    def forward(self, images):
        x = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=True)
        #x = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True)(images)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # (batch, 768, 17, 17) >> (batch, out_feat_size, 17, 17)
        features = self.emb_features(features)
        # (batch, 2048) >> (batch, out_feat_size)
        cnn_code = self.emb_cnn_code(x)        

        return features, cnn_code


class DAMSMTextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, out_feat_size=256, drop_rate=0.5, bidirectional=True):
        super().__init__()
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.hidden_size = out_feat_size // self.num_directions
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=Constants.PAD)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_size, batch_first=True, dropout=drop_rate, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=drop_rate)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)

    def forward(self, captions, cap_lens, no_reshape=False):
        # (batch, seq_len) >> (batch, seq_len, embedding_dim)
        embeddings = self.dropout(self.embedding(captions))
        embeddings = pack_padded_sequence(embeddings, cap_lens, batch_first=True, enforce_sorted=False)
        #embeddings = pack_padded_sequence(embeddings, cap_lens, batch_first=True)

        # (batch, seq_len, embedding_dim) >> (batch, seq_len, out_feat_size), (num_directions, batch, hidden_size)
        word_features, sent_features = self.lstm(embeddings)
        word_features = pad_packed_sequence(word_features, batch_first=True)[0]

        if not no_reshape:
            # (batch, seq_len, out_feat_size) >> (batch, out_feat_size, seq_len)
            word_features = word_features.transpose(1, 2)

            # (num_directions, batch, hidden_size) >> (batch, num_directions, hidden_size)
            sent_features = sent_features[0].transpose(0, 1).contiguous()
            # (batch, num_directions, hidden_size) >> (batch, out_feat_size)
            sent_features = sent_features.view(-1, self.hidden_size * self.num_directions)

        return word_features, sent_features