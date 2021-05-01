
## [Pytorch] ############################################################################
import torch
import torch.nn as nn
#########################################################################################

## [Self-Module] ########################################################################
from my_utils.pytorch_utils import weights_init
#########################################################################################


def conv4x4(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=False)

def Conv3x3Block(in_channel, out_channel):
    block = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(0.2, inplace=True))
    return block

def DownBlock(in_channel, out_channel):
    block = nn.Sequential(
        conv4x4(in_channel, out_channel),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(0.2, inplace=True))
    return block

def FirstDownBlocks(D_hidden):
    block = nn.Sequential(
        conv4x4(3, D_hidden),
        nn.LeakyReLU(0.2, inplace=True),

        conv4x4(D_hidden, D_hidden * 2),
        nn.BatchNorm2d(D_hidden * 2),
        nn.LeakyReLU(0.2, inplace=True),

        conv4x4(D_hidden * 2, D_hidden * 4),
        nn.BatchNorm2d(D_hidden * 4),
        nn.LeakyReLU(0.2, inplace=True),

        conv4x4(D_hidden * 4, D_hidden * 8),
        nn.BatchNorm2d(D_hidden * 8),
        nn.LeakyReLU(0.2, inplace=True))
    return block

class GetLogits(nn.Module):
    def __init__(self, embedding_dim, D_hidden, is_conditional=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.is_conditional = is_conditional
        if self.is_conditional:
            self.cond_conv = Conv3x3Block(D_hidden * 8 + embedding_dim, D_hidden * 8)
        
        self.make_logits = nn.Conv2d(D_hidden * 8, 1, kernel_size=4, stride=4, bias=True)

    def forward(self, img4x4_code, embed_sent=None):
        if not self.is_conditional or embed_sent is None:
            logits_code = img4x4_code
        else:
            sent_code = embed_sent.view(-1, self.embedding_dim, 1, 1)
            sent_code = sent_code.repeat(1, 1, 4, 4)
            logits_code = torch.cat((img4x4_code, sent_code), dim=1)
            logits_code = self.cond_conv(logits_code)
        
        logits = self.make_logits(logits_code)
        return logits.view(-1)


class D_net64(nn.Module):
    def __init__(self, embedding_dim, D_hidden):
        super().__init__()
        self.first_down = FirstDownBlocks(D_hidden)
        
        self.get_uncond_logits = GetLogits(embedding_dim, D_hidden, is_conditional=False)
        self.get_cond_logits = GetLogits(embedding_dim, D_hidden, is_conditional=True)

        self.apply(weights_init)

    def forward(self, img64, conditions, return_feat=False):
        img4x4_code = self.first_down(img64)
        uncond_logits = self.get_uncond_logits(img4x4_code)
        cond_logits = self.get_cond_logits(img4x4_code, conditions)
        
        if return_feat:
            return img4x4_code, uncond_logits, cond_logits
        return uncond_logits, cond_logits


class D_net128(nn.Module):
    def __init__(self, embedding_dim, D_hidden):
        super().__init__()
        self.first_down = FirstDownBlocks(D_hidden)
        self.second_down = DownBlock(D_hidden * 8, D_hidden * 16)
        self.down_hidden = Conv3x3Block(D_hidden * 16, D_hidden * 8)

        self.get_uncond_logits = GetLogits(embedding_dim, D_hidden, is_conditional=False)
        self.get_cond_logits = GetLogits(embedding_dim, D_hidden, is_conditional=True)

        self.apply(weights_init)

    def forward(self, img128, conditions, return_feat=False):
        img8x8_code = self.first_down(img128)
        img4x4_code = self.second_down(img8x8_code)
        img4x4_code = self.down_hidden(img4x4_code)

        uncond_logits = self.get_uncond_logits(img4x4_code)
        cond_logits = self.get_cond_logits(img4x4_code, conditions)
        
        if return_feat:
            return img4x4_code, uncond_logits, cond_logits
        return uncond_logits, cond_logits


class D_net256(nn.Module):
    def __init__(self, embedding_dim, D_hidden):
        super().__init__()
        self.first_down = FirstDownBlocks(D_hidden)
        self.second_down = DownBlock(D_hidden * 8, D_hidden * 16)
        self.third_down = DownBlock(D_hidden * 16, D_hidden * 32)
        self.down_hidden = Conv3x3Block(D_hidden * 32, D_hidden * 16)
        self.down_hidden_2 = Conv3x3Block(D_hidden * 16, D_hidden * 8)

        self.get_uncond_logits = GetLogits(embedding_dim, D_hidden, is_conditional=False)
        self.get_cond_logits = GetLogits(embedding_dim, D_hidden, is_conditional=True)

        self.apply(weights_init)

    def forward(self, img256, conditions, return_feat=False):
        img16x16_code = self.first_down(img256)
        img8x8_code = self.second_down(img16x16_code)
        img4x4_code = self.third_down(img8x8_code)
        img4x4_code = self.down_hidden(img4x4_code)
        img4x4_code = self.down_hidden_2(img4x4_code)
        
        uncond_logits = self.get_uncond_logits(img4x4_code)
        cond_logits = self.get_cond_logits(img4x4_code, conditions)
        
        if return_feat:
            return img4x4_code, uncond_logits, cond_logits
        return uncond_logits, cond_logits