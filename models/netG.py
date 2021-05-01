
## [Pytorch] ############################################################################
import torch
import torch.nn as nn
#########################################################################################

## [Self-Module] ########################################################################
from my_utils.pytorch_utils import weights_init
#########################################################################################


def conv1x1(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)

def conv3x3(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)


class GLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.sigmoid

    def forward(self, x):
        channel = x.size(1)
        channel = int(channel/2)
        return x[:, :channel] * self.sigmoid(x[:, channel:])


def UpBlock(in_channel, out_channel):
    up_block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_channel, out_channel * 2),
        nn.BatchNorm2d(out_channel * 2),
        GLU(),
    )
    return up_block


class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(channel, channel * 2),
            nn.BatchNorm2d(channel * 2),
            GLU(),
            conv3x3(channel, channel),
            nn.BatchNorm2d(channel),
        )

    def forward(self, x):
        return self.block(x) + x


# ---------------- Generator ---------------- #
class ConditioningAugmentation(nn.Module):
    def __init__(self, embedding_dim, condition_dim, bilingual):
        super().__init__()
        self.cond_dim = condition_dim
        in_dim = embedding_dim * 2 if bilingual else embedding_dim
        self.fc = nn.Linear(in_dim, self.cond_dim * 2 * 2, bias=True)
        self.glu = GLU()
    
    def encode(self, embedding):
        x = self.glu(self.fc(embedding))
        mean = x[:, :self.cond_dim]
        log_var = x[:, self.cond_dim:]
        return mean, log_var

    def sampling(self, mean, log_var):
        std = log_var.mul(0.5).exp()
        eps = torch.zeros_like(std).normal_()
        return eps.mul(std).add(mean)

    def forward(self, embedding):
        mean, log_var = self.encode(embedding)
        c_code = self.sampling(mean, log_var)
        return c_code, mean, log_var


class G_First_Stage(nn.Module):
    def __init__(self, condition_dim, noise_dim, G_hidden):
        super().__init__()
        self.first_channel = G_hidden * 16
        self.fc = nn.Sequential(
            nn.Linear(condition_dim + noise_dim, self.first_channel * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(self.first_channel * 4 * 4 * 2),
            GLU(),
        )
        
        self.make8 = UpBlock(self.first_channel, G_hidden * 8)
        self.make16 = UpBlock(G_hidden * 8, G_hidden * 4)
        self.make32 = UpBlock(G_hidden * 4, G_hidden * 2)
        self.make64 = UpBlock(G_hidden * 2, G_hidden)

    def forward(self, c_code, noise):
        """
        input:
            c_code(batch, condition_dim)
            noise(batch, noise_dim)
        output:
            img64_code(batch, G_hidden, 64, 64)
        """
        c_z_code = torch.cat((c_code, noise), dim=1)

        img4_code = self.fc(c_z_code)
        img4_code = img4_code.view(-1, self.first_channel, 4, 4)

        img8_code = self.make8(img4_code)
        img16_code = self.make16(img8_code)
        img32_code = self.make32(img16_code)
        img64_code = self.make64(img32_code)

        return img64_code


class MakeAttention(nn.Module):
    def __init__(self, embedding_dim, condition_dim, G_hidden):
        super().__init__()
        self.words_conv = conv1x1(embedding_dim, G_hidden)
        self.sent_linear = nn.Linear(condition_dim, G_hidden, bias=True)
        self.sent_conv = conv1x1(G_hidden, G_hidden)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, img_query, words_source, mask=None):
        """
        input:
            img_query(batch, G_hidden, height, width)
            words_source(batch, embedding_dim, words_len)
            sent_source(batch, condition_dim)
        """
        batch_size, G_hidden, height, width = img_query.size()
        img_len = height * width

        # -- word-level attnetion --
        # (batch, G_hidden, height, width) >> (batch, img_len, G_hidden)
        query = img_query.view(batch_size, G_hidden, img_len)
        queryT = query.transpose(1, 2).contiguous()

        # (batch, embedding_dim, words_len) >> (batch, G_hidden, words_len)
        words_sourceT = words_source.unsqueeze(3)
        words_sourceT = self.words_conv(words_sourceT).squeeze(3)

        # (batch, img_len, G_hidden)*(batch, G_hidden, words_len) >> (batch, img_len, words_len)
        words_attn = torch.bmm(queryT, words_sourceT)

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.repeat(1, img_len, 1)
            words_attn.data.masked_fill_(mask.data, -float('inf'))

        words_attn = self.softmax(words_attn)
        # (batch, img_len, words_len) >> (batch, words_len, img_len)
        words_attn = words_attn.transpose(1, 2).contiguous()
        
        # (batch, G_hidden, words_len)*(batch, words_len, img_len) >> (batch, G_hidden, img_len)
        words_value = torch.bmm(words_sourceT, words_attn)
        words_value = words_value.view(batch_size, G_hidden, height, width)
        words_attn = words_attn.view(batch_size, -1, height, width)

        """
        # -- sentence-level attention --
        # (batch, condition_dim) >> (batch, G_hidden)
        sent_source = self.sent_linear(sent_source)
        # (batch, G_hidden) >> (batch, G_hidden, height, width)
        sent_source = sent_source.view(batch_size, G_hidden, 1, 1)
        sent_source = sent_source.repeat(1, 1, height, width)
        sent_attn = torch.mul(img_query, sent_source)
        sent_attn = self.sent_conv(sent_attn)
        sent_value = torch.mul(sent_source, sent_attn)
        """

        return words_value, words_attn


class G_AfterFirst_Stage(nn.Module):
    def __init__(self, embedding_dim, condition_dim, G_hidden,
                 num_resblocks=2, bilingual=True):
        super().__init__()
        self.bilingual = bilingual
        self.src_make_attn = MakeAttention(embedding_dim, condition_dim, G_hidden)
        self.res_layer = self.make_res_layer(G_hidden * 2, num_resblocks)
        self.up_layer = UpBlock(G_hidden * 2, G_hidden)

        if bilingual:
            self.tgt_make_attn = MakeAttention(embedding_dim, condition_dim, G_hidden)
            self.conv = conv1x1(G_hidden * 3, G_hidden * 2)

    def make_res_layer(self, channel, num_resblocks):
        layers = []
        for _ in range(num_resblocks):
            layers.append(ResBlock(channel))
        return nn.Sequential(*layers)

    def forward(self, img_code, src_word_embs, src_mask=None, tgt_word_embs=None, tgt_mask=None):
        """
        input:
            img_code(batch, G_hidden, height, width)
            embed_words(batch, embedding_dim, seq_len)
        output:
            out_img_code(batch, G_hidden, new_height, new_width)
        """

        src_words_value, src_words_attn = self.src_make_attn(img_code, src_word_embs, src_mask)
        if self.bilingual:
            tgt_words_value, _ = self.tgt_make_attn(img_code, tgt_word_embs, tgt_mask)
            img_word_code = torch.cat((img_code, src_words_value, tgt_words_value), dim=1)
            img_word_code = self.conv(img_word_code)
        else:
            img_word_code = torch.cat((img_code, src_words_value), dim=1)
        out_img_code = self.res_layer(img_word_code)
        out_img_code = self.up_layer(out_img_code)
        return out_img_code, src_words_attn


class Code2Image(nn.Module):
    def __init__(self, G_hidden):
        super().__init__()
        self.code2img = nn.Sequential(
            conv3x3(G_hidden, 3),
            nn.Tanh(),
        )

    def forward(self, code):
        image = self.code2img(code)
        return image

    
class Generator(nn.Module):
    def __init__(self, embedding_dim, condition_dim, noise_dim, G_hidden,
                 stage_num=3, num_resblocks=2, bilingual=True):
        super().__init__()
        self.stage_num = stage_num
        self.bilingual = bilingual

        self.CA = ConditioningAugmentation(embedding_dim, condition_dim, bilingual)
        
        if stage_num > 0:
            self.stage1 = G_First_Stage(condition_dim, noise_dim, G_hidden)
            self.code2image1 = Code2Image(G_hidden)
        if stage_num > 1:
            self.stage2 = G_AfterFirst_Stage(
                embedding_dim, condition_dim, G_hidden,
                num_resblocks=num_resblocks, bilingual=bilingual,
            )
            self.code2image2 = Code2Image(G_hidden)
        if stage_num > 2:
            self.stage3 = G_AfterFirst_Stage(
                embedding_dim, condition_dim, G_hidden,
                num_resblocks=num_resblocks, bilingual=bilingual,
            )
            self.code2image3 = Code2Image(G_hidden)

        self.apply(weights_init)

    def forward(self, noise, src_word_embs, src_sent_emb, src_mask=None,
                tgt_word_embs=None, tgt_sent_emb=None, tgt_mask=None, only_finest=False):
        fake_imgs = []
        attn_maps = []

        if self.bilingual:
            sent_embedding = torch.cat((src_sent_emb, tgt_sent_emb), dim=1)
        else:
            sent_embedding = src_sent_emb
        c_code, mu, log_var = self.CA(sent_embedding)

        if self.stage_num > 0:
            img_code = self.stage1(c_code, noise)
            if not only_finest:
                fake_img64 = self.code2image1(img_code)
                fake_imgs.append(fake_img64)
        if self.stage_num > 1:
            img_code, _ = self.stage2(img_code, src_word_embs, src_mask, tgt_word_embs, tgt_mask)
            if not only_finest:
                fake_img128 = self.code2image2(img_code)
                fake_imgs.append(fake_img128)
            #if attn128 is not None:
            #    attn_maps.append(attn128)
        if self.stage_num > 2:
            img_code, _ = self.stage3(img_code, src_word_embs, src_mask, tgt_word_embs, tgt_mask)
            fake_img256 = self.code2image3(img_code)
            fake_imgs.append(fake_img256)
            #if attn256 is not None:
            #    attn_maps.append(attn256)

        return fake_imgs, attn_maps, mu, log_var