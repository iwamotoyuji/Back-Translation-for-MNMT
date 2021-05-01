import math

## [Pytroch] ############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
#########################################################################################

## [Self-Module] ########################################################################
import my_utils.Constants as Constants
from models.layers import ImageEncoderLayer, TextEncoderLayer, EncoderLayer, DecoderLayer
from models.frelu import FReLU
#########################################################################################


def build_embedding(vocab_size, embedding_dim):
    embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=Constants.PAD)
    nn.init.normal_(embedding.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(embedding.weight[Constants.PAD], 0)
    return embedding

def make_sinusoid_encoding_table(max_position_num, d_model, padding_idx=None):
    half_d_model = d_model / 2
    magic = math.log(10000) / half_d_model
    magic = torch.exp(torch.arange(half_d_model, dtype=torch.float) * -magic)
    magic = torch.arange(max_position_num, dtype=torch.float).unsqueeze(1) * magic.unsqueeze(0)
    magic = torch.cat((torch.sin(magic).unsqueeze(-1), torch.cos(magic).unsqueeze(-1)), dim=2)
    magic = magic.view(max_position_num, -1)

    if padding_idx is not None:
        magic[padding_idx] = 0.
    return magic

def make_attn_pad_mask(Q_seq, K_seq):
    # (batch, K_len)
    attn_mask = K_seq.eq(Constants.PAD)
    # (batch, K_len) >> (batch, Q_len, K_len)
    attn_mask = attn_mask.unsqueeze(1).expand(-1, Q_seq.size(1), -1)
    return attn_mask

def make_subsequent_mask(seq):
    batch_size, seq_len = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((seq_len, seq_len), device=seq.device, dtype=torch.uint8),
        diagonal=1
    )
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)
    return subsequent_mask


class ImageEncoder(nn.Module):
    def __init__(self, d_model, head_num, d_k, d_v, d_inner, layer_num,
                 dropout=0.1, cnn_fine_tuning=False, init_weight=False,
                 fused_layer_norm=False):
        super().__init__()
        self.d_model = d_model

        resnet = models.resnet50(pretrained=True)
        
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        for param in self.resnet.parameters():
            param.requires_grad = cnn_fine_tuning
        self.fc = nn.Linear(resnet.fc.in_features, d_model)

        self.layer_stack = nn.ModuleList(
            [ImageEncoderLayer(
                d_model, head_num, d_k, d_v, d_inner,
                dropout, init_weight, fused_layer_norm,
            ) for _ in range(layer_num)]
        )
        self.CNN_dropout = nn.Dropout(p=0.5)
        #self.frelu = FReLU(d_model)

    def forward(self, image, return_attns=False):
        img_enc_self_attn_list = []

        batch_size = image.size(0)
        CNN_output = self.resnet(image)
        CNN_output = CNN_output.permute(0,2,3,1).contiguous()
        CNN_output = self.CNN_dropout(CNN_output)
        CNN_output = self.fc(CNN_output)
        
        CNN_output = F.relu(CNN_output)
        #CNN_output = CNN_output.permute(0,3,1,2).contiguous()
        #CNN_output = self.frelu(CNN_output, layer_norm=False)
        #CNN_output = CNN_output.permute(0,2,3,1).contiguous()

        CNN_output = CNN_output.view(batch_size, -1, self.d_model)

        img_enc_output = CNN_output
        for img_enc_layer in self.layer_stack:
            img_enc_output, img_enc_self_attn = img_enc_layer(img_enc_output)
            if return_attns:
                img_enc_self_attn_list += [img_enc_self_attn]

        if return_attns:
            return img_enc_output, img_enc_self_attn_list
        return img_enc_output,


class TextEncoder(nn.Module):
    def __init__(self, embedding, max_position_num, 
                 d_model, head_num, d_k, d_v, d_inner, layer_num,
                 dropout=0.1, init_weight=False, fused_layer_norm=False):
        super().__init__()
        self.sqrt_d_model = d_model ** 0.5

        self.src_embedding = embedding
        self.pos_encoding = nn.Embedding.from_pretrained(
            make_sinusoid_encoding_table(max_position_num, d_model, padding_idx=Constants.PAD),
            freeze=True
        )
        self.layer_stack = nn.ModuleList(
            [TextEncoderLayer(
                d_model, head_num, d_k, d_v, d_inner,
                dropout, init_weight, fused_layer_norm
            ) for _ in range(layer_num)]
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src_seq, src_pos, img_enc_output, return_attns=False):
        enc_self_attn_list, img_enc_attn_list = [], []

        # -- Prepare Mask --
        self_attn_mask = make_attn_pad_mask(Q_seq=src_seq, K_seq=src_seq)

        # -- Embedding + Positional Encoding --
        #enc_output = torch.tanh(self.src_embedding(src_seq) * self.sqrt_d_model) + self.pos_encoding(src_pos)
        enc_output = self.src_embedding(src_seq) * self.sqrt_d_model + self.pos_encoding(src_pos)
        enc_output = self.dropout(enc_output)
 
        # -- Encoder_layers forward --
        for enc_layer in self.layer_stack:
            enc_output, enc_self_attn, img_enc_attn = enc_layer(
                enc_output, img_enc_output,
                self_attn_mask=self_attn_mask
            )
            if return_attns:
                enc_self_attn_list += [enc_self_attn]
                img_enc_attn_list += [img_enc_attn]

        if return_attns:
            return enc_output, enc_self_attn_list, img_enc_attn_list
        return enc_output,


class Encoder(nn.Module):
    def __init__(self, embedding, max_position_num, 
                 d_model, head_num, d_k, d_v, d_inner, layer_num,
                 dropout=0.1, init_weight=False, fused_layer_norm=False):
        super().__init__()
        self.sqrt_d_model = d_model ** 0.5

        self.src_embedding = embedding
        self.pos_encoding = nn.Embedding.from_pretrained(
            make_sinusoid_encoding_table(max_position_num, d_model, padding_idx=Constants.PAD),
            freeze=True
        )
        self.layer_stack = nn.ModuleList(
            [EncoderLayer(
                d_model, head_num, d_k, d_v, d_inner,
                dropout, init_weight, fused_layer_norm
            ) for _ in range(layer_num)]
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src_seq, src_pos, return_attns=False):
        enc_self_attn_list = []

        # -- Prepare Mask --
        self_attn_mask = make_attn_pad_mask(Q_seq=src_seq, K_seq=src_seq)

        # -- Embedding + Positional Encoding --
        enc_output = self.src_embedding(src_seq) * self.sqrt_d_model + self.pos_encoding(src_pos)
        enc_output = self.dropout(enc_output)
 
        # -- Encoder_layers forward --
        for enc_layer in self.layer_stack:
            enc_output, enc_self_attn = enc_layer(
                enc_output,
                self_attn_mask=self_attn_mask
            )
            if return_attns:
                enc_self_attn_list += [enc_self_attn]

        if return_attns:
            return enc_output, enc_self_attn_list
        return enc_output,


class Decoder(nn.Module):
    def __init__(self, embedding, max_position_num,
                 d_model, head_num, d_k, d_v, d_inner, layer_num,
                 dropout=0.1, init_weight=False, fused_layer_norm=False):
        super().__init__()
        self.sqrt_d_model = d_model ** 0.5

        self.tgt_embedding = embedding
        self.pos_encoding = nn.Embedding.from_pretrained(
            make_sinusoid_encoding_table(max_position_num, d_model, padding_idx=Constants.PAD),
            freeze=True
        )
        self.layer_stack = nn.ModuleList(
            [DecoderLayer(
                d_model, head_num, d_k, d_v, d_inner,
                dropout, init_weight, fused_layer_norm
            ) for _ in range(layer_num)]
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
        dec_self_attn_list, dec_enc_attn_list = [], []

        # -- Prepare Masks --
        subsequent_mask = make_subsequent_mask(tgt_seq)
        attn_pad_mask = make_attn_pad_mask(Q_seq=tgt_seq, K_seq=tgt_seq)
        self_attn_mask = (subsequent_mask + attn_pad_mask).gt(0)

        dec_enc_attn_mask = make_attn_pad_mask(Q_seq=tgt_seq, K_seq=src_seq)
 
        # -- Embedding + Positional Encoding --
        dec_output = self.tgt_embedding(tgt_seq) * self.sqrt_d_model + self.pos_encoding(tgt_pos)
        dec_output = self.dropout(dec_output)

        # -- Decoder_layers forward --
        for dec_layer in self.layer_stack:
            dec_output, dec_self_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                self_attn_mask=self_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask
            )
            if return_attns:
                dec_self_attn_list += [dec_self_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_self_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, max_position_num,
                 d_model, head_num, d_k, d_v, d_inner, layer_num, dropout=0.1,
                 shared_embedding=False, share_dec_input_output_embed=False,
                 init_weight=False, fused_layer_norm=False):
        super().__init__()

        if shared_embedding:
            if src_vocab_size != tgt_vocab_size:
                raise ValueError("When using shared_embedding, please share the dictionary")
            self.encoder_embedding = build_embedding(tgt_vocab_size, d_model)
            self.decoder_embedding = self.encoder_embedding
        else:
            self.encoder_embedding = build_embedding(src_vocab_size, d_model)
            self.decoder_embedding = build_embedding(tgt_vocab_size, d_model)

        self.encoder = Encoder(
            self.encoder_embedding, max_position_num,
            d_model, head_num, d_k, d_v, d_inner, layer_num,
            dropout, init_weight, fused_layer_norm,
        )
        self.decoder = Decoder(
            self.decoder_embedding, max_position_num,
            d_model, head_num, d_k, d_v, d_inner, layer_num,
            dropout, init_weight, fused_layer_norm,
        )
        
        self.share_dec_input_output_embed = share_dec_input_output_embed
        if not self.share_dec_input_output_embed:
            self.output_embed = nn.Parameter(torch.Tensor(tgt_vocab_size, d_model))
            nn.init.xavier_uniform_(self.output_embed)
            self.output_embed_bias = nn.Parameter(torch.Tensor(tgt_vocab_size))
            nn.init.constant_(self.output_embed_bias, 0.)
        
    def get_device(self):
        return next(self.parameters()).device

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
        # -- Encode --
        enc_output, *_ = self.encoder(src_seq, src_pos)
        # -- Decode --
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)

        if self.share_dec_input_output_embed:
            dec_output = F.linear(dec_output, self.decoder_embedding.weight)
        else:
            dec_output = F.linear(dec_output, self.output_embed, self.output_embed_bias)

        # -- Reshape --
        dec_output = dec_output.view(-1, dec_output.size(-1))
        
        return dec_output

    def forward_encoder(self, src_seq, src_pos):
        enc_out, *_ = self.encoder(src_seq, src_pos)
        return (src_seq, enc_out)

    def forward_decoder(self, tgt_seq, tgt_pos, src_seq, enc_out):
        dec_out, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_out)
        if self.share_dec_input_output_embed:
            dec_out = F.linear(dec_out, self.decoder_embedding.weight)
        else:
            dec_out = F.linear(dec_out, self.output_embed, self.output_embed_bias)
        return dec_out

    def select_enc_outs(self, enc_outs, required_ids):
        src_seq, enc_out = enc_outs
        src_seq = src_seq.index_select(0, required_ids)
        enc_out = enc_out.index_select(0, required_ids)
        return (src_seq, enc_out)


class MultimodalTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, max_position_num,
                 d_model, head_num, d_k, d_v, d_inner, layer_num, dropout=0.1,
                 shared_embedding=False, share_dec_input_output_embed=False,
                 cnn_fine_tuning=False, init_weight=False, fused_layer_norm=False):
        super().__init__()

        if shared_embedding:
            if src_vocab_size != tgt_vocab_size:
                raise ValueError("When using shared_embedding, please share the dictionary")
            self.encoder_embedding = build_embedding(tgt_vocab_size, d_model)
            self.decoder_embedding = self.encoder_embedding
        else:
            self.encoder_embedding = build_embedding(src_vocab_size, d_model)
            self.decoder_embedding = build_embedding(tgt_vocab_size, d_model)

        self.img_encoder = ImageEncoder(
            d_model, head_num, d_k, d_v, d_inner, layer_num,
            dropout, cnn_fine_tuning, init_weight, fused_layer_norm,
        )
        self.encoder = TextEncoder(
            self.encoder_embedding, max_position_num,
            d_model, head_num, d_k, d_v, d_inner, layer_num,
            dropout, init_weight, fused_layer_norm,
        )
        self.decoder = Decoder(
            self.decoder_embedding, max_position_num,
            d_model, head_num, d_k, d_v, d_inner, layer_num,
            dropout, init_weight, fused_layer_norm,
        )
        
        self.share_dec_input_output_embed = share_dec_input_output_embed
        if not self.share_dec_input_output_embed:
            self.output_embed = nn.Parameter(torch.Tensor(tgt_vocab_size, d_model))
            nn.init.xavier_uniform_(self.output_embed)
            self.output_embed_bias = nn.Parameter(torch.Tensor(tgt_vocab_size))
            nn.init.constant_(self.output_embed_bias, 0.)

    def get_device(self):
        return next(self.parameters()).device

    def set_resnet_requires_grad(self, requires=False):
        for param in self.img_encoder.resnet.parameters():
            param.requires_grad = requires

    def forward(self, image, src_seq, src_pos, tgt_seq, tgt_pos):
        # -- Image Encode --
        img_enc_output, *_ = self.img_encoder(image)
        # -- Encode --
        enc_output, *_ = self.encoder(src_seq, src_pos, img_enc_output)
        # -- Decode --
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)

        if self.share_dec_input_output_embed:
            dec_output = F.linear(dec_output, self.decoder_embedding.weight)
        else:
            dec_output = F.linear(dec_output, self.output_embed, self.output_embed_bias)
            
        # -- Reshape --
        dec_output = dec_output.view(-1, dec_output.size(-1))
        
        return dec_output

    def forward_encoder(self, image, src_seq, src_pos):
        if image is None:
            img_enc_out = None
        else:
            img_enc_out, *_ = self.img_encoder(image)
        enc_out, *_ = self.encoder(src_seq, src_pos, img_enc_out)
        return (src_seq, enc_out)

    def forward_decoder(self, tgt_seq, tgt_pos, src_seq, enc_out):
        dec_out, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_out)
        if self.share_dec_input_output_embed:
            dec_out = F.linear(dec_out, self.decoder_embedding.weight)
        else:
            dec_out = F.linear(dec_out, self.output_embed, self.output_embed_bias)
        return dec_out

    def select_enc_outs(self, enc_outs, required_ids):
        src_seq, enc_out = enc_outs
        src_seq = src_seq.index_select(0, required_ids)
        enc_out = enc_out.index_select(0, required_ids)
        return (src_seq, enc_out)


def check_arguments(old_args, new_args):
    def must_equal(arg_name, old_arg, new_arg):
        if old_arg != new_arg:
            raise ValueError(f"{arg_name} should be the equal. (old:{old_arg} new:{new_arg})")

    must_equal("src_vocab_size", old_args.src_vocab_size, new_args.src_vocab_size)
    must_equal("tgt_vocab_size", old_args.tgt_vocab_size, new_args.tgt_vocab_size)
    must_equal("d_model", old_args.d_model, new_args.d_model)
    must_equal("head_num", old_args.head_num, new_args.head_num)
    must_equal("d_k", old_args.d_k, new_args.d_k)
    must_equal("d_v", old_args.d_v, new_args.d_v)
    must_equal("d_inner", old_args.d_inner, new_args.d_inner)
    must_equal("layer_num", old_args.layer_num, new_args.layer_num)
    must_equal("shared_embedding", old_args.shared_embedding, new_args.shared_embedding)
    must_equal("share_dec_input_output_embed", old_args.share_dec_input_output_embed, new_args.share_dec_input_output_embed)
    must_equal("fused_layer_norm", old_args.fused_layer_norm, new_args.fused_layer_norm)