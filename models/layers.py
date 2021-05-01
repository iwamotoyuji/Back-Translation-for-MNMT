## [Pytorch] ############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
#########################################################################################


class ScaledDotProductAttention(nn.Module):
    def __init__(self, sqrt_d_k, dropout=0.1):
        super().__init__()
        self.sqrt_d_k = sqrt_d_k
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, Q, K, V, mask=None):
        # Scaled
        Q /= self.sqrt_d_k
        
        # (batch * head_num, Q_len, d_v) × (batch * head_num, d_v, K_len) = (batch * head_num, Q_len, K_len)
        attn_weight = torch.bmm(Q, K.transpose(1, 2))

        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask, -float('inf'))
        attn_weight = self.softmax(attn_weight)
        attn_weight = self.dropout(attn_weight)

        # (batch * head_num, Q_len, K_len) × (batch * head_num, V_len, d_v) = (batch * head_num, Q_len, d_v)
        output = torch.bmm(attn_weight, V)

        return output, attn_weight


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head_num, d_k, d_v, dropout=0.1, init_weight=False):
        super().__init__()
        self.head_num = head_num
        self.d_k = d_k
        self.d_v = d_v

        self.W_q = nn.Linear(d_model, head_num * d_k)
        self.W_k = nn.Linear(d_model, head_num * d_k)
        self.W_v = nn.Linear(d_model, head_num * d_v)
        self.W_o = nn.Linear(head_num * d_v, d_model)
        if init_weight:
            nn.init.xavier_uniform_(self.W_q.weight, gain=1 / (2.0 ** 0.5))
            nn.init.xavier_uniform_(self.W_k.weight, gain=1 / (2.0 ** 0.5))
            nn.init.xavier_uniform_(self.W_v.weight, gain=1 / (2.0 ** 0.5))
            nn.init.xavier_uniform_(self.W_o.weight)

        self.attention = ScaledDotProductAttention(sqrt_d_k=d_k**0.5, dropout=dropout)

    def forward(self, Q, K, V, mask=None):
        # -- Get batch_size, Q_len, K_len, V_len --
        batch_size, Q_len, _ = Q.size()
        K_len = K.size(1)
        V_len = V.size(1)

        # (batch, len, d_model) >> (batch, len, head_num, d_k(d_v))
        Q = self.W_q(Q).view(batch_size, Q_len, self.head_num, self.d_k)
        K = self.W_k(K).view(batch_size, K_len, self.head_num, self.d_k)
        V = self.W_v(V).view(batch_size, V_len, self.head_num, self.d_v)

        # (batch, len, head_num, d_k(d_v)) >> (batch * head_num, len, d_k(d_v))
        Q = Q.permute(2, 0, 1, 3).contiguous().view(-1, Q_len, self.d_k)
        K = K.permute(2, 0, 1, 3).contiguous().view(-1, K_len, self.d_k)
        V = V.permute(2, 0, 1, 3).contiguous().view(-1, V_len, self.d_v)

        # -- Prepare Mask --
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)

        # -- Scaled Dot-Product Attention --
        output, attn_weight = self.attention(Q, K, V, mask=mask)

        # (batch * head_num, Q_len, d_v) >> (head_num, batch, Q_len, d_v)
        output = output.view(self.head_num, batch_size, Q_len, self.d_v)

        # (head_num, batch, Q_len, d_v) >> (batch, Q_len, head_num * d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, Q_len, -1)

        # (batch, Q_len, head_num * d_v) >> (batch, Q_len, d_model)
        output = self.W_o(output)

        return output, attn_weight


class Feed_Forward_Linear(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1, init_weight=False):
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_inner)
        self.W_2 = nn.Linear(d_inner, d_model)
        if init_weight:
            nn.init.xavier_uniform_(self.W_1.weight)
            nn.init.xavier_uniform_(self.W_2.weight)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        output = F.relu(self.W_1(x))
        output = self.W_2(self.dropout(output))
        return output


class ImageEncoderLayer(nn.Module):
    def __init__(self, d_model, head_num, d_k, d_v, d_inner,
                 dropout=0.1, init_weight=False, fused_layer_norm=False):
        super().__init__()

        if fused_layer_norm:
            from apex.normalization import FusedLayerNorm as LayerNorm
        else:
            from torch.nn import LayerNorm

        self.multi_self_attn = MultiHeadAttention(
            d_model, head_num, d_k, d_v, dropout=dropout, init_weight=init_weight
        )
        self.feed_forward = Feed_Forward_Linear(
            d_model, d_inner, dropout=dropout, init_weight=init_weight
        )
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, img_input):
        img_input = img_input.float()
        residual = img_input
        img_enc_output, img_enc_self_attn = self.multi_self_attn(
            img_input, img_input, img_input
        )
        img_enc_output = self.dropout(img_enc_output)
        img_enc_output = self.layer_norm1(img_enc_output + residual)
        residual = img_enc_output
        img_enc_output = self.feed_forward(img_enc_output)
        img_enc_output = self.dropout(img_enc_output)
        img_enc_output = self.layer_norm2(img_enc_output + residual)
 
        return img_enc_output, img_enc_self_attn


class TextEncoderLayer(nn.Module):
    def __init__(self, d_model, head_num, d_k, d_v, d_inner,
                 dropout=0.1, init_weight=False, fused_layer_norm=False):
        super().__init__()

        if fused_layer_norm:
            from apex.normalization import FusedLayerNorm as LayerNorm
        else:
            from torch.nn import LayerNorm

        self.multi_self_attn = MultiHeadAttention(
            d_model, head_num, d_k, d_v, dropout=dropout, init_weight=init_weight
        )
        self.multi_enc_img_attn = MultiHeadAttention(
            d_model, head_num, d_k, d_v, dropout=dropout, init_weight=init_weight
        )
        self.feed_forward = Feed_Forward_Linear(
            d_model, d_inner, dropout=dropout, init_weight=init_weight
        )
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, enc_input, img_enc_output=None, self_attn_mask=None):
        residual = enc_input
        enc_output, enc_self_attn = self.multi_self_attn(
            enc_input, enc_input, enc_input, mask=self_attn_mask
        )
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm1(enc_output + residual)
        enc_img_attn = None

        if img_enc_output is not None:
            residual = enc_output
            enc_output, enc_img_attn = self.multi_enc_img_attn(
                enc_output, img_enc_output, img_enc_output, mask=None
            )
            enc_output = self.dropout(enc_output)
            enc_output = self.layer_norm2(enc_output + residual)

        residual = enc_output
        enc_output = self.feed_forward(enc_output)
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm3(enc_output + residual)

        return enc_output, enc_self_attn, enc_img_attn



class EncoderLayer(nn.Module):
    def __init__(self, d_model, head_num, d_k, d_v, d_inner,
                 dropout=0.1, init_weight=False, fused_layer_norm=False):
        super().__init__()

        if fused_layer_norm:
            from apex.normalization import FusedLayerNorm as LayerNorm
        else:
            from torch.nn import LayerNorm

        self.multi_self_attn = MultiHeadAttention(
            d_model, head_num, d_k, d_v, dropout=dropout, init_weight=init_weight
        )
        self.feed_forward = Feed_Forward_Linear(
            d_model, d_inner, dropout=dropout, init_weight=init_weight
        )
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, enc_input, self_attn_mask=None):
        residual = enc_input
        enc_output, enc_self_attn = self.multi_self_attn(
            enc_input, enc_input, enc_input, mask=self_attn_mask
        )
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm1(enc_output + residual)

        residual = enc_output
        enc_output = self.feed_forward(enc_output)
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm2(enc_output + residual)

        return enc_output, enc_self_attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, head_num, d_k, d_v, d_inner,
                 dropout=0.1, init_weight=False, fused_layer_norm=False):
        super().__init__()

        if fused_layer_norm:
            from apex.normalization import FusedLayerNorm as LayerNorm
        else:
            from torch.nn import LayerNorm

        self.multi_self_attn = MultiHeadAttention(
            d_model, head_num, d_k, d_v, dropout=dropout, init_weight=init_weight
        )
        self.multi_dec_enc_attn = MultiHeadAttention(
            d_model, head_num, d_k, d_v, dropout=dropout, init_weight=init_weight
        )
        self.feed_forward = Feed_Forward_Linear(
            d_model, d_inner, dropout=dropout, init_weight=init_weight
        )
        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, dec_input, enc_output, self_attn_mask=None, dec_enc_attn_mask=None):
        residual = dec_input
        dec_output, dec_self_attn = self.multi_self_attn(
            dec_input, dec_input, dec_input, mask=self_attn_mask
        )
        dec_output = self.dropout(dec_output)
        dec_output = self.layer_norm1(dec_output + residual)
        residual = dec_output
        dec_output, dec_enc_attn = self.multi_dec_enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask
        )
        dec_output = self.dropout(dec_output)
        dec_output = self.layer_norm2(dec_output + residual)

        residual = dec_output
        dec_output = self.feed_forward(dec_output)
        dec_output = self.dropout(dec_output)
        dec_output = self.layer_norm3(dec_output + residual)

        return dec_output, dec_self_attn, dec_enc_attn
