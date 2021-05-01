import numpy as np

## [Pytorch] ############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
#########################################################################################


"""
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    # Returns cosine similarity between x1 and x2, computed along dim.

    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
"""

def DAMSM_attention(query, source, gamma1):
    """
    input:
        query(batch, feat_size, query_len) # word
        source(batch, feat_size, height, width) # image
    output:
        extraced_img_feat(batch, feat_size, seq_len)
    """
    batch_size, feat_size, query_len = query.size()
    height, width = source.size(2), source.size(3)
    source_len = height * width

    # (batch, feat_size, height, width) >> (batch, source_len, feat_size)
    source = source.view(batch_size, feat_size, source_len)
    sourceT = source.transpose(1, 2).contiguous()

    # (batch, source_len, feat_size) * (batch, feat_size, query_len) = (batch, source_len, query_len)
    attn = torch.bmm(sourceT, query)
    attn = F.softmax(attn, dim=2)
    # (batch, source_len, query_len) >> (batch, query_len, source_len)
    attn = attn.transpose(1, 2).contiguous()
    attn *= gamma1
    attn = F.softmax(attn, dim=2)

    # (batch, query_len, source_len) >> (batch, source_len, query_len)
    attnT = attn.transpose(1, 2).contiguous()

    # (batch, feat_size, source_len) * (batch, source_len, query_len) = (batch, feat_size, query_len)
    value = torch.bmm(source, attnT)

    return value, attn.view(batch_size, query_len, height, width)


def words_loss(img_feat, text_feat, labels, cap_lens, class_ids, opt):
    """
    input:
        img_feat(batch, feat_size, 17, 17)
        text_feat(batch, feat_size, seq_len)
        labels(batch) [0, 1, 2, 3, 4, 5, 6,...]
        cap_lens(batch)
        class_ids(batch)
        opt{batch_size, gamma1, gamma2, gamma3}
    """

    attn_maps = []
    similarities = []
    batch_size = img_feat.size(0)

    if not hasattr(words_loss, "arange"):
        words_loss.arange = torch.arange(batch_size).to(img_feat.device)
    arange = words_loss.arange[:batch_size]
    masks = class_ids.eq(class_ids[arange, None])
    masks[arange, arange] = False

    context = img_feat

    for i in range(batch_size):
        words_num = cap_lens[i]
        # (batch, feat_size, seq_len) >> (1, feat_size, words_num)
        words = text_feat[i, :, :words_num].unsqueeze(0).contiguous()
        # (1, feat_size, words_num) >> (batch, feat_size, words_num)
        words = words.repeat(batch_size, 1, 1)
        #context = img_feat

        # (batch, feat_size, words_num)
        extraced_img_feat, attn = DAMSM_attention(words, context, opt.gamma1)
        attn_maps.append(attn[i].unsqueeze(0).contiguous())

        # (batch, feat_size, words_num) >> (batch, words_num, feat_size)
        words = words.transpose(1, 2).contiguous()
        extraced_img_feat = extraced_img_feat.transpose(1, 2).contiguous()

        # (batch, words_num, feat_size) >> (batch * words_num, feat_size)
        words = words.view(batch_size * words_num, -1)
        extraced_img_feat = extraced_img_feat.view(batch_size * words_num, -1)

        # (batch * words_num, feat_size) >> (batch * words_num)
        row_simil = F.cosine_similarity(words, extraced_img_feat)
        # (batch * words_num) >> (batch, words_num)
        row_simil = row_simil.view(batch_size, words_num)

        # (batch, word_num) >> (batch, 1)
        row_simil.mul_(opt.gamma2).exp_()
        row_simil = row_simil.sum(dim=1, keepdim=True)
        row_simil = torch.log(row_simil)

        similarities.append(row_simil)

    # (batch, 1) * batch >> (batch, batch)
    similarities = torch.cat(similarities, dim=1)
    similarities *= opt.gamma3

    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))

    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = F.cross_entropy(similarities, labels)
        loss1 = F.cross_entropy(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, attn_maps


def sent_loss(img_feat, text_feat, labels, class_ids, opt, eps=1e-8):
    """
    input:
        img_feat(batch, feat_size)
        text_feat(batch, feat_size)
    """

    batch_size = img_feat.size(0)

    if not hasattr(sent_loss, "arange"):
        sent_loss.arange = torch.arange(batch_size).to(img_feat.device)
    arange = sent_loss.arange[:batch_size]
    masks = class_ids.eq(class_ids[arange, None])
    masks[arange, arange] = False


    # (batch, feat_size) >> (1, batch, feat_size)
    if img_feat.dim() == 2:
        img_feat = img_feat.unsqueeze(0)
        text_feat = text_feat.unsqueeze(0)

    # (1, batch, feat_size) >> (1, batch, 1)
    img_feat_norm = torch.norm(img_feat, 2, dim=2, keepdim=True)
    text_feat_norm = torch.norm(text_feat, 2, dim=2, keepdim=True)
    # (1, batch, feat_size) * (1, feat_size, batch) >> (1, batch, batch)
    scores0 = torch.bmm(img_feat, text_feat.transpose(1, 2))
    norm0 = torch.bmm(img_feat_norm, text_feat_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * opt.gamma3

    # (1, batch, batch) >> (batch, batch)
    scores0 = scores0.squeeze()

    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    
    if labels is not None:
        loss0 = F.cross_entropy(scores0, labels)
        loss1 = F.cross_entropy(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


# ##################Loss for G and Ds##############################
def discriminator_loss(netD, real_imgs, fake_imgs, conditions, real_labels, fake_labels):
    # Forward
    real_feat, uncond_real_logits, cond_real_logits = netD(real_imgs, conditions, return_feat=True)
    uncond_real_errD = F.binary_cross_entropy_with_logits(uncond_real_logits, real_labels)
    cond_real_errD = F.binary_cross_entropy_with_logits(cond_real_logits, real_labels)

    uncond_fake_logits, cond_fake_logits = netD(fake_imgs.detach(), conditions)
    uncond_fake_errD = F.binary_cross_entropy_with_logits(uncond_fake_logits, fake_labels)
    cond_fake_errD = F.binary_cross_entropy_with_logits(cond_fake_logits, fake_labels)

    batch_size = real_feat.size(0)
    cond_wrong_logits = netD.get_cond_logits(real_feat[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD = F.binary_cross_entropy_with_logits(cond_wrong_logits, fake_labels[1:batch_size])

    errD = ((uncond_real_errD + cond_real_errD) / 2. +
            (uncond_fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)

    """
    fake_features = netD(fake_imgs.detach())
    # loss
    #
    cond_real_logits = netD.COND_DNET(real_features, conditions)
    cond_real_errD = nn.BCEWithLogitsLoss()(cond_real_logits, real_labels)
    cond_fake_logits = netD.COND_DNET(fake_features, conditions)
    cond_fake_errD = nn.BCEWithLogitsLoss()(cond_fake_logits, fake_labels)
    #
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD = nn.BCEWithLogitsLoss()(cond_wrong_logits, fake_labels[1:batch_size])

    if netD.UNCOND_DNET is not None:
        real_logits = netD.UNCOND_DNET(real_features)
        fake_logits = netD.UNCOND_DNET(fake_features)
        real_errD = nn.BCEWithLogitsLoss()(real_logits, real_labels)
        fake_errD = nn.BCEWithLogitsLoss()(fake_logits, fake_labels)
        errD = ((real_errD + cond_real_errD) / 2. +
                (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
    else:
        errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.
    """
    return errD


def generator_loss(netD, fake_img, conditions, real_labels):
    uncond_logits, cond_logits = netD(fake_img, conditions)
    uncond_errG = F.binary_cross_entropy_with_logits(uncond_logits, real_labels)
    cond_errG = F.binary_cross_entropy_with_logits(cond_logits, real_labels)
    errG = uncond_errG + cond_errG

    return errG


def DAMSM_loss(DAMSM_CNN, fake_img, word_embs, sent_emb, match_labels, cap_lens, class_ids, opt):
    image_word_feat, image_sent_feat = DAMSM_CNN(fake_img)
    w_loss0, w_loss1, _ = words_loss(image_word_feat, word_embs, match_labels, cap_lens, class_ids, opt)
    w_loss = (w_loss0 + w_loss1) * opt.lambda1
    s_loss0, s_loss1 = sent_loss(image_sent_feat, sent_emb, match_labels, class_ids, opt)
    s_loss = (s_loss0 + s_loss1) * opt.lambda1

    return w_loss, s_loss


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    #KLD = torch.mean(KLD_element).mul_(-0.5)
    KLD = (KLD_element / KLD_element.numel()).sum()
    KLD = KLD.mul_(-0.5)
    return KLD
