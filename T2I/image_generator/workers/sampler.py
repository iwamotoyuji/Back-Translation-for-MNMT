import os
from tqdm import tqdm
import numpy as np
from PIL import Image

## [Pytorch] ############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
#########################################################################################

## [Self-Module] ########################################################################
from datasets.T2I_dataset import ModelConnector
from my_utils.general_utils import mkdirs
from my_utils.pytorch_utils import get_device
import my_utils.Constants as Constants
#########################################################################################


class Sampler():
    def __init__(self, src_DAMSM_RNN, tgt_DAMSM_RNN, netG, MMT, dataloader, opt):
        self.src_DAMSM_RNN = src_DAMSM_RNN
        self.tgt_DAMSM_RNN = tgt_DAMSM_RNN
        self.netG = netG
        self.MMT = MMT

        self.dataloader = dataloader
        self.device = get_device(self.netG)
        self.opt = opt
        
        self.model_connector = ModelConnector(
            opt.batch_size, opt.words_limit,
            MMT_id2word=dataloader.dataset.MMT_index2word,
            T2I_word2id=dataloader.dataset.tgt_word2index,
            bpe=opt.bpe,
        )


    def get_dec_pos(self, dec_seq_len, inst_num):
        # (dec_seq_len)
        dec_pos = torch.arange(1, dec_seq_len + 1, dtype=torch.long)
        # (dec_seq_len) >> (inst_num, dec_seq_len)
        dec_pos = dec_pos.unsqueeze(0).repeat(inst_num, 1)
        return dec_pos

    def generate_T2I_tgt_text(self, MMT_imgs, MMT_src_text, MMT_src_pos):
        MMT_imgs = F.interpolate(MMT_imgs, (224,224))
        # --- Apex level O2 only ---
        #if self.opt.apex:
        #    MMT_imgs = MMT_imgs.half()

        MMT_imgs = MMT_imgs.to(self.device)
        MMT_src_text = MMT_src_text.to(self.device)
        MMT_src_pos = MMT_src_pos.to(self.device)

        MMT = self.MMT.module if "DataParallel" in str(type(self.MMT)) else self.MMT
        enc_outs = MMT.forward_encoder(MMT_imgs, MMT_src_text, MMT_src_pos)

        batch_size, src_len = enc_outs[0].size()
        dec_max_len = src_len + 20

        remaining_batch_num = batch_size
        is_finished = [False for _ in range(batch_size)]
        finalized_info = [None for _ in range(batch_size)]

        dec_seqs = torch.full((batch_size, dec_max_len + 2), Constants.PAD, dtype=torch.long)
        dec_seqs[:, 0] = Constants.BOS

        required_batch_ids = None
        for step in range(dec_max_len + 1):
            if required_batch_ids is not None:
                enc_outs = MMT.select_enc_outs(enc_outs, required_batch_ids.to(self.device))
            
            dec_pos = self.get_dec_pos(step + 1, dec_seqs.size(0))
            dec_out = MMT.forward_decoder(
                dec_seqs[:, :step + 1].to(self.device),
                dec_pos.to(self.device),
                *enc_outs
            )

            dec_out = dec_out[:, -1, :].cpu()
            # --- Apex level O2 only ---
            #if self.opt.apex:
            #    dec_out = dec_out.float()

            word_ids = torch.argmax(dec_out, dim=1)

            if step >= dec_max_len:
                word_ids[:] = Constants.EOS
            
            dec_seqs[:, step + 1] = word_ids

            eos_mask = word_ids.eq(Constants.EOS)
            finished_batch_ids = [x.item() for x in eos_mask.nonzero()]          

            finished_batch_num = len(finished_batch_ids)
            if finished_batch_num > 0:
                finalize_dec_seqs = dec_seqs[eos_mask][:, 1:step+2]
                
                curr_to_whole_batch_idx = []
                whole_batch_idx = 0
                for f in is_finished:
                    if not f:
                        curr_to_whole_batch_idx.append(whole_batch_idx)
                    whole_batch_idx += 1

                for i in range(finished_batch_num):
                    whole_batch_idx = curr_to_whole_batch_idx[finished_batch_ids[i]]
                    assert finalized_info[whole_batch_idx] is None                    
                    finalized_info[whole_batch_idx] = finalize_dec_seqs[i]
                    is_finished[whole_batch_idx] = True

                remaining_batch_num -= finished_batch_num

                # --- Break Check ---
                assert remaining_batch_num >= 0    
                if remaining_batch_num == 0:
                    break
                assert step < dec_max_len

                dec_seqs = dec_seqs[~eos_mask]
                required_batch_ids = (~eos_mask).nonzero().squeeze(-1)
            else:
                required_batch_ids = None

        #finalized_info = np.array(finalized_info)
        T2I_tgt_text, T2I_tgt_len = self.model_connector.make_T2I_tgt(finalized_info)

        return T2I_tgt_text, T2I_tgt_len


    @torch.no_grad()
    def sampling(self):
        self.src_DAMSM_RNN.eval()
        self.tgt_DAMSM_RNN.eval()
        self.netG.eval()
        self.MMT.eval()

        device = get_device(self.netG)

        save_dir = self.opt.save_image_dir
        noise = torch.FloatTensor(self.opt.batch_size, self.opt.noise_dim).to(device)

        for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
            pbar = tqdm(self.dataloader, ascii=True, mininterval=0.5, ncols=90)
            for imgs, MMT_src_text, MMT_src_pos, T2I_src_text, T2I_src_len, filenames in pbar:
                T2I_tgt_text, T2I_tgt_len = self.generate_T2I_tgt_text(imgs[-1], MMT_src_text, MMT_src_pos)
                
                batch_size = MMT_src_text.size(0)
                T2I_src_text = T2I_src_text.to(self.device)
                T2I_src_len = T2I_src_len.to(self.device)
                T2I_tgt_text = T2I_tgt_text.to(self.device)
                T2I_tgt_len = T2I_tgt_len.to(self.device)

                ##########################################################
                # (1) Prepare training data and Compute text embeddings
                ##########################################################
                src_words_embs, src_sent_emb = self.src_DAMSM_RNN(T2I_src_text, T2I_src_len)
                src_mask = (T2I_src_text == Constants.PAD)
                num_words = src_words_embs.size(2)
                if src_mask.size(1) > num_words:
                    src_mask = src_mask[:, :num_words]

                tgt_words_embs, tgt_sent_emb = self.tgt_DAMSM_RNN(T2I_tgt_text, T2I_tgt_len)
                tgt_mask = (T2I_tgt_text == Constants.PAD)
                num_words = tgt_words_embs.size(2)
                if tgt_mask.size(1) > num_words:
                    tgt_mask = tgt_mask[:, :num_words]

                ##########################################################
                # (2) Generate fake images
                ##########################################################
                noise.data.normal_(0, 1)
                fake_imgs, _, _, _ = self.netG(
                    noise[:batch_size], src_words_embs, tgt_words_embs,
                    src_sent_emb, tgt_sent_emb,
                    src_mask, tgt_mask
                )

                ##########################################################
                # (3) Save images
                ##########################################################
                for j in range(batch_size):
                    file_path = '%s/%s' % (save_dir, filenames[j])
                    k = -1
                    # for k in range(len(fake_imgs)):
                    im = fake_imgs[k][j].data.cpu().numpy()
                    # [-1, 1] --> [0, 255]
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    im.save(file_path)
