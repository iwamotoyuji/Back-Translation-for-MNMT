from time import time
from tqdm import tqdm

## [Pytorch] ############################################################################
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast
#########################################################################################

## [Self-Module] ########################################################################
from datasets.BT_T2I_dataset import ModelConnector
import my_utils.Constants as Constants
from my_utils.losses import words_loss
from my_utils.losses import discriminator_loss, generator_loss, DAMSM_loss, KL_loss
from my_utils.general_utils import get_logger, args2string
from my_utils.pytorch_utils import get_state_dict, set_requires_grad, get_device, load_params, copy_params
#########################################################################################


class Trainer():
    def __init__(self, MNMT, src_DAMSM_CNN, src_DAMSM_RNN, tgt_DAMSM_RNN, netG, netsD,
                 MNMT_optimizer, netG_optimizer, netD_optimizers, DAMSM_optimizer,
                 MNMT_loader, T2I_loader, scaler, scheduler, opt, validator=None):
        self.MNMT = MNMT         
        self.src_DAMSM_CNN = src_DAMSM_CNN
        self.src_DAMSM_RNN = src_DAMSM_RNN
        self.tgt_DAMSM_RNN = tgt_DAMSM_RNN
        self.netG = netG
        self.netsD = netsD
        self.MNMT_optimizer = MNMT_optimizer        
        self.netG_optimizer = netG_optimizer
        self.netD_optimizers = netD_optimizers
        self.DAMSM_optimizer = DAMSM_optimizer
        self.MNMT_loader = MNMT_loader        
        self.T2I_loader = T2I_loader
        self.scaler = scaler
        self.scheduler = scheduler
        self.opt = opt
        self.validator = validator

        self.save_model_dir = opt.save_model_dir
        self.save_image_dir = opt.save_image_dir
        self.stage_num = opt.stage_num

        self.device = get_device(self.netG)
        self.avg_param_G = copy_params(self.netG)
        self.real_labels = torch.FloatTensor(opt.T2I_batch_size).fill_(1).to(self.device)
        self.fake_labels = torch.FloatTensor(opt.T2I_batch_size).fill_(0).to(self.device)
        self.match_labels = torch.LongTensor(range(opt.T2I_batch_size)).to(self.device)
        self.T2I_noise = torch.FloatTensor(opt.T2I_batch_size, opt.noise_dim).to(self.device)
        self.MNMT_noise = torch.FloatTensor(opt.MNMT_batch_size, opt.noise_dim).to(self.device)
        self.model_connector = ModelConnector(
            opt.T2I_batch_size, opt.train_words_limit,
            MNMT_id2word=MNMT_loader.dataset.tgt_index2word,
            T2I_word2id=T2I_loader.dataset.tgt_word2index,
            bpe=opt.bpe,
        )

        self.logger = get_logger(opt.save_log_path, overwrite=opt.overwrite)
        self.logger.info(args2string(opt))

        self.best_bleu_score = 0.
        self.best_cnt = 0
        self.stop_cnt = 0


    def save_models(self, cnt, state_dict, model_name):
        save_dict = {
            "cnt": cnt,
            "models": {
                "MNMT": state_dict,
                "tgt_DAMSM": None if self.tgt_DAMSM_RNN is None else get_state_dict(self.tgt_DAMSM_RNN),
                "netG": get_state_dict(self.netG),
            },
            "optims": {
                "MNMT": self.MNMT_optimizer.state_dict(),
                "tgt_DAMSM": None if self.DAMSM_optimizer is None else self.DAMSM_optimizer.state_dict(),
                "netG": self.netG_optimizer.state_dict(),
            },
            "scaler": self.scaler.state_dict(),
            "steps_cnt": self.scheduler.current_steps,
            "settings": self.opt,
        }
        for i in range(self.stage_num):
            netD_name = 'netD_' + str(64 * 2**i)
            save_dict["models"][netD_name] = get_state_dict(self.netsD[i])
            save_dict["optims"][netD_name] = self.netD_optimizers[i].state_dict()
        torch.save(save_dict, f"{self.save_model_dir}/{model_name}")

    def get_dec_pos(self, dec_seq_len, inst_num):
        # (dec_seq_len)
        dec_pos = torch.arange(1, dec_seq_len + 1, dtype=torch.long)
        # (dec_seq_len) >> (inst_num, dec_seq_len)
        dec_pos = dec_pos.unsqueeze(0).repeat(inst_num, 1)
        return dec_pos

    def make_tgt_seq(self, teacher, batch_size):
        BOS_tensor = torch.full(
            (batch_size, 1),
            Constants.BOS,
            dtype=torch.long,
        )
        tgt_seq = teacher.clone()
        tgt_seq[teacher == Constants.EOS] = Constants.PAD
        tgt_seq = torch.cat([BOS_tensor, tgt_seq], dim=1)[:,:-1]
        return tgt_seq

    def text_embedding(self, model, text, text_len):
        text = text.to(self.device)
        #text_len = text_len.to(self.device)
        word_embs, sent_emb = model(text, text_len)
        mask = (text == Constants.PAD)
        num_words = word_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]
        return word_embs, sent_emb, mask

    def validation(self, cnt):
        bleu_score, state_dict = self.validator.get_bleu(
            use_beam=self.opt.use_beam, return_sentences=False
        )
        self.logger.info(f"bleu : {bleu_score:.4f}")
        
        if bleu_score >= self.best_bleu_score:
            self.best_bleu_score = bleu_score
            self.best_cnt = cnt
            self.stop_cnt = 0
            self.save_models(cnt, state_dict, "best.pth")
        else:
            self.stop_cnt += 1

        return state_dict

    def generate_T2I_tgt_text(self, MNMT_imgs, MNMT_src_text, MNMT_src_pos):
        MNMT_imgs = F.interpolate(MNMT_imgs, (224,224))

        MNMT_imgs = MNMT_imgs.to(self.device)
        MNMT_src_text = MNMT_src_text.to(self.device)
        MNMT_src_pos = MNMT_src_pos.to(self.device)

        MNMT = self.MNMT.module if "DataParallel" in str(type(self.MNMT)) else self.MNMT
        enc_outs = MNMT.forward_encoder(MNMT_imgs, MNMT_src_text, MNMT_src_pos)

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
                enc_outs = MNMT.select_enc_outs(enc_outs, required_batch_ids.to(self.device))
            
            dec_pos = self.get_dec_pos(step + 1, dec_seqs.size(0))
            dec_out = MNMT.forward_decoder(
                dec_seqs[:, :step + 1].to(self.device),
                dec_pos.to(self.device),
                *enc_outs
            )

            dec_out = dec_out[:, -1, :].cpu()
            word_ids = torch.argmax(dec_out, dim=1)

            if step >= dec_max_len:
                word_ids[:] = Constants.EOS
            
            dec_seqs[:, step + 1] = word_ids

            eos_mask = word_ids.eq(Constants.EOS)
            finished_batch_ids = [x.item() for x in eos_mask.nonzero(as_tuple=False)]          

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
                required_batch_ids = (~eos_mask).nonzero(as_tuple=False).squeeze(-1)
            else:
                required_batch_ids = None

        T2I_tgt_text, T2I_tgt_len = self.model_connector.make_T2I_tgt(finalized_info)

        return T2I_tgt_text, T2I_tgt_len


    def T2I_train_epoch(self):
        # --- Change model mode and Set requires_grad ---
        self.MNMT.eval()
        set_requires_grad(self.MNMT, False)

        if self.opt.bilingual:
            if self.opt.tgt_rnn_fine_tuning:
                self.tgt_DAMSM_RNN.train()
                self.DAMSM_optimizer.zero_grad()
            else:
                self.tgt_DAMSM_RNN.eval()
            set_requires_grad(self.tgt_DAMSM_RNN, self.opt.tgt_rnn_fine_tuning)
        self.netG.train()
        set_requires_grad(self.netG, True)
        self.netG_optimizer.zero_grad()
        for i in range(self.stage_num):
            self.netsD[i].train()
            self.netD_optimizers[i].zero_grad()
        
        epoch_netD_loss = [0.] * self.stage_num
        epoch_netG_loss = [0.] * (self.stage_num + 3) # w_loss, s_loss, KL_loss

        pbar = tqdm(self.T2I_loader, ascii=True, mininterval=0.5)
        for batch_cnt, train_datas in enumerate(pbar, 1):
            imgs, MNMT_src_text, MNMT_src_pos, T2I_src_text, T2I_src_len, class_ids = train_datas
            
            with autocast(self.opt.use_amp):
                if self.opt.bilingual:
                    T2I_tgt_text, T2I_tgt_len = self.generate_T2I_tgt_text(imgs[-1], MNMT_src_text, MNMT_src_pos)

                imgs = [img.to(self.device) for img in imgs]
                class_ids = class_ids.to(self.device)

                src_word_embs, src_sent_emb, src_mask = self.text_embedding(self.src_DAMSM_RNN, T2I_src_text, T2I_src_len)
                src_word_embs, src_sent_emb = src_word_embs.detach(), src_sent_emb.detach()

                if self.opt.bilingual:
                    tgt_word_embs, tgt_sent_emb, tgt_mask = self.text_embedding(self.tgt_DAMSM_RNN, T2I_tgt_text, T2I_tgt_len)
                    if not self.opt.tgt_rnn_fine_tuning:
                        tgt_word_embs, tgt_sent_emb = tgt_word_embs.detach(), tgt_sent_emb.detach()                    
                else:
                    tgt_word_embs = None
                    tgt_sent_emb = None
                    tgt_mask = None

                self.T2I_noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = self.netG(
                    self.T2I_noise, src_word_embs, src_sent_emb, src_mask,
                    tgt_word_embs, tgt_sent_emb, tgt_mask,
                )

                for i in range(self.stage_num):
                    self.netD_optimizers[i].zero_grad()
                    errD = discriminator_loss(
                        self.netsD[i], imgs[i], fake_imgs[i], src_sent_emb,
                        self.real_labels, self.fake_labels,
                    )
                    self.scaler.scale(errD).backward()
                    self.scaler.step(self.netD_optimizers[i])
                    self.scaler.update()
                    epoch_netD_loss[i] += errD.item()

                errG_total = 0.
                self.netG_optimizer.zero_grad()
                if self.opt.tgt_rnn_fine_tuning:
                    self.DAMSM_optimizer.zero_grad()
                for i in range(self.stage_num):
                    errG = generator_loss(
                        self.netsD[i], fake_imgs[i], src_sent_emb, self.real_labels,
                    )
                    errG_total += errG
                    epoch_netG_loss[i] += errG.item()

                    if i == (self.stage_num -1):
                        w_loss, s_loss = DAMSM_loss(
                            self.src_DAMSM_CNN, fake_imgs[i], src_word_embs,
                            src_sent_emb, self.match_labels, T2I_src_len,
                            class_ids, self.opt,
                        )
                        errG_total += w_loss + s_loss
                        epoch_netG_loss[i+1] += w_loss.item()
                        epoch_netG_loss[i+2] += s_loss.item()

                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                epoch_netG_loss[i+3] += kl_loss.item()

                self.scaler.scale(errG_total).backward()
                self.scaler.step(self.netG_optimizer)
                if self.opt.tgt_rnn_fine_tuning:
                    self.scaler.unscale_(self.DAMSM_optimizer)
                    clip_grad_norm_(self.tgt_DAMSM_RNN.parameters(), 0.25)
                    self.scaler.step(self.DAMSM_optimizer)
                self.scaler.update()

            for p, avg_p in zip(self.netG.parameters(), self.avg_param_G):
                avg_p.mul_(0.999).add_(p.data, alpha=0.001)

            """
            # save images
            if gen_iterations % 1000 == 0:
                backup_para = copy_G_params(self.netG)
                load_params(self.netG, avg_param_G)
                self.save_img_results(fixed_noise, sent_emb,
                                        words_embs, mask, self.DAMSM_CNN,
                                        captions, cap_lens, epoch, name='average')
                load_params(self.netG, backup_para)
            """

            batch_netD_loss = [loss / batch_cnt for loss in epoch_netD_loss]
            batch_netG_loss = [loss / batch_cnt for loss in epoch_netG_loss]

            D_logs = ""
            G_logs = ""
            for i in range(self.stage_num):
                D_logs += f"D{i}: {batch_netD_loss[i]:.2f} "
                G_logs += f"G{i}: {batch_netG_loss[i]:.2f} "
            G_logs += f"w_loss: {batch_netG_loss[i+1]:.2f} "
            G_logs += f"s_loss: {batch_netG_loss[i+2]:.2f} "
            G_logs += f"KL_loss: {batch_netG_loss[i+3]:.2f} "

            total_netD_loss = sum(batch_netD_loss)
            D_logs += f"total: {total_netD_loss:.2f}"
            total_netG_loss = sum(batch_netG_loss)
            G_logs += f"total: {total_netG_loss:.2f}"

            pbar.set_description(
                f"{D_logs} | {G_logs}"
            )
            #torch.cuda.empty_cache()

        return D_logs, G_logs


    def generate_MNMT_imgs(self, T2I_src_text, T2I_src_len, T2I_tgt_text, T2I_tgt_len):
        batch_size = T2I_src_text.size(0)

        src_word_embs, src_sent_emb, src_mask = self.text_embedding(self.src_DAMSM_RNN, T2I_src_text, T2I_src_len)
        if self.opt.bilingual:
            tgt_word_embs, tgt_sent_emb, tgt_mask = self.text_embedding(self.tgt_DAMSM_RNN, T2I_tgt_text, T2I_tgt_len)
        else:
            tgt_word_embs = None
            tgt_sent_emb = None
            tgt_mask = None

        MNMT_noise = self.MNMT_noise[:batch_size]
        MNMT_noise.data.normal_(0, 1)

        fake_imgs, *_ = self.netG(
            MNMT_noise, src_word_embs, src_sent_emb, src_mask,
            tgt_word_embs, tgt_sent_emb, tgt_mask,
            only_finest=True,
        )
        MNMT_imgs = fake_imgs[-1]
        MNMT_imgs = F.interpolate(MNMT_imgs, (224,224))

        return MNMT_imgs


    def MNMT_train_epoch(self):
        # --- Change model mode and Set requires_grad ---
        self.MNMT.train()
        set_requires_grad(self.MNMT, True)
        if not self.opt.cnn_fine_tuning:
            MNMT = self.MNMT.module if "DataParallel" in str(type(self.MNMT)) else self.MNMT
            MNMT.set_resnet_requires_grad(False)

        if self.opt.bilingual:
            self.tgt_DAMSM_RNN.eval()
            set_requires_grad(self.tgt_DAMSM_RNN, False)
        self.netG.eval()
        set_requires_grad(self.netG, False)
        
        self.MNMT_optimizer.zero_grad()
        epoch_loss = 0.
        
        pbar = tqdm(self.MNMT_loader, ncols=90, mininterval=0.5, ascii=True)
        for batch_cnt, (T2I_src_text, T2I_src_len, T2I_tgt_text, T2I_tgt_len, MNMT_src_text, MNMT_src_pos, teacher, MNMT_tgt_pos) in enumerate(pbar, 1):
            with autocast(self.opt.use_amp):
                MNMT_imgs = self.generate_MNMT_imgs(T2I_src_text, T2I_src_len, T2I_tgt_text, T2I_tgt_len)

                MNMT_tgt_text = self.make_tgt_seq(teacher, teacher.size(0))
                MNMT_src_text = MNMT_src_text.to(self.device)
                MNMT_src_pos = MNMT_src_pos.to(self.device)
                MNMT_tgt_text = MNMT_tgt_text.to(self.device)
                MNMT_tgt_pos = MNMT_tgt_pos.to(self.device)
                teacher = teacher.to(self.device)

                pred = self.MNMT(MNMT_imgs, MNMT_src_text, MNMT_src_pos, MNMT_tgt_text, MNMT_tgt_pos)
                teacher = teacher.contiguous().view(-1)
                if self.opt.no_smoothing:
                    loss = F.cross_entropy(pred, teacher, ignore_index=Constants.PAD)
                else:
                    eps = 0.1
                    class_num = pred.size(1)

                    one_hot = torch.zeros_like(pred).scatter(1, teacher.view(-1, 1), 1)
                    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (class_num - 1)
                    log_prb = F.log_softmax(pred, dim=1)

                    non_pad_mask = teacher.ne(Constants.PAD)
                    loss = -(one_hot * log_prb).sum(dim=1)
                    loss = loss.masked_select(non_pad_mask).mean()

            epoch_loss += loss.item()
            loss /= self.opt.MNMT_grad_accumulation

            self.scaler.scale(loss).backward()

            if batch_cnt % self.opt.MNMT_grad_accumulation == 0:
                self.scaler.unscale_(self.MNMT_optimizer)
                clip_grad_norm_(self.MNMT.parameters(), self.opt.max_norm)
                self.scheduler.update_lr()
                self.scaler.step(self.MNMT_optimizer)
                self.scaler.update()
                self.MNMT_optimizer.zero_grad()

            avg_epoch_loss = epoch_loss / batch_cnt
            logs = f"word_loss : {avg_epoch_loss:.2f}"
            pbar.set_description(logs)
            #torch.cuda.empty_cache()

        return logs 


    def train(self, start_cnt):
        self.device = get_device(self.MNMT)

        start_all = time()
        for epoch_cnt in range(start_cnt, self.opt.max_epoch + 1):
            self.logger.info(f"\n[ Epoch {epoch_cnt} ]")

            # --- train MNMT ---
            start_span = time()
            backup_para = copy_params(self.netG)

            load_params(self.netG, self.avg_param_G)
            logs = self.MNMT_train_epoch()
            time_span = (time() - start_span) / 60
            self.logger.info(f"{logs}, time : {time_span:.2f} min")

            # --- valid MNMT ---
            if self.validator is not None:
                state_dict = self.validation(epoch_cnt)
            else:
                state_dict = get_state_dict(self.MNMT)
            if self.stop_cnt == self.opt.early_stop:
                break
            self.save_models(epoch_cnt, state_dict, f"epoch_{epoch_cnt}.pth")

            # --- train T2I ---
            start_span = time()
            load_params(self.netG, backup_para)
            for _ in range(self.opt.T2I_per_MNMT):
                D_logs, G_logs = self.T2I_train_epoch()
            time_span = (time() - start_span) / 60
            self.logger.info(f"{D_logs}\n{G_logs}\ntime : {time_span:.2f} min")

        time_all = (time() - start_all) / 3600
        self.logger.info(f"\nbest_epoch : {self.best_cnt}, best_score : {self.best_bleu_score}, time : {time_all:.2f} h")
