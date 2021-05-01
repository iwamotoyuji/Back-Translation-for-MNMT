from time import time
from tqdm import tqdm

## [Pytorch] ############################################################################
import torch
from torch.cuda.amp import autocast
from torchvision.utils import save_image
#########################################################################################

## [Self-Module] ########################################################################
import my_utils.Constants as Constants
from my_utils.general_utils import get_logger, args2string
from my_utils.pytorch_utils import get_state_dict, get_device, load_params, copy_params, set_requires_grad
from my_utils.losses import discriminator_loss, generator_loss, DAMSM_loss, KL_loss, words_loss
#########################################################################################


class Trainer():
    def __init__(self, src_DAMSM_CNN, src_DAMSM_RNN, tgt_DAMSM_RNN, netG, netsD,
                 netG_optimizer, netD_optimizers, train_loader, scaler, opt):
        self.src_DAMSM_CNN = src_DAMSM_CNN
        self.src_DAMSM_RNN = src_DAMSM_RNN
        self.tgt_DAMSM_RNN = tgt_DAMSM_RNN
        self.netG = netG
        self.netsD = netsD
        self.netG_optimizer = netG_optimizer
        self.netD_optimizers = netD_optimizers
        self.train_loader = train_loader
        self.scaler = scaler
        self.opt = opt

        self.save_model_dir = opt.save_model_dir
        self.save_image_dir = opt.save_image_dir
        self.stage_num = opt.stage_num

        self.device = get_device(self.netG)
        self.avg_param_G = copy_params(self.netG)
        self.real_labels = torch.FloatTensor(opt.batch_size).fill_(1).to(self.device)
        self.fake_labels = torch.FloatTensor(opt.batch_size).fill_(0).to(self.device)
        self.match_labels = torch.arange(opt.batch_size).to(self.device)
        self.noise = torch.FloatTensor(opt.batch_size, opt.noise_dim).to(self.device)
        self.fixed_noise = torch.FloatTensor(opt.batch_size, opt.noise_dim).normal_(0, 1).to(self.device)
        self.src_word_embs, self.src_sent_emb, self.src_mask, self.tgt_word_embs, self.tgt_sent_emb, self.tgt_mask = self.get_fixed_embs()

        self.logger = get_logger(opt.save_log_path, overwrite=opt.overwrite)
        self.logger.info(args2string(opt))


    def get_fixed_embs(self):
        data_iter = iter(self.train_loader)
        if self.opt.bilingual:
            imgs, src_text, src_text_len, tgt_text, tgt_text_len, class_ids, _ = data_iter.next()
        else:
            imgs, src_text, src_text_len, class_ids, _ = data_iter.next()

        src_word_embs, src_sent_emb, src_mask = self.text_embedding(self.src_DAMSM_RNN, src_text, src_text_len)
        src_word_embs, src_sent_emb = src_word_embs.detach(), src_sent_emb.detach()

        if self.opt.bilingual:
            tgt_word_embs, tgt_sent_emb, tgt_mask = self.text_embedding(self.tgt_DAMSM_RNN, tgt_text, tgt_text_len)
            tgt_word_embs, tgt_sent_emb = tgt_word_embs.detach(), tgt_sent_emb.detach()            
        else:
            tgt_word_embs = None
            tgt_sent_emb = None
            tgt_mask = None

        return src_word_embs, src_sent_emb, src_mask, tgt_word_embs, tgt_sent_emb, tgt_mask


    def save_fixed_images(self, epoch_i):
        backup_para = copy_params(self.netG)
        load_params(self.netG, self.avg_param_G)
        fixed_imgs, _, _, _ = self.netG(
            self.fixed_noise, self.src_word_embs, self.src_sent_emb, self.src_mask,
            self.tgt_word_embs, self.tgt_sent_emb, self.tgt_mask,
        )
        filename = f"{self.save_image_dir}/{epoch_i}.png"
        save_image(fixed_imgs[-1], filename) 
        load_params(self.netG, backup_para)


    def save_model(self, cnt, model_name):
        backup_para = copy_params(self.netG)
        load_params(self.netG, self.avg_param_G)
        save_dict = {
            "cnt": cnt,
            "netG": get_state_dict(self.netG),
            "optimG": self.netG_optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "settings": self.opt,
        }
        for i in range(self.stage_num):
            netD_name = "netD_" + str(64 * 2**i)
            optimD_name = "optimD_" + str(64 * 2**i)
            save_dict[netD_name] = get_state_dict(self.netsD[i])
            save_dict[optimD_name] = self.netD_optimizers[i].state_dict()
        torch.save(save_dict, f"{self.save_model_dir}/{model_name}")
        load_params(self.netG, backup_para)

    def text_embedding(self, model, text, text_len):
        text = text.to(self.device)
        #text_len = text_len.to(self.device)
        word_embs, sent_emb = model(text, text_len)
        mask = (text == Constants.PAD)
        num_words = word_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]
        return word_embs, sent_emb, mask

    def train_epoch(self):
        self.netG_optimizer.zero_grad()
        for i in range(self.stage_num):
            self.netD_optimizers[i].zero_grad()

        epoch_netD_loss = [0.] * self.stage_num
        epoch_netG_loss = [0.] * (self.stage_num + 3) # w_loss, s_loss, KL_loss

        pbar = tqdm(self.train_loader, ascii=True, mininterval=0.5)
        for batch_cnt, train_datas in enumerate(pbar, 1):
            if self.opt.bilingual:
                imgs, src_text, src_text_len, tgt_text, tgt_text_len, class_ids, _ = train_datas                
            else:
                imgs, src_text, src_text_len, class_ids, _ = train_datas
            imgs = [img.to(self.device) for img in imgs]
            class_ids = class_ids.to(self.device)

            # --- text embedding ---
            src_word_embs, src_sent_emb, src_mask = self.text_embedding(self.src_DAMSM_RNN, src_text, src_text_len)
            src_word_embs, src_sent_emb = src_word_embs.detach(), src_sent_emb.detach()

            if self.opt.bilingual:
                tgt_word_embs, tgt_sent_emb, tgt_mask = self.text_embedding(self.tgt_DAMSM_RNN, tgt_text, tgt_text_len)
                tgt_word_embs, tgt_sent_emb = tgt_word_embs.detach(), tgt_sent_emb.detach()
            else:
                tgt_word_embs = None
                tgt_sent_emb = None
                tgt_mask = None
            
            with autocast(self.opt.use_amp):
                # --- Generate fake images ---
                self.noise.data.normal_(0, 1)
                fake_imgs, _, mu, logvar = self.netG(
                    self.noise, src_word_embs, src_sent_emb, src_mask,
                    tgt_word_embs, tgt_sent_emb, tgt_mask,
                )

                # --- Update D network ---
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

                # ---Update G network: maximize log(D(G(z))) ---
                errG_total = 0.
                self.netG_optimizer.zero_grad()
                for i in range(self.stage_num):
                    errG = generator_loss(
                        self.netsD[i], fake_imgs[i], src_sent_emb, self.real_labels,
                    )
                    errG_total += errG
                    epoch_netG_loss[i] += errG.item()

                    if i == (self.stage_num -1):
                        w_loss, s_loss = DAMSM_loss(
                            self.src_DAMSM_CNN, fake_imgs[i], src_word_embs,
                            src_sent_emb, self.match_labels, src_text_len,
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
                self.scaler.update()

            for p, avg_p in zip(self.netG.parameters(), self.avg_param_G):
                avg_p.mul_(0.999).add_(p.data, alpha=0.001)

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

        return D_logs, G_logs


    def train(self, start_epoch=1):
        self.device = get_device(self.netG)
        self.netG.train()
        for i in range(self.stage_num):
            self.netsD[i].train()

        start_all = time()
        for epoch_cnt in range(start_epoch, self.opt.max_epoch + 1):
            self.logger.info(f"\n[ Epoch {epoch_cnt} ]")

            start_span = time()
            D_logs, G_logs = self.train_epoch()
            time_span = (time() - start_span) / 60
            self.logger.info(f"{D_logs}\n{G_logs}\ntime : {time_span:.2f} min")

            if epoch_cnt % self.opt.display_freq == 0:
                self.save_fixed_images(epoch_cnt)

            if epoch_cnt % self.opt.save_freq == 0:
                self.save_model(epoch_cnt, f"epoch_{epoch_cnt}.pth")

        self.save_model(epoch_cnt, f"epoch_{epoch_cnt}.pth")
        time_all = (time() - start_all) / 3600
        self.logger.info(f"time : {time_all:.2f} h")