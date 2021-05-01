from time import time
from tqdm import tqdm

## [Pytorch] ############################################################################
import torch
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast
#from torch.autograd import detect_anomaly
#########################################################################################

## [Self-Module] ########################################################################
from my_utils.losses import words_loss, sent_loss
from my_utils.general_utils import get_logger, args2string
from my_utils.pytorch_utils import get_state_dict, get_device
#########################################################################################


class DAMSMTrainer:
    def __init__(self, image_encoder, text_encoder, train_loader,
                 image_optimizer, text_optimizer, scaler, opt, valid_loader=None):
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.train_loader = train_loader
        self.image_optimizer = image_optimizer
        self.text_optimizer = text_optimizer
        self.scaler = scaler
        self.opt = opt
        self.valid_loader = valid_loader

        self.device = get_device(self.image_encoder)
        self.logger = get_logger(opt.save_log_path, overwrite=opt.overwrite)
        self.logger.info(args2string(opt))

        self.labels = torch.arange(opt.batch_size).to(self.device)
        
        self.best_epoch = 0
        self.best_score = 99999.


    def save_model(self, cnt, model_name):
        save_dict = {
            "cnt": cnt,
            "image_encoder": get_state_dict(self.image_encoder),
            "text_encoder": get_state_dict(self.text_encoder),
            "image_optimizer": self.image_optimizer.state_dict(),
            "text_optimizer": self.text_optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "settings": self.opt,
        }
        torch.save(save_dict, f"{self.opt.save_model_dir}/{model_name}")

    def cal_loss(self, image, captions, cap_lens, class_ids, key):
        image = image.to(self.device)
        captions = captions.to(self.device)
        #cap_lens = cap_lens.to(self.device)
        class_ids = class_ids.to(self.device)

        image_word_feat, image_sent_feat = self.image_encoder(image)
        text_word_feat, text_sent_feat = self.text_encoder(captions, cap_lens)

        w_loss0, w_loss1, _ = words_loss(
            image_word_feat,
            text_word_feat,
            self.labels,
            cap_lens,
            class_ids,
            self.opt
        )
        s_loss0, s_loss1 = sent_loss(
            image_sent_feat,
            text_sent_feat,
            self.labels,
            class_ids,
            self.opt
        )
        return w_loss0, w_loss1, s_loss0, s_loss1

    def evaluate(self):
        self.image_encoder.eval()
        self.text_encoder.eval()

        w_total_loss = 0.
        s_total_loss = 0.

        pbar = tqdm(self.valid_loader, ascii=True, mininterval=0.5, ncols=90)
        for batch_cnt, valid_datas in enumerate(pbar, 1):
            w_loss0, w_loss1, s_loss0, s_loss1 = self.cal_loss(*valid_datas)
            w_total_loss += (w_loss0.item() + w_loss1.item())
            s_total_loss += (s_loss0.item() + s_loss1.item())

            #if batch_cnt == 50:
            #    break
        self.image_encoder.train()
        self.text_encoder.train()

        w_cur_loss = w_total_loss / batch_cnt
        s_cur_loss = s_total_loss / batch_cnt

        return w_cur_loss, s_cur_loss

    def _train_epoch(self):
        self.image_optimizer.zero_grad()
        self.text_optimizer.zero_grad()

        w_total_loss0 = 0.
        w_total_loss1 = 0.
        s_total_loss0 = 0.
        s_total_loss1 = 0.

        pbar = tqdm(self.train_loader, ascii=True, mininterval=0.5, ncols=90)
        for batch_cnt, train_datas in enumerate(pbar, 1):
            with autocast(self.opt.use_amp):
                w_loss0, w_loss1, s_loss0, s_loss1 = self.cal_loss(*train_datas)
            w_total_loss0 += w_loss0.item()
            w_total_loss1 += w_loss1.item()
            s_total_loss0 += s_loss0.item()
            s_total_loss1 += s_loss1.item()
            loss = w_loss0 + w_loss1 + s_loss0 + s_loss1
            loss /= self.opt.grad_accumulation

            self.scaler.scale(loss).backward()

            if batch_cnt % self.opt.grad_accumulation == 0:
                self.scaler.unscale_(self.text_optimizer)
                clip_grad_norm_(self.text_encoder.parameters(), self.opt.max_norm)
                self.scaler.step(self.image_optimizer)
                self.scaler.step(self.text_optimizer)
                self.scaler.update()
                self.image_optimizer.zero_grad()
                self.text_optimizer.zero_grad()

            w_avg_loss0 = w_total_loss0 / batch_cnt
            w_avg_loss1 = w_total_loss1 / batch_cnt
            s_avg_loss0 = s_total_loss0 / batch_cnt
            s_avg_loss1 = s_total_loss1 / batch_cnt
            pbar.set_description(
                f"w_loss : {w_avg_loss0:.2f} {w_avg_loss1:.2f}, s_loss : {s_avg_loss0:.2f} {s_avg_loss1:.2f}"
            )

        return w_avg_loss0, w_avg_loss1, s_avg_loss0, s_avg_loss1


    def train(self, start_epoch=1):
        self.device = get_device(self.image_encoder)
        self.image_encoder.train()
        self.text_encoder.train()

        lr = self.opt.lr
        start_all = time()
        for epoch_cnt in range(start_epoch, self.opt.max_epoch + 1):
            self.logger.info(f"\n[ Epoch {epoch_cnt} ]")

            start_span = time()
            train_losses = self._train_epoch()
            time_span = (time() - start_span) / 60
            self.logger.info(
                f"train_w_loss : {train_losses[0]:.2f} {train_losses[1]:.2f} "
                f"train_s_loss : {train_losses[2]:.2f} {train_losses[3]:.2f} "
                f"time : {time_span:.2f}"
            )

            if epoch_cnt > self.opt.max_epoch / 3:
                if self.valid_loader is not None:
                    valid_losses = self.evaluate()
                    self.logger.info(
                        f"valid_w_loss : {valid_losses[0]:.2f} "
                        f"valid_s_loss : {valid_losses[1]:.2f} "
                        f"lr : {lr:.5f}"
                    )
                    if sum(valid_losses) <= self.best_score:
                        self.best_score = sum(valid_losses)
                        self.best_epoch = epoch_cnt
                        self.save_model(epoch_cnt, "best.pth")

                if epoch_cnt % self.opt.save_freq == 0:
                    self.save_model(epoch_cnt, f"epoch_{epoch_cnt}.pth")

            if lr > self.opt.lr / 10.:
                print("reset!!")
                lr *= 0.98
                for param_group in self.image_optimizer.param_groups:
                    param_group['lr'] = lr
                for param_group in self.text_optimizer.param_groups:
                    param_group['lr'] = lr

        time_all = (time() - start_all) / 3600
        self.logger.info(f"\nbest_epoch : {self.best_epoch}, best_score : {self.best_score}, time : {time_all:.2f} h")
