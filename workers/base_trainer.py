from time import time
from tqdm import tqdm

## [Pytorch] ############################################################################
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast
#########################################################################################

## [Self-Module] ########################################################################
import my_utils.Constants as Constants
from my_utils.general_utils import get_logger, args2string
from my_utils.pytorch_utils import get_state_dict, get_device
#########################################################################################


class BaseTrainer:
    def __init__(self, model, train_loader, optimizer, scaler, scheduler,
                 opt, validator=None):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.opt = opt
        self.validator = validator

        self.device = get_device(self.model)
        self.logger = get_logger(opt.save_log_path, overwrite=opt.overwrite)
        self.logger.info(args2string(opt))

        self.best_bleu_score = 0.
        self.best_cnt = 0

    def cal_loss(self):
        raise NotImplementedError()


    def save_model(self, cnt, state_dict, model_name):
        save_dict = {
            "cnt": cnt,
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "steps_cnt": self.scheduler.current_steps,
            "settings": self.opt,
        }
        torch.save(save_dict, f"{self.opt.save_model_dir}/{model_name}")


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
        

    def validation(self, cnt):
        bleu_score, state_dict = self.validator.get_bleu(
            use_beam=self.opt.use_beam, return_sentences=False
        )
        self.logger.info(f"bleu : {bleu_score:.4f}")
        self.model.train()

        if bleu_score >= self.best_bleu_score:
            self.best_bleu_score = bleu_score
            self.best_cnt = cnt
            self.save_model(cnt, state_dict, "best.pth")
        
        return state_dict

        
    def _train_epoch(self):
        self.optimizer.zero_grad()
        epoch_loss = 0.

        pbar = tqdm(self.train_loader, ncols=90, mininterval=0.5, ascii=True)
        for batch_cnt, train_datas in enumerate(pbar, 1):
            with autocast(self.opt.use_amp):
                loss, _ = self.cal_loss(*train_datas)
            epoch_loss += loss.item()
            loss /= self.opt.grad_accumulation

            self.scaler.scale(loss).backward()

            if batch_cnt % self.opt.grad_accumulation == 0:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.opt.max_norm)
                self.scheduler.update_lr()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            avg_epoch_loss = epoch_loss / batch_cnt
            pbar.set_description(f"word_loss : {avg_epoch_loss:.2f}")
            #torch.cuda.empty_cache()

        return avg_epoch_loss


    def train_by_epoch(self, start_epoch=1):
        self.device = get_device(self.model)
        self.model.train()

        start_all = time()
        for epoch_cnt in range(start_epoch, self.opt.max_epoch + 1):
            self.logger.info(f"\n[ Epoch {epoch_cnt} ]")

            start_span = time()
            avg_epoch_loss = self._train_epoch()
            time_span = (time() - start_span) / 60
            self.logger.info(f"word_loss : {avg_epoch_loss:.2f}, time : {time_span:.2f} min")

            if self.validator is not None:
                state_dict = self.validation(epoch_cnt)
            else:
                state_dict = get_state_dict(self.model)

            if epoch_cnt > self.opt.max_epoch / 3:
                self.save_model(epoch_cnt, state_dict, f"epoch_{epoch_cnt}.pth")

        time_all = (time() - start_all) / 3600
        self.logger.info(f"\nbest_epoch : {self.best_cnt}, best_score : {self.best_bleu_score}, time : {time_all:.2f} h")


    def train_by_step(self, start_step=1):
        self.device = get_device(self.model)

        self.model.train()
        self.optimizer.zero_grad()
        data_iter = iter(self.train_loader)
        checkpoint_interval = 1500
        checkpoint_cnt = 1
        checkpoint_loss = 0.

        step_cnt = start_step
        iter_num = (self.opt.max_step - start_step + 1) * self.opt.grad_accumulation
        start_all = time()
        start_span = time()
        pbar = tqdm(range(iter_num), ncols=90, mininterval=0.5, ascii=True)
        for _ in pbar:
            try:
                train_datas = data_iter.next()
            except StopIteration:
                data_iter = iter(self.train_loader)
                train_datas = data_iter.next()

            with autocast(self.opt.use_amp):
                loss, batch_size = self.cal_loss(*train_datas)
            checkpoint_loss += loss.item()
            loss /= self.opt.grad_accumulation

            self.scaler.scale(loss).backward()

            if checkpoint_cnt % self.opt.grad_accumulation == 0:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.opt.max_norm)
                self.scheduler.update_lr()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                avg_checkpoint_loss = checkpoint_loss / checkpoint_cnt
                pbar.set_description(
                    f"\r[{step_cnt}/{self.opt.max_step}] " \
                    f"word_loss : {avg_checkpoint_loss:.2f}, batch_size : {batch_size:<5}"
                )                
            
                if step_cnt % checkpoint_interval == 0:
                    print()
                    time_span = (time() - start_span) / 60
                    self.logger.info(f"\n[ Step {step_cnt} ]")
                    self.logger.info(f"word_loss : {avg_checkpoint_loss:.2f}, time : {time_span:.2f} min")

                    if self.validator is not None:
                        state_dict = self.validation(step_cnt)
                    else:
                        state_dict = get_state_dict(self.model)

                    if step_cnt > self.opt.max_step / 3:
                        self.save_model(step_cnt, state_dict, f"step_{step_cnt}.pth")

                    start_span = time()
                    checkpoint_cnt = 0
                    checkpoint_loss = 0.
                    print()

                step_cnt += 1
            checkpoint_cnt += 1


        step_cnt = step_cnt - 1
        if step_cnt % checkpoint_interval != 0:
            self.logger.info(f"\n[ Step {step_cnt} ]")
            self.logger.info(f"word_loss : {avg_checkpoint_loss:.2f}")
            if self.validator is not None:
                state_dict = self.validation(step_cnt)
            else:
                state_dict = get_state_dict(self.model)
            self.save_model(step_cnt, state_dict, f"step_{step_cnt}.pth")
            
        time_all = (time() - start_all) / 3600
        self.logger.info(f"\nbest_step : {self.best_cnt}, best_score : {self.best_bleu_score}, time : {time_all:.2f} h")