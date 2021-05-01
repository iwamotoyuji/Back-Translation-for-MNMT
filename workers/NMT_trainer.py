
## [Pytorch] ############################################################################
import torch
import torch.nn.functional as F
#########################################################################################

## [Self-Module] ########################################################################
import my_utils.Constants as Constants
from workers.base_trainer import BaseTrainer
#########################################################################################


"""
def cal_performance(pred, teacher, no_smoothing=False):
    loss = cal_loss(pred, teacher, no_smoothing)

    teacher = teacher.contiguous().view(-1)
    pred = pred.max(1)[1]
    non_pad_mask = teacher.ne(Constants.PAD)
    correct_cnt = pred.eq(teacher)
    correct_cnt = correct_cnt.masked_select(non_pad_mask).sum().item()

    return loss, correct_cnt
"""

class NMTTrainer(BaseTrainer):
    def __init__(self, model, train_loader, optimizer, scaler, scheduler,
                 opt, validator=None):
        super().__init__(model, train_loader, optimizer, scaler, scheduler,
                         opt, validator=validator)

    def cal_loss(self, src_seq, src_pos, teacher, tgt_pos):
        batch_size = src_seq.size(0)
        tgt_seq = self.make_tgt_seq(teacher, batch_size)
        src_seq = src_seq.to(self.device)
        src_pos = src_pos.to(self.device)
        tgt_seq = tgt_seq.to(self.device)
        tgt_pos = tgt_pos.to(self.device)

        pred = self.model(src_seq, src_pos, tgt_seq, tgt_pos)
        teacher = teacher.contiguous().view(-1)
        #pred = pred.cpu()
        teacher = teacher.to(self.device)
        if self.opt.no_smoothing:
            loss = F.cross_entropy(pred, teacher, ignore_index=Constants.PAD)
        else:
            """
            eps = 0.1
            class_num = pred.size(1)
            non_pad_mask = teacher.ne(Constants.PAD)
            pred = pred[non_pad_mask]
            teacher = teacher[non_pad_mask]

            one_hot = torch.zeros_like(pred).scatter(1, teacher.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (class_num - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -F.log_softmax(pred, dim=1)
            loss = (one_hot * loss).sum(dim=1).mean()
            """
            eps = 0.1
            class_num = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, teacher.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (class_num - 1)
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = teacher.ne(Constants.PAD)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).mean()

        return loss, batch_size
