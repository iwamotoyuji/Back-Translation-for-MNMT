import numpy as np

class VaswaniScheduler():
    def __init__(self, optimizer, d_model, warmup_steps_num, current_steps_cnt=0):
        self.optimizer = optimizer
        self.warmup_steps_num = warmup_steps_num
        self.current_steps_cnt = current_steps_cnt
        self.init_lr = np.power(d_model, -0.5)

    def _get_lr_scale(self):
        return np.min([
            np.power(self.current_steps_cnt, -0.5),
            np.power(self.warmup_steps_num, -1.5) * self.current_steps_cnt])

    def update_lr(self):
        self.current_steps_cnt += 1
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class Scheduler():
    def __init__(self, optimizer, init_lr=0., end_lr=7e-4, warmup_steps=4000, current_steps=0):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.end_lr = end_lr
        self.warmup_steps = warmup_steps
        self.current_steps = current_steps

        self.linear_factor = (end_lr - init_lr) / warmup_steps
        self.decay_factor = end_lr * warmup_steps ** 0.5
        
        self.lr = init_lr

    def update_lr(self):
        self.current_steps += 1
        if self.current_steps < self.warmup_steps:
            self.lr = self.init_lr + self.linear_factor * self.current_steps
        else:
            self.lr = self.decay_factor * self.current_steps ** -0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr