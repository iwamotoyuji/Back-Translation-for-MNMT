from copy import deepcopy

## [Pytorch] ############################################################################
import torch.nn as nn
#########################################################################################


def set_lr(optimizer, lr):
    for parm_group in optimizer.param_groups:
        parm_group['lr'] = lr

def get_state_dict(model):
    _model = model.module if 'DataParallel' in str(type(model)) else model
    return deepcopy(_model.state_dict())

def get_device(model):
    _model = model.module if 'DataParallel' in str(type(model)) else model
    return next(_model.parameters()).device

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

def load_params(model, new_params):
    for p, new_p in zip(model.parameters(), new_params):
        p.data.copy_(new_p)

def copy_params(model, to_cpu=False):
    if to_cpu:
        params = deepcopy(list(p.data.cpu() for p in model.parameters()))
    else:
        params = deepcopy(list(p.data for p in model.parameters()))
    return params

def set_requires_grad(model, requires=False):
    for p in model.parameters():
        p.requires_grad = requires
