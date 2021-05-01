import os
import sys
import argparse
import random
import numpy as np
from pathlib import Path
from socket import gethostname
pardir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(pardir))

## [Pytorch] ############################################################################
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
#########################################################################################

## [Self-Module] ########################################################################
from datasets.DAMSM_dataset import get_train_loader, get_valid_loader
from models.DAMSM import DAMSMImageEncoder, DAMSMTextEncoder
from workers.DAMSM_trainer import DAMSMTrainer
from my_utils.general_utils import mkdirs
#########################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DAMSM network")
    parser.add_argument('-b', '--batch_size', type=int, default=48)
    parser.add_argument('-m', '--max_epoch', type=int, default=120)

    parser.add_argument('-d', '--d_scale', default='small', help="small or large")
    parser.add_argument('-e', '--experiment_name', default='test')
    parser.add_argument('-g', '--gpu_ids', default='0')
    parser.add_argument('-l', '--lang', default='en')    
    parser.add_argument('-r', '--random_seed', type=int, default=42, help="None: not fixed")
    parser.add_argument('-w', '--workers', type=int, default=4, help="Number of workers in the loader")

    parser.add_argument('--bpe', default=None)
    parser.add_argument('--grad_accumulation', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true', help="Overwrite log file")
    parser.add_argument('--restart', type=int, default=None)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_memo', action='store_true', help="Speeding up the process by making notes")
    parser.add_argument('--save_freq', type=int, default=1, help="How often to save the model")
    parser.add_argument('--cudnn_benchmark', action='store_true', help="True: random not fixed")
    
    parser.add_argument('--stage_num', type=int, default=3)
    parser.add_argument('--words_limit', type=int, default=15)
    parser.add_argument('--out_feat_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.002, help="0.002 or 0.0002")
    parser.add_argument('--gamma1', type=float, default=4.0, help="1,2,5 good, 4 best, 10 100 bad")
    parser.add_argument('--gamma2', type=float, default=5.0)
    parser.add_argument('--gamma3', type=float, default=10.0, help="10 good, 1 100 bad")
    parser.add_argument('--max_norm', type=float, default=0.25)
    
    args = parser.parse_args()
    
    if args.bpe =="None":
        args.bpe = None

    args.data_path = pardir / "datasets/resources"
    args.data_path /= "{mode}_%s_data_bpe%s.pickle" % (args.d_scale, args.bpe)
    args.data_path = str(args.data_path)

    return args


def main():
    opt = parse_args()
    opt.server_name = gethostname()

    # -- CUDA setting --
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Random Seed setting --
    if opt.random_seed is None:
        opt.random_seed = random.randint(1, 10000)
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = opt.cudnn_benchmark

    # -- PATH setting --
    save_result_dir = Path(__file__).parent / "results" / opt.experiment_name
    opt.save_model_dir = str(save_result_dir / "trained_models")
    opt.save_log_path = str(save_result_dir / "train.log")
    mkdirs(opt.save_model_dir)

    # -- Prepare DataLoader --
    train_loader = get_train_loader(opt)
    valid_loader = get_valid_loader(opt, shuffle=True)
    opt.vocab_size = train_loader.dataset.vocab_size

    # -- Prepare Models --
    image_encoder = DAMSMImageEncoder(opt.out_feat_size).to(device)
    text_encoder = DAMSMTextEncoder(opt.vocab_size, out_feat_size=opt.out_feat_size).to(device)

    # --- Prepare optimizers and scaler ---
    from torch.optim import Adam
    image_optimizer = Adam(
        filter(lambda x: x.requires_grad, image_encoder.parameters()),
        lr=opt.lr, betas=(0.5, 0.999)
    )
    text_optimizer = Adam(
        text_encoder.parameters(),
        lr=opt.lr, betas=(0.5, 0.999)
    )
    scaler = GradScaler(init_scale=65536.0, enabled=opt.use_amp)

    # --- Restart setting ---
    start_cnt = 1
    if opt.restart is not None:
        start_cnt = opt.restart + 1
        #for _ in range(opt.restart):
        #    if opt.lr > opt.lr / 10.:
        #        opt.lr *= 0.98
        model_name = f"epoch_{opt.restart}.pth"
        saved_path = f"{opt.save_model_dir}/{model_name}"
        saved_dict = torch.load(saved_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(saved_dict["image_encoder"])
        text_encoder.load_state_dict(saved_dict["text_encoder"])
        image_optimizer.load_state_dict(saved_dict["image_optimizer"])
        text_optimizer.load_state_dict(saved_dict["text_optimizer"])
        scaler.load_state_dict(saved_dict["scaler"])
        print(f"[Info]Loading complete ({saved_path})")

    # -- DataParallel setting --
    gpus = [i for i in range(len(opt.gpu_ids.split(',')))]
    if len(gpus) > 1:
        image_encoder = nn.DataParallel(image_encoder, device_ids=gpus)
        text_encoder = nn.DataParallel(text_encoder, device_ids=gpus)

    # -- Prepare workers --
    trainer = DAMSMTrainer(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        train_loader=train_loader,
        image_optimizer=image_optimizer,
        text_optimizer=text_optimizer,
        scaler=scaler,
        opt=opt,
        valid_loader=valid_loader,
    )

    # -- Train --
    trainer.train(start_cnt)


if __name__ == '__main__':
    main()
