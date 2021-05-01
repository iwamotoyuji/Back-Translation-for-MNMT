import os
import sys
import argparse
import random
import numpy as np
from pathlib import Path
from socket import gethostname
pardir = Path(__file__).resolve().parent.parent
sys.path.append(str(pardir))

## [Pytorch] ############################################################################
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
#########################################################################################

## [Self-Module] ########################################################################
from datasets.NMT_dataset import get_train_val_loader
from models.transformers import Transformer, check_arguments
from workers.NMT_trainer import NMTTrainer
from my_utils.general_utils import mkdirs
from workers.lr_scheduler import Scheduler
from workers.sentence_generator import ScoreCalculator
#########################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description="Train transformer model")
    parser.add_argument('-b', '--batch_size', type=int, default=None)
    parser.add_argument('-t', '--token_size', type=int, default=None)
    parser.add_argument('--max_epoch', type=int, default=None)
    parser.add_argument('--max_step', type=int, default=None)

    parser.add_argument('-d', '--d_scale', default='small', help="small or large")
    parser.add_argument('-e', '--experiment_name', default='test')
    parser.add_argument('-g', '--gpu_ids', default='0')
    parser.add_argument('-r', '--random_seed', type=int, default=42, help="None: random not fixed")
    parser.add_argument('-w', '--workers', type=int, default=4, help="Number of workers in the loader")

    parser.add_argument('--adapt_NMT', default=None, help="(experiment_name,step_cnt) you want to adapt")
    parser.add_argument('--bpe', default=None)
    parser.add_argument('--check_point_average', type=int, default=1)
    parser.add_argument('--grad_accumulation', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true', help="Overwrite log file")
    parser.add_argument('--restart', type=int, default=None)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_fused', action='store_true')
    parser.add_argument('--use_beam', action='store_true')
    parser.add_argument('--src_lang', default='en')
    parser.add_argument('--tgt_lang', default='de')
    
    parser.add_argument('--max_position_num', type=int, default=500)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--d_inner', type=int, default=2048)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--layer_num', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--shared_embedding', action='store_true')
    parser.add_argument('--share_dec_input_output_embed', action='store_true')
    parser.add_argument('--init_weight', action='store_true')
    
    parser.add_argument('--end_lr', type=float, default=7e-4)
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--max_norm', type=float, default=5.0)
    parser.add_argument('--no_smoothing', action='store_true')
    #parser.add_argument('--early_stop', type=int, default=5)
    
    args = parser.parse_args()
    if (args.batch_size is None and args.token_size is None) or \
       (args.batch_size is not None and args.token_size is not None):
        raise ValueError("Please specify either batch_size or token_size")
    if (args.max_epoch is None and args.max_step is None) or \
       (args.max_epoch is not None and args.max_step is not None):
        raise ValueError("Please specify either max_epoch or max_step")

    if args.bpe =="None":
        args.bpe = None
    args.data_path = pardir / "datasets/resources"
    args.data_path /= "{mode}_%s_data_bpe%s.pickle" % (args.d_scale, args.bpe)
    args.data_path = str(args.data_path)

    return args


def main():
    opt = parse_args()
    opt.sever_name = gethostname()

    # --- CUDA setting ---
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Random Seed setting ---
    if opt.random_seed is None:
        opt.random_seed = random.randint(1, 10000)
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.random_seed)
        cudnn.deterministic = True

    # --- PATH setting ---
    save_result_dir = Path(__file__).parent / "results" / opt.experiment_name
    opt.save_model_dir = str(save_result_dir / "trained_models")
    opt.save_log_path = str(save_result_dir / "train.log")
    mkdirs(opt.save_model_dir)

    # --- Prepare DataLoader ---
    train_loader, valid_loader = get_train_val_loader(opt)
    opt.src_vocab_size = train_loader.dataset.src_vocab_size
    opt.tgt_vocab_size = train_loader.dataset.tgt_vocab_size

    # --- Prepare Model ---
    model = Transformer(
        src_vocab_size=opt.src_vocab_size,
        tgt_vocab_size=opt.tgt_vocab_size,
        max_position_num=opt.max_position_num,
        d_model=opt.d_model,
        head_num=opt.head_num,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_inner=opt.d_inner,
        layer_num=opt.layer_num,
        dropout=opt.dropout,
        shared_embedding=opt.shared_embedding,
        share_dec_input_output_embed=opt.share_dec_input_output_embed,
        init_weight=opt.init_weight,
        fused_layer_norm=opt.use_fused,
    ).to(device)

    # --- Prepare optimizer and scaler ---
    if opt.use_fused:
        from apex.optimizers import FusedAdam as Adam
    else:
        from torch.optim import Adam
    optimizer = Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        betas=(0.9, 0.98), eps=1e-09, weight_decay=opt.weight_decay)
    scaler = GradScaler(init_scale=65536.0, enabled=opt.use_amp)

    # --- Restart setting ---
    start_cnt = 1
    steps_cnt = 0
    if opt.adapt_NMT is not None:
        ex_name, step_cnt = opt.adapt_NMT.split(',')
        saved_path = f"{Path(__file__).parent}/results/{ex_name}/trained_models/step_{step_cnt}.pth"
        saved_dict = torch.load(saved_path, map_location=lambda storage, loc: storage)
        check_arguments(saved_dict["settings"], opt)
        model.load_state_dict(saved_dict["model"])
        print(f"[Info]Loading complete ({saved_path})")

    if opt.restart is not None:
        start_cnt = opt.restart + 1
        if opt.restart < 500:
            model_name = f"epoch_{opt.restart}.pth"
        else:
            model_name = f"step_{opt.restart}.pth"
        saved_path = f"{opt.save_model_dir}/{model_name}"
        saved_dict = torch.load(saved_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(saved_dict["model"])
        optimizer.load_state_dict(saved_dict["optimizer"])
        scaler.load_state_dict(saved_dict["scaler"])
        steps_cnt = saved_dict["steps_cnt"]
        print(f"[Info]Loading complete ({saved_path})")
        
    scheduler = Scheduler(
        optimizer=optimizer,
        init_lr=0., end_lr=opt.end_lr,
        warmup_steps=opt.warmup_steps, current_steps=steps_cnt,
    )

    # --- DataParallel setting ---
    gpus = [i for i in range(len(opt.gpu_ids.split(',')))]
    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus)

    # --- Prepare trainer and validator ---
    validator = ScoreCalculator(
        model=model,
        data_loader=valid_loader,
        references=valid_loader.dataset.tgt_insts,
        bpe=opt.bpe,
        cp_avg_num=opt.check_point_average,
    )
    trainer = NMTTrainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        opt=opt,
        validator=validator,
    )

    # --- Train ---
    if opt.max_epoch is not None:
        trainer.train_by_epoch(start_cnt)  
    else:      
        trainer.train_by_step(start_cnt)


if __name__ == '__main__':
    main()
