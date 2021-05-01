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
from datasets.T2I_dataset import get_train_loader
from models.netG import Generator
from models.netD import D_net64, D_net128, D_net256
from models.DAMSM import DAMSMImageEncoder, DAMSMTextEncoder
from my_utils.general_utils import mkdirs
from my_utils.pytorch_utils import weights_init, set_requires_grad
from workers.BiAttnGAN_trainer import Trainer
#########################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain a BiAttnGAN network")
    parser.add_argument('-b', '--batch_size', type=int, default=20)
    parser.add_argument('-m', '--max_epoch', type=int, default=120)

    parser.add_argument('-d', '--d_scale', default='small', help="small or large")
    parser.add_argument('-e', '--experiment_name', default='test')
    parser.add_argument('-g', '--gpu_ids', default='0')
    parser.add_argument('-r', '--random_seed', type=int, default=42, help="None: not fixed")
    parser.add_argument('-w', '--workers', type=int, default=4, help="Number of workers in the loader")

    parser.add_argument('--DAMSM', default='en_small,de_small', help="pretrained DAMSM (for src_lang,for tgt_lang)")
    parser.add_argument('--bpe', default=None)
    parser.add_argument('--bilingual', action='store_true')
    parser.add_argument('--grad_accumulation', type=int, default=1)
    parser.add_argument('--tgt_rnn_fine_tuning', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help="Overwrite log file")
    parser.add_argument('--restart', type=int, default=None)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_memo', action='store_true', help="Speeding up the process by making notes")
    parser.add_argument('--save_freq', type=int, default=5, help="How often to save the model")
    parser.add_argument('--display_freq', type=int, default=10)    
    parser.add_argument('--cudnn_benchmark', action='store_true', help="True: random not fixed")
    parser.add_argument('--src_lang', default='en')
    parser.add_argument('--tgt_lang', default='de')

    parser.add_argument('--stage_num', type=int, default=3, help="1:64, 2:128, 3:256")
    parser.add_argument('--words_limit', type=int, default=12)
    parser.add_argument('--G_hidden', type=int, default=48)
    parser.add_argument('--D_hidden', type=int, default=96)
    parser.add_argument('--condition_dim', type=int, default=100)
    parser.add_argument('--noise_dim', type=int, default=100)
    parser.add_argument('--num_resblocks', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--gamma1', type=float, default=4.0, help="1,2,5 good, 4 best, 10 100 bad")
    parser.add_argument('--gamma2', type=float, default=5.0)
    parser.add_argument('--gamma3', type=float, default=10.0, help="10 good, 1 100 bad")
    parser.add_argument('--lambda1', type=float, default=50.0)
    
    args = parser.parse_args()

    if args.bpe =="None":
        args.bpe = None

    args.data_path = pardir / "datasets/resources"
    args.data_path /= "{mode}_%s_data_bpe%s.pickle" % (args.d_scale, args.bpe)
    args.data_path = str(args.data_path)

    return args


def prepare_models(opt, device):
    # -- Load DAMSM models --
    DAMSM_result_dir = pardir / "T2I/DAMSM/results"
    src_dir, tgt_dir = opt.DAMSM.split(',')
    src_DAMSM_path = DAMSM_result_dir / src_dir / "trained_models/epoch_100.pth"
    src_DAMSM_data = torch.load(src_DAMSM_path, map_location=lambda storage, loc: storage)
    embedding_dim = src_DAMSM_data['settings'].out_feat_size
    src_vocab_size = src_DAMSM_data['settings'].vocab_size
    if src_vocab_size != opt.src_vocab_size:
        raise ValueError("[Error] The dictionary size is different from the dictionary size in DAMSM training.")

    src_DAMSM_CNN = DAMSMImageEncoder(embedding_dim)
    src_DAMSM_CNN.load_state_dict(src_DAMSM_data['image_encoder'])
    set_requires_grad(src_DAMSM_CNN, False)
    src_DAMSM_CNN.eval()
    src_DAMSM_CNN.to(device)
    src_DAMSM_RNN = DAMSMTextEncoder(src_vocab_size, out_feat_size=embedding_dim)
    src_DAMSM_RNN.load_state_dict(src_DAMSM_data['text_encoder'])
    set_requires_grad(src_DAMSM_RNN, False)
    src_DAMSM_RNN.eval()
    src_DAMSM_RNN.to(device)
    print(f"[Info] Loading complete ({src_DAMSM_path})")

    if opt.bilingual:
        tgt_DAMSM_path = DAMSM_result_dir / tgt_dir / "trained_models/epoch_100.pth"
        tgt_DAMSM_data = torch.load(tgt_DAMSM_path, map_location=lambda storage, loc: storage)
        embedding_dim = tgt_DAMSM_data['settings'].out_feat_size
        tgt_vocab_size = tgt_DAMSM_data['settings'].vocab_size
        if tgt_vocab_size != opt.tgt_vocab_size:
            raise ValueError("[Error] The dictionary size is different from the dictionary size in DAMSM training.")

        tgt_DAMSM_RNN = DAMSMTextEncoder(tgt_vocab_size, out_feat_size=embedding_dim)
        tgt_DAMSM_RNN.load_state_dict(tgt_DAMSM_data['text_encoder'])
        set_requires_grad(tgt_DAMSM_RNN, opt.tgt_rnn_fine_tuning)
        if opt.tgt_rnn_fine_tuning:
            tgt_DAMSM_RNN.train()
        else:
            tgt_DAMSM_RNN.eval()
        tgt_DAMSM_RNN.to(device)
        print(f"[Info] Loading complete ({tgt_DAMSM_path})")
    else:
        tgt_DAMSM_RNN = None

    # -- Prepare G and D models --
    netG = Generator(
        embedding_dim=embedding_dim,
        condition_dim=opt.condition_dim,
        noise_dim=opt.noise_dim,
        G_hidden=opt.G_hidden,
        stage_num=opt.stage_num,
        num_resblocks=opt.num_resblocks,
        bilingual=opt.bilingual,
    )
    netG.apply(weights_init)
    netG.to(device)

    netsD = []
    if opt.stage_num > 0:
        netsD.append(D_net64(embedding_dim, opt.D_hidden))
    if opt.stage_num > 1:
        netsD.append(D_net128(embedding_dim, opt.D_hidden))
    if opt.stage_num > 2:
        netsD.append(D_net256(embedding_dim, opt.D_hidden))
    for i in range(opt.stage_num):
        netsD[i].apply(weights_init)
        netsD[i].to(device)

    return src_DAMSM_CNN, src_DAMSM_RNN, tgt_DAMSM_RNN, netG, netsD


def main():
    opt = parse_args()
    opt.server_name = gethostname()

    # -- CUDA setting --
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
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
    opt.save_image_dir = str(save_result_dir / "generated_images")
    opt.save_model_dir = str(save_result_dir / "trained_models")
    opt.save_log_path = str(save_result_dir / "train.log")
    mkdirs(opt.save_image_dir, opt.save_model_dir)

    # -- Prepare DataLoader --
    train_loader = get_train_loader(opt)
    opt.src_vocab_size = train_loader.dataset.src_vocab_size
    opt.tgt_vocab_size = train_loader.dataset.tgt_vocab_size

    # -- Prepare Models --
    src_DAMSM_CNN, src_DAMSM_RNN, tgt_DAMSM_RNN, netG, netsD = prepare_models(opt, device)

    # --- Prepare optimizers and scaler ---
    from torch.optim import Adam   
    netG_optimizer = Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    netD_optimizers = []
    for i in range(opt.stage_num):
        optimizer = Adam(netsD[i].parameters(), lr=opt.lr, betas=(0.5, 0.999))
        netD_optimizers.append(optimizer)
    scaler = GradScaler(init_scale=65536.0, enabled=opt.use_amp)

    # -- Restart setting --
    start_cnt = 1
    if opt.restart is not None:
        start_cnt = opt.restart + 1
        saved_path = f"{opt.save_model_dir}/epoch_{opt.restart}.pth"
        saved_dict = torch.load(saved_path, map_location=lambda storage, loc: storage)
        netG.load_state_dict(saved_dict["netG"])
        netG_optimizer.load_state_dict(saved_dict["optimG"])
        for i in range(opt.stage_num):
            model_name = "netD_" + str(64 * 2**i)
            optim_name = "optimD_" + str(64 * 2**i)
            netsD[i].load_state_dict(saved_dict[model_name])
            netD_optimizers[i].load_state_dict(saved_dict[optim_name])
        scaler.load_state_dict(saved_dict["scaler"])
        print(f"[Info]Loading complete ({saved_path})")

    # -- DataParallel setting --
    gpus = [i for i in range(len(opt.gpu_ids.split(',')))]
    if len(gpus) > 1:
        netG = nn.DataParallel(netG, device_ids=gpus)

    # -- Train --
    trainer = Trainer(
        src_DAMSM_CNN=src_DAMSM_CNN,
        src_DAMSM_RNN=src_DAMSM_RNN,
        tgt_DAMSM_RNN=tgt_DAMSM_RNN,
        netG=netG,
        netsD=netsD,
        netG_optimizer=netG_optimizer,
        netD_optimizers=netD_optimizers,
        train_loader=train_loader,
        scaler=scaler,
        opt=opt,
    )
    trainer.train(start_cnt)


if __name__ == "__main__":
    main()