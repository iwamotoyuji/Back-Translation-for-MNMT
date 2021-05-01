import os
import argparse
import random
import numpy as np
from pathlib import Path
from socket import gethostname
pardir = Path(__file__).parent

## [Pytorch] ############################################################################
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
#########################################################################################

## [Self-Module] ########################################################################
from datasets.BT_MNMT_dataset import get_train_val_loader
from datasets.BT_T2I_dataset import get_train_loader
from models.DAMSM import DAMSMImageEncoder, DAMSMTextEncoder
from models.netG import Generator
from models.netD import D_net64, D_net128, D_net256
from models.transformers import MultimodalTransformer, check_arguments
from my_utils.general_utils import mkdirs
from my_utils.pytorch_utils import weights_init, set_requires_grad
from workers.BT_trainer import Trainer
from workers.lr_scheduler import Scheduler
from workers.sentence_generator import ScoreCalculator
#########################################################################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-Mb', '--MNMT_batch_size', type=int, default=128)
    parser.add_argument('-Tb', '--T2I_batch_size', type=int, default=32)
    parser.add_argument('-m', '--max_epoch', type=int, default=15)

    parser.add_argument('-d', '--d_scale', default='small', help="small or large")
    parser.add_argument('-e', '--experiment_name', default='test')
    parser.add_argument('-g', '--gpu_ids', default='0')
    parser.add_argument('-r', '--random_seed', type=int, default=42, help="None is not fixed")
    parser.add_argument('-w', '--workers', type=int, default=4, help="Number of workers in the loader")

    parser.add_argument('--MNMT', default='pre_small_bpe7000_base,24', help="(experiment_name,epoch) pretrained MNMT")
    parser.add_argument('--T2I', default='small_bpe7000_base,100', help="(experiment_name,epoch) pretrained T2I")

    parser.add_argument('--adapt_init_MNMT', action='store_true')
    parser.add_argument('--adapt_prop_MNMT', action='store_true')
    parser.add_argument('--bpe', default=None)
    parser.add_argument('--check_point_average', type=int, default=1)
    parser.add_argument('--MNMT_grad_accumulation', type=int, default=1)
    #arser.add_argument('--T2I_grad_accumulation', type=int, default=1)
    parser.add_argument('--T2I_per_MNMT', type=int, default=1, help="T2I's epochs per MNMT's epochs")
    parser.add_argument('--tgt_rnn_fine_tuning', action='store_true')
    parser.add_argument('--cnn_fine_tuning', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help="Overwrite log file")
    parser.add_argument('--restart', type=int, default=None)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_beam', action='store_true')
    parser.add_argument('--use_memo', action='store_true', help="Speeding up the process by making notes")
    parser.add_argument('--cudnn_benchmark', action='store_true', help="True: random not fixed")
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--src_lang', default='en')
    parser.add_argument('--tgt_lang', default='de')

    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--stage_num', type=int, default=3, help="1:64, 2:128, 3:256")
    parser.add_argument('--train_words_limit', type=int, default=12)
    parser.add_argument('--eval_words_limit', type=int, default=20)
    parser.add_argument('--DAMSM_lr', type=float, default=0.0002)
    parser.add_argument('--T2I_lr', type=float, default=0.0002)
    parser.add_argument('--end_lr', type=float, default=7e-4)
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--weight_decay', type=float, default=1e-06)
    parser.add_argument('--max_norm', type=float, default=5.0)
    parser.add_argument('--no_smoothing', action='store_true')
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


def prepare_MNMT_model(opt, device):
    # --- Load MNMT model setting ---
    MNMT_dir, MNMT_epoch = opt.MNMT.split(',')
    if opt.adapt_prop_MNMT:
        MNMT_path = pardir / "results" / MNMT_dir / f"trained_models/epoch_{MNMT_epoch}.pth"
    else:
        MNMT_path = pardir / "MNMT/results" / MNMT_dir / f"trained_models/epoch_{MNMT_epoch}.pth"

    MNMT_data = torch.load(MNMT_path, map_location=lambda storage, loc: storage)
    if opt.adapt_prop_MNMT:
        pre_MNMT_dir, pre_MNMT_epoch = MNMT_data["settings"].MNMT.split(',')
        pre_MNMT_path = pardir / "MNMT/results" / pre_MNMT_dir / f"trained_models/epoch_{pre_MNMT_epoch}.pth"
        MNMT_settings = torch.load(pre_MNMT_path, map_location=lambda storage, loc: storage)["settings"]
    else:
        MNMT_settings = MNMT_data["settings"]
    MNMT_src_vocab_size = MNMT_settings.src_vocab_size
    MNMT_tgt_vocab_size = MNMT_settings.tgt_vocab_size
    if MNMT_src_vocab_size != opt.MNMT_src_vocab_size:
        raise ValueError(f"[Error] Not match dict size. now:{opt.MNMT_src_vocab_size} pre:{MNMT_src_vocab_size}")
    if MNMT_tgt_vocab_size != opt.MNMT_tgt_vocab_size:
        raise ValueError(f"[Error] Not match dict size. now:{opt.MNMT_tgt_vocab_size} pre:{MNMT_tgt_vocab_size}")
    
    # --- Load MNMT models ---
    MNMT = MultimodalTransformer(
        src_vocab_size=MNMT_src_vocab_size,
        tgt_vocab_size=MNMT_tgt_vocab_size,
        max_position_num=MNMT_settings.max_position_num,
        d_model=MNMT_settings.d_model,
        head_num=MNMT_settings.head_num,
        d_k=MNMT_settings.d_k,
        d_v=MNMT_settings.d_v,
        d_inner=MNMT_settings.d_inner,
        layer_num=MNMT_settings.layer_num,
        dropout=opt.dropout,
        cnn_fine_tuning=opt.cnn_fine_tuning,
        shared_embedding=MNMT_settings.shared_embedding,
        share_dec_input_output_embed=MNMT_settings.share_dec_input_output_embed,
        init_weight=False,
        fused_layer_norm=MNMT_settings.use_fused,
    )
    if opt.adapt_prop_MNMT:
        model_data = MNMT_data['models']['MNMT']
    else:
        model_data = MNMT_data['model']
    MNMT.load_state_dict(model_data)
    MNMT.to(device)
    print(f"[Info] Loading complete ({MNMT_path})")

    return MNMT, MNMT_settings.d_model


def prepare_T2I_models(opt, device):    
    # --- Load T2I model setting ---
    T2I_dir, T2I_epoch = opt.T2I.split(',')
    T2I_path = pardir / "T2I/image_generator/results" / T2I_dir / f"trained_models/epoch_{T2I_epoch}.pth"
    T2I_data = torch.load(T2I_path, map_location=lambda storage, loc: storage)
    T2I_settings = T2I_data["settings"]
    T2I_src_vocab_size = T2I_settings.src_vocab_size
    T2I_tgt_vocab_size = T2I_settings.tgt_vocab_size
    if T2I_src_vocab_size != opt.T2I_src_vocab_size:
        raise ValueError(f"[Error] Not match dict size. now:{opt.T2I_src_vocab_size} pre:{T2I_src_vocab_size}")
    if T2I_tgt_vocab_size != opt.T2I_tgt_vocab_size:
        raise ValueError(f"[Error] Not match dict size. now:{opt.T2I_tgt_vocab_size} pre:{T2I_tgt_vocab_size}")

    # --- Load DAMSM models ---
    DAMSM_result_dir = pardir / "T2I/DAMSM/results"
    src_dir, tgt_dir = T2I_settings.DAMSM.split(',')

    src_DAMSM_path = DAMSM_result_dir / src_dir / "trained_models/best.pth"
    src_DAMSM_data = torch.load(src_DAMSM_path, map_location=lambda storage, loc: storage)
    embedding_dim = src_DAMSM_data["settings"].out_feat_size
    src_DAMSM_CNN = DAMSMImageEncoder(embedding_dim)
    src_DAMSM_CNN.load_state_dict(src_DAMSM_data["image_encoder"])
    set_requires_grad(src_DAMSM_CNN, False)
    src_DAMSM_CNN.eval()
    src_DAMSM_CNN.to(device)
    src_DAMSM_RNN = DAMSMTextEncoder(T2I_src_vocab_size, out_feat_size=embedding_dim)
    src_DAMSM_RNN.load_state_dict(src_DAMSM_data["text_encoder"])
    set_requires_grad(src_DAMSM_RNN, False)
    src_DAMSM_RNN.eval()
    src_DAMSM_RNN.to(device)
    print(f"[Info] Loading complete ({src_DAMSM_path})")

    if T2I_settings.bilingual:
        tgt_DAMSM_path = DAMSM_result_dir / tgt_dir / "trained_models/best.pth"
        tgt_DAMSM_data = torch.load(tgt_DAMSM_path, map_location=lambda storage, loc: storage)
        embedding_dim = tgt_DAMSM_data["settings"].out_feat_size
        tgt_DAMSM_RNN = DAMSMTextEncoder(T2I_tgt_vocab_size, out_feat_size=embedding_dim)
        tgt_DAMSM_RNN.load_state_dict(tgt_DAMSM_data["text_encoder"])
        tgt_DAMSM_RNN.to(device)
        print(f"[Info] Loading complete ({tgt_DAMSM_path})")
    else:
        tgt_DAMSM_RNN = None
        opt.tgt_rnn_fine_tuning = False

    # --- Load G and D models ---
    netG = Generator(
        embedding_dim=embedding_dim,
        condition_dim=T2I_settings.condition_dim,
        noise_dim=T2I_settings.noise_dim,
        G_hidden=T2I_settings.G_hidden,
        stage_num=T2I_settings.stage_num,
        num_resblocks=T2I_settings.num_resblocks,
        bilingual=T2I_settings.bilingual,
    )
    netG.apply(weights_init)
    netG.load_state_dict(T2I_data["netG"])
    netG.to(device)

    netsD = []
    if opt.stage_num > 0:
        netsD.append(D_net64(embedding_dim, T2I_settings.D_hidden))
    if opt.stage_num > 1:
        netsD.append(D_net128(embedding_dim, T2I_settings.D_hidden))
    if opt.stage_num > 2:
        netsD.append(D_net256(embedding_dim, T2I_settings.D_hidden))
    for i in range(opt.stage_num):
        netsD[i].apply(weights_init)
        model_name = "netD_" + str(64 * 2**i)
        netsD[i].load_state_dict(T2I_data[model_name])
        netsD[i].to(device)
    print(f"[Info] Loading complete ({T2I_path})")

    opt.noise_dim = T2I_settings.noise_dim
    opt.bilingual = T2I_settings.bilingual

    return src_DAMSM_CNN, src_DAMSM_RNN, tgt_DAMSM_RNN, netG, netsD


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
    opt.save_image_dir = str(save_result_dir / "generated_images")
    opt.save_model_dir = str(save_result_dir / "trained_models")
    opt.save_log_path = str(save_result_dir / "train.log")
    mkdirs(opt.save_image_dir, opt.save_model_dir)

    # -- Prepare DataLoader --
    MNMT_loader, valid_loader = get_train_val_loader(opt)
    T2I_loader = get_train_loader(opt)
    opt.T2I_src_vocab_size = T2I_loader.dataset.src_vocab_size
    opt.T2I_tgt_vocab_size = T2I_loader.dataset.tgt_vocab_size
    opt.MNMT_src_vocab_size = MNMT_loader.dataset.src_vocab_size
    opt.MNMT_tgt_vocab_size = MNMT_loader.dataset.tgt_vocab_size

    # -- Prepare Models --
    MNMT, d_model = prepare_MNMT_model(opt, device)
    src_DAMSM_CNN, src_DAMSM_RNN, tgt_DAMSM_RNN, netG, netsD = prepare_T2I_models(opt, device)

    # -- Prepare optimizer and scaler --
    from torch.optim import Adam
    MNMT_optimizer = Adam(filter(lambda x: x.requires_grad, MNMT.parameters()),
                         betas=(0.9, 0.98), eps=1e-09, weight_decay=opt.weight_decay)    
    netG_optimizer = Adam(netG.parameters(), lr=opt.T2I_lr, betas=(0.5, 0.999))
    netD_optimizers = []
    for i in range(opt.stage_num):
        optimizer = Adam(netsD[i].parameters(), lr=opt.T2I_lr, betas=(0.5, 0.999))
        netD_optimizers.append(optimizer)
    if opt.tgt_rnn_fine_tuning:
        DAMSM_optimizer = Adam(tgt_DAMSM_RNN.parameters(), lr=opt.DAMSM_lr, betas=(0.5, 0.999))
    else:
        DAMSM_optimizer = None
    scaler = GradScaler(init_scale=65536.0, enabled=opt.use_amp)
        
    # -- Restart setting --
    start_cnt = 1
    steps_cnt = 0    
    if opt.restart:
        start_cnt = opt.restart + 1
        saved_path = opt.save_model_dir / f"epoch_{opt.restart}.pth"
        saved_dict = torch.load(saved_path, map_location=lambda storage, loc: storage)
        MNMT.load_state_dict(saved_dict["models"]["MNMT"])
        MNMT_optimizer.load_state_dict(saved_dict["optims"]["MNMT"])
        netG.load_state_dict(saved_dict["models"]["netG"])
        netG_optimizer.load_state_dict(saved_dict["optims"]["netG"])
        for i in range(opt.state_num):
            model_name = "netD_" + str(64 * 2**i)
            netsD[i].load_state_dict(saved_dict["models"][model_name])
            netD_optimizers[i].load_state_dict(saved_dict["optims"][model_name])
        scaler.load_state_dict(saved_dict["scaler"])
        steps_cnt = saved_dict["steps_cnt"]
        print(f"[Info]Loading complete ({saved_path})")
    
    scheduler = Scheduler(
        optimizer=MNMT_optimizer,
        init_lr=0., end_lr=opt.end_lr,
        warmup_steps=opt.warmup_steps, current_steps=steps_cnt,
    )

    # -- DataParallel setting --
    gpus = [i for i in range(len(opt.gpu_ids.split(',')))]
    if len(gpus) > 1:
        MNMT = nn.DataParallel(MNMT, device_ids=gpus)        
        netG = nn.DataParallel(netG, device_ids=gpus)
    
    # -- Train --
    if valid_loader is not None:
        validator = ScoreCalculator(
            model=MNMT,
            data_loader=valid_loader,
            references=valid_loader.dataset.MNMT_tgt_insts,
            bpe=opt.bpe,
            cp_avg_num=opt.check_point_average,
        )
    else:
        validator = None

    trainer = Trainer(
        MNMT=MNMT,        
        src_DAMSM_CNN=src_DAMSM_CNN,
        src_DAMSM_RNN=src_DAMSM_RNN,
        tgt_DAMSM_RNN=tgt_DAMSM_RNN,
        netG=netG,
        netsD=netsD,
        MNMT_optimizer=MNMT_optimizer,       
        netG_optimizer=netG_optimizer,
        netD_optimizers=netD_optimizers,
        DAMSM_optimizer=DAMSM_optimizer,
        MNMT_loader=MNMT_loader,               
        T2I_loader=T2I_loader,
        scaler=scaler,
        scheduler=scheduler,
        opt=opt,
        validator=validator,
    )
    trainer.train(start_cnt)


if __name__ == "__main__":
    main()
