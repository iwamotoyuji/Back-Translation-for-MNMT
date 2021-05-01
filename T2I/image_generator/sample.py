import os
import sys
import argparse
import random
import numpy as np
from pathlib import Path
pardir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(pardir))

## [Pytorch] ############################################################################
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
#########################################################################################

## [Self-Module] ########################################################################
from datasets.T2I_dataset import get_test_loader
from my_utils.general_utils import mkdirs
from my_utils.pytorch_utils import weights_init, set_requires_grad
from GAN.DAMSM.models.DAMSM import DAMSMTextEncoder
from models.G_net import Generator
from MMT.models.multimodal_transformer import MultimodalTransformer, check_arguments
from workers.sampler import Sampler
#########################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description='Sampling Test Images')
    parser.add_argument('-b', '--batch_size', type=int, default=12)
    parser.add_argument('-e', '--experiment_name', default='test')
    parser.add_argument('-g', '--gpu_ids', default='0')
    parser.add_argument('-Tl', '--T2I_load_epoch', type=int, default=100)
    parser.add_argument('-Ml', '--MMT_load_epoch', type=int, default=24)
    parser.add_argument('-r', '--random_seed', type=int, default=100)    
    parser.add_argument('-s', '--stage_num', type=int, default=3)
    parser.add_argument('-w', '--workers', type=int, default=4)

    parser.add_argument('--words_limit', type=int, default=20)
    args = parser.parse_args()

    return args


def prepare_models(opt, device):
    # -- Load Train settings --
    T2I_dir = Path(__file__).parent / "results" / opt.experiment_name
    T2I_model_path = T2I_dir / f"trained_models/epoch_{opt.T2I_load_epoch}.pth"
    opt.save_image_dir = T2I_dir / f"generated_images/epoch_{opt.T2I_load_epoch}/coco"
    T2I_data = torch.load(T2I_model_path, map_location=lambda storage, loc: storage)
    settings = T2I_data['settings']

    # -- Load DAMSM networks --
    DAMSM_result_dir = pardir / "GAN/DAMSM/results"
    src, tgt = settings.DAMSM.split(',')
    src_DAMSM_path = DAMSM_result_dir / src / "trained_models/best.pth"
    src_DAMSM_data = torch.load(src_DAMSM_path, map_location=lambda storage, loc: storage)
    embedding_dim = src_DAMSM_data['settings'].out_feat_size
    src_vocab_size = src_DAMSM_data['settings'].vocab_size
    src_DAMSM_RNN = DAMSMTextEncoder(src_vocab_size, out_feat_size=embedding_dim)
    src_DAMSM_RNN.load_state_dict(src_DAMSM_data['text_encoder'])
    set_requires_grad(src_DAMSM_RNN, False)
    src_DAMSM_RNN.to(device)
    src_DAMSM_RNN.eval()
    print(f"[Info] Loading complete ({src_DAMSM_path})")

    tgt_DAMSM_path = DAMSM_result_dir / tgt / "trained_models/best.pth"
    tgt_DAMSM_data = torch.load(tgt_DAMSM_path, map_location=lambda storage, loc: storage)
    tgt_vocab_size = tgt_DAMSM_data['settings'].vocab_size
    tgt_DAMSM_RNN = DAMSMTextEncoder(tgt_vocab_size, out_feat_size=embedding_dim)
    tgt_DAMSM_RNN.load_state_dict(tgt_DAMSM_data['text_encoder'])
    set_requires_grad(tgt_DAMSM_RNN, False)
    tgt_DAMSM_RNN.to(device)
    tgt_DAMSM_RNN.eval()
    print(f"[Info] Loading complete ({tgt_DAMSM_path})")

    # -- Load G network --
    netG = Generator(
        embedding_dim=embedding_dim,
        condition_dim=settings.condition_dim,
        noise_dim=settings.noise_dim,
        G_hidden=settings.G_hidden,
        stage_num=settings.stage_num,
        num_resblocks=settings.num_resblocks)
    netG.apply(weights_init)
    netG.load_state_dict(T2I_data['netG'])
    set_requires_grad(netG, False)
    netG.to(device)
    netG.eval()
    print(f"[Info] Loading complete ({T2I_model_path})")


    # --- Load MMT models ---
    MMT_model_path = f"{pardir}/MMT/results/pre_{opt.experiment_name}/trained_models/epoch_{opt.MMT_load_epoch}.pth"
    MMT_data = torch.load(MMT_model_path, map_location=lambda storage, loc: storage)
    MMT_settings = MMT_data["settings"]

    MMT = MultimodalTransformer(
        src_vocab_size=MMT_settings.src_vocab_size,
        tgt_vocab_size=MMT_settings.tgt_vocab_size,
        max_position_num=MMT_settings.max_position_num,
        d_model=MMT_settings.d_model,
        head_num=MMT_settings.head_num,
        d_k=MMT_settings.d_k,
        d_v=MMT_settings.d_v,
        d_inner=MMT_settings.d_inner,
        layer_num=MMT_settings.layer_num,
        cnn_fine_tuning=False,
        shared_embedding=MMT_settings.shared_embedding,
        share_dec_input_output_embed=MMT_settings.share_dec_input_output_embed,
    )
    MMT.load_state_dict(MMT_data['model'])
    set_requires_grad(MMT, False)
    MMT.to(device)
    MMT.eval()
    print(f"[Info] Loading complete ({MMT_model_path})")

    opt.data_path = settings.data_path
    opt.src_lang = settings.src_lang
    opt.tgt_lang = settings.tgt_lang
    opt.d_scale = settings.d_scale
    opt.bpe = settings.bpe
    opt.noise_dim = settings.noise_dim    

    return src_DAMSM_RNN, tgt_DAMSM_RNN, netG, MMT


def main():
    opt = parse_args()

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
        cudnn.benchmark = False

    # -- Prepare Models --
    src_DAMSM_RNN, tgt_DAMSM_RNN, netG, MMT = prepare_models(opt, device)

    # -- Prepare DataLoader --
    dataloader = get_test_loader(opt)
    mkdirs(opt.save_image_dir)

    sampler = Sampler(src_DAMSM_RNN, tgt_DAMSM_RNN, netG, MMT, dataloader, opt)
    sampler.sampling()


if __name__ == '__main__':
    main()
