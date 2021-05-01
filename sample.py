import os
import argparse
import random
import numpy as np
from pathlib import Path
pardir = Path(__file__).parent

## [Pytorch] ############################################################################
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
#########################################################################################

## [Self-Module] ########################################################################
from datasets.T2I_dataset import get_eval_loader
from my_utils.general_utils import mkdirs
from my_utils.pytorch_utils import weights_init, set_requires_grad
from GAN.DAMSM.models.DAMSM import DAMSMTextEncoder
from GAN.image_generator.models.G_net import Generator
from MMT.models.multimodal_transformer import MultimodalTransformer
from workers.sampler import Sampler
#########################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description="Sampling Test Images")
    parser.add_argument('-b', '--batch_size', type=int, default=12)
    parser.add_argument('-e', '--experiment_name', default='test')
    parser.add_argument('-g', '--gpu_ids', default='0')
    parser.add_argument('-l', '--load_epoch', type=int, default=0)
    parser.add_argument('-r', '--random_seed', type=int, default=100)
    parser.add_argument('-s', '--stage_num', type=int, default=3)
    parser.add_argument('-w', '--workers', type=int, default=4)

    parser.add_argument('-i', '--image_id', type=str, default=None)
    parser.add_argument('--mode', default="eval")

    parser.add_argument('--noise_num', type=int, default=5)
    parser.add_argument('--eval_words_limit', type=int, default=20)
    args = parser.parse_args()

    return args


def prepare_models(opt, device):
    # --- Load model setting ---
    trained_dir = pardir / "results" / opt.experiment_name
    if opt.load_epoch != 0:
        trained_model_path = trained_dir / f"trained_models/epoch_{opt.load_epoch}.pth"
        opt.save_image_dir = trained_dir / f"generated_images/epoch_{opt.load_epoch}/coco"
    else:
        trained_model_path = trained_dir / "trained_models/best.pth"
        opt.save_image_dir = trained_dir / "generated_images/best/coco"

    trained_data = torch.load(trained_model_path, map_location=lambda storage, loc: storage)
    trained_setting = trained_data['settings']

    # --- Load T2I model settings ---
    T2I_dir, T2I_epoch = trained_setting.T2I.split(',')
    T2I_path = pardir / "GAN/image_generator/results" / T2I_dir / f"trained_models/epoch_{T2I_epoch}.pth"
    T2I_data = torch.load(T2I_path, map_location=lambda storage, loc: storage)
    T2I_settings = T2I_data['settings']

    # --- Load DAMSM models ---
    DAMSM_result_dir = pardir / "GAN/DAMSM/results"
    src_dir, tgt_dir = T2I_settings.DAMSM.split(',')
    src_DAMSM_path = DAMSM_result_dir / src_dir / "trained_models/best.pth"
    src_DAMSM_data = torch.load(src_DAMSM_path, map_location=lambda storage, loc: storage)
    embedding_dim = src_DAMSM_data['settings'].out_feat_size
    T2I_src_vocab_size = src_DAMSM_data['settings'].vocab_size

    src_DAMSM_RNN = DAMSMTextEncoder(T2I_src_vocab_size, out_feat_size=embedding_dim)
    src_DAMSM_RNN.load_state_dict(src_DAMSM_data['text_encoder'])
    set_requires_grad(src_DAMSM_RNN, False)
    src_DAMSM_RNN.eval()
    src_DAMSM_RNN.to(device)
    print(f"[Info] Loading complete ({src_DAMSM_path})")

    tgt_DAMSM_RNN = DAMSMTextEncoder(trained_setting.T2I_tgt_vocab_size, out_feat_size=embedding_dim)
    tgt_DAMSM_RNN.load_state_dict(trained_data['models']['tgt_DAMSM'])
    set_requires_grad(tgt_DAMSM_RNN, False)
    tgt_DAMSM_RNN.eval()
    tgt_DAMSM_RNN.to(device)

    # --- Load G network ---
    netG = Generator(
        embedding_dim=embedding_dim,
        condition_dim=T2I_settings.condition_dim,
        noise_dim=T2I_settings.noise_dim,
        G_hidden=T2I_settings.G_hidden,
        stage_num=T2I_settings.stage_num,
        num_resblocks=T2I_settings.num_resblocks
    )
    netG.apply(weights_init)
    netG.load_state_dict(trained_data['models']['netG'])
    set_requires_grad(netG, False)
    netG.to(device)
    netG.eval()

    # --- Load MMT model setting ---
    pre_MMT_dir, pre_MMT_epoch = trained_data["settings"].MMT.split(',')
    pre_MMT_path = f"{pardir}/MMT/results/{pre_MMT_dir}/trained_models/epoch_{pre_MMT_epoch}.pth"
    MMT_settings = torch.load(pre_MMT_path, map_location=lambda storage, loc: storage)["settings"]

    # --- Load MMT models ---
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
    MMT.load_state_dict(trained_data['models']['MMT'])
    MMT.to(device)
    print(f"[Info] Loading complete ({trained_model_path})")

    opt.data_path = trained_setting.data_path
    opt.src_lang = trained_setting.src_lang
    opt.tgt_lang = trained_setting.tgt_lang
    opt.d_scale = trained_setting.d_scale
    opt.bpe = trained_setting.bpe
    opt.noise_dim = trained_setting.noise_dim

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

    # -- Prepare DataLoader --
    src_DAMSM_RNN, tgt_DAMSM_RNN, netG, MMT = prepare_models(opt, device)
    test_loader = get_eval_loader(opt)
    mkdirs(opt.save_image_dir)

    sampler = Sampler(src_DAMSM_RNN, tgt_DAMSM_RNN, netG, MMT, test_loader, opt)
    if opt.image_id:
        sampler.sampling_from_image_id(opt.image_id)
    else:
        sampler.sampling()


if __name__ == '__main__':
    main()
