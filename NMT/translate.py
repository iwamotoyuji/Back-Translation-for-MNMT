import os
import sys
import argparse
import numpy as np
import pickle
from pathlib import Path
pardir = Path(__file__).resolve().parent.parent
sys.path.append(str(pardir))

## [Pytorch] ############################################################################
import torch
#########################################################################################

## [Self-Module] ########################################################################
from datasets.NMT_dataset import get_trans_loader
from models.transformers import Transformer
import my_utils.Constants as Constants
from workers.sentence_generator import SentenceGenerator
#########################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description='translate source sentences')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-e', '--experiment_name', default='test')
    parser.add_argument('-g', '--gpu_ids', default='0')
    parser.add_argument('-l', '--load_num', type=int, default=0)
    parser.add_argument('-w', '--workers', type=int, default=4)

    parser.add_argument('--tgt_mode', default="train")
    parser.add_argument('--use_beam', action='store_true')
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--len_penalty', type=float, default=0.6)

    args = parser.parse_args()
    return args


def main():
    opt = parse_args()

    # -- CUDA setting --
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -- Load Model and settings --
    trained_dir = Path(__file__).parent / "results" / opt.experiment_name
    if opt.load_num != 0:
        if opt.load_num < 500:
            model_name = f"epoch_{opt.load_num}.pth"
        else:
            model_name = f"step_{opt.load_num}.pth"
    else:
        model_name = "best.pth"
    trained_model_path = trained_dir / "trained_models" / model_name
    
    trained_datas = torch.load(trained_model_path)
    settings = trained_datas['settings']

    model = Transformer(
        src_vocab_size=settings.src_vocab_size,
        tgt_vocab_size=settings.tgt_vocab_size,
        max_position_num=settings.max_position_num,
        d_model=settings.d_model,
        head_num=settings.head_num,
        d_k=settings.d_k,
        d_v=settings.d_v,
        d_inner=settings.d_inner,
        layer_num=settings.layer_num,
        dropout=settings.dropout,
        shared_embedding=settings.shared_embedding,
        share_dec_input_output_embed=settings.share_dec_input_output_embed,
        init_weight=False,
        fused_layer_norm=settings.use_fused,
    ).to(device)
    model.load_state_dict(trained_datas['model'])
    print(f"[Info] Loading completed from {trained_model_path}")
    
    # -- Prepare DataLoader --
    opt.data_path = settings.data_path
    opt.src_lang = settings.src_lang
    opt.tgt_lang = settings.tgt_lang
    opt.d_scale = settings.d_scale
    opt.bpe = settings.bpe
    trans_loader = get_trans_loader(opt)

    # -- Prepare Translator --
    translator = SentenceGenerator(
        model=model,
        tgt_index2word=trans_loader.dataset.tgt_index2word,
        bpe=opt.bpe,
        beam_size=opt.beam_size,
        len_penalty=opt.len_penalty,
    )

    # -- Translate  --
    pred_words, pred_indices = translator.generate_loader(
        trans_loader,
        use_beam=opt.use_beam,
        return_indices=True,
    )

    data_path = opt.data_path.format(mode=opt.tgt_mode)
    img_cap = Constants.D_SCALE[opt.d_scale]['img_cap']
    cap_per_img = Constants.CAP_PER_IMG[img_cap]    
    with open(data_path, 'rb') as f:
        x = pickle.load(f)
        img_insts = x[img_cap]['img']

    # -- Make no bpe indices --
    if opt.bpe:
        bpe_indices = pred_indices
        word2id = trans_loader.dataset.no_bpe_word2id
        pred_indices = [[word2id.get(word, Constants.UNK) for word in sent] for sent in pred_words]

    # -- Remove sentences of length 1 --
    if cap_per_img == 1:
        num_insts = len(img_insts)
        assert num_insts == len(pred_indices)

        indices = []
        tgt_indices = []
        for i in range(num_insts):
            tgt = pred_indices[i]
            if len(tgt) != 1:
                indices.append(i)
                tgt_indices.append(tgt)

        img_insts = list(np.array(img_insts)[indices])
        src_indices = list(np.array(x[img_cap][opt.src_lang]['id'])[indices])
        x[img_cap]['img'] = img_insts
        x[img_cap][opt.src_lang]['id'] = src_indices
        x[img_cap][opt.tgt_lang]['id'] = tgt_indices
        x[img_cap][opt.tgt_lang]['word'] = pred_words
        if opt.bpe:
            src_bpe = list(np.array(x[img_cap][opt.src_lang]['bpe'])[indices])
            tgt_bpe = list(np.array(bpe_indices)[indices])
            x[img_cap][opt.src_lang]['bpe'] = src_bpe
            x[img_cap][opt.tgt_lang]['bpe'] = tgt_bpe
    else:
        x[img_cap][opt.tgt_lang]['id'] = pred_indices
        x[img_cap][opt.tgt_lang]['word'] = pred_words
        if opt.bpe:
            x[img_cap][opt.tgt_lang]['bpe'] = bpe_indices

    # -- Save --
    with open(data_path, 'wb') as f:
        pickle.dump(x, f)

    output_path = trained_dir / 'init_tgt.txt'
    with open(output_path, mode='w', encoding='utf-8') as output_file:
        for pred_sentence in pred_words:
            pred_line = ' '.join(pred_sentence)
            output_file.write(pred_line + '\n')
    print('[Info] Finish!!')


if __name__ == "__main__":
    main()
