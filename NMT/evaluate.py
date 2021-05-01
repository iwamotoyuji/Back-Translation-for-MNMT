import os
import sys
import argparse
from pathlib import Path
pardir = Path(__file__).resolve().parent.parent
sys.path.append(str(pardir))

## [Pytorch] ############################################################################
import torch
#########################################################################################

## [Self-Module] ########################################################################
from datasets.NMT_dataset import get_eval_loader
from models.transformers import Transformer
from workers.sentence_generator import ScoreCalculator
#########################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description="evaluate trained model")
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-e', '--experiment_name', default="test")
    parser.add_argument('-g', '--gpu_ids', default='0')
    parser.add_argument('-l', '--load_num', type=int, default=0)
    parser.add_argument('-w', '--workers', type=int, default=4)

    parser.add_argument('--use_beam', action='store_true')
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--len_penalty', type=float, default=0.6)

    args = parser.parse_args()
    return args


def main():
    opt = parse_args()

    # -- CUDA setting --
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    settings = trained_datas["settings"]

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
    model.load_state_dict(trained_datas["model"])
    print(f"[Info]Loading completed from {trained_model_path}")
    
    # -- Prepare DataLoader --
    opt.data_path = settings.data_path
    opt.src_lang = settings.src_lang
    opt.tgt_lang = settings.tgt_lang
    opt.d_scale = settings.d_scale
    opt.bpe = settings.bpe
    opt.adapt_NMT = settings.adapt_NMT
    eval_loader = get_eval_loader(opt)

    # -- Prepare Evaluator --
    evaluator = ScoreCalculator(
        model=model,
        data_loader=eval_loader,
        references=eval_loader.dataset.tgt_insts,
        bpe=opt.bpe,
        beam_size=opt.beam_size,
        len_penalty=opt.len_penalty,
    )

    # -- Translate and Evaluate --
    bleu_score, pred_words = evaluator.get_bleu(
        use_beam=opt.use_beam,
        return_sentences=True
    )
    print(f"BLEU = {bleu_score:.2f}")
    
    output_path = trained_dir / "result.txt"
    with open(output_path, mode='w', encoding='utf-8') as output_file:
        for pred_sentence in pred_words:
            pred_line = ' '.join(pred_sentence)
            output_file.write(pred_line + '\n')
    print("[Info] Finish!!")


if __name__ == "__main__":
    main()
