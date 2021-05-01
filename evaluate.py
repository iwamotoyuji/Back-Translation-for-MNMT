import os
import argparse
from pathlib import Path

## [Pytorch] ############################################################################
import torch
#########################################################################################

## [Self-Module] ########################################################################
from datasets.BT_MNMT_dataset import get_eval_loader
from models.transformers import MultimodalTransformer
from workers.sentence_generator import ScoreCalculator
#########################################################################################


def parse_args():
    parser = argparse.ArgumentParser(description='evaluate unsupervised MNMT')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-e', '--experiment_name', default="test")
    parser.add_argument('-g', '--gpu_ids', default='0')
    parser.add_argument('-l', '--load_num', type=int, default=0)
    parser.add_argument('-w', '--workers', type=int, default=2)

    parser.add_argument('--use_beam', action='store_true')
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--len_penalty', type=float, default=0.6)
    parser.add_argument('--no_use_image_info', action='store_true')

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
    
    trained_data = torch.load(trained_model_path, map_location=lambda storage, loc: storage)
    settings = trained_data['settings']

    opt.data_path = settings.data_path
    opt.src_lang = settings.src_lang
    opt.tgt_lang = settings.tgt_lang
    opt.d_scale = settings.d_scale
    opt.bpe = settings.bpe
    opt.adapt_init_MNMT = settings.adapt_init_MNMT
    opt.adapt_prop_MNMT = settings.adapt_prop_MNMT

    if opt.adapt_prop_MNMT:
        MNMT_dir, MNMT_epoch = settings.MNMT.split(',')
        model_path = Path(__file__).parent / "results" / MNMT_dir / f"trained_models/epoch_{MNMT_epoch}.pth"
        model_data = torch.load(model_path, map_location=lambda storage, loc: storage)
        settings = model_data['settings']

    MNMT_dir, MNMT_epoch = settings.MNMT.split(',')
    pre_MNMT_path = Path(__file__).parent / "MNMT/results" / MNMT_dir / f"trained_models/epoch_{MNMT_epoch}.pth"
    pre_MNMT_settings = torch.load(pre_MNMT_path, map_location=lambda storage, loc: storage)['settings']

    model = MultimodalTransformer(
        src_vocab_size=pre_MNMT_settings.src_vocab_size,
        tgt_vocab_size=pre_MNMT_settings.tgt_vocab_size,
        max_position_num=pre_MNMT_settings.max_position_num,
        d_model=pre_MNMT_settings.d_model,
        head_num=pre_MNMT_settings.head_num,
        d_k=pre_MNMT_settings.d_k,
        d_v=pre_MNMT_settings.d_v,
        d_inner=pre_MNMT_settings.d_inner,
        layer_num=pre_MNMT_settings.layer_num,
        dropout=pre_MNMT_settings.dropout,
        cnn_fine_tuning=False,
        shared_embedding=pre_MNMT_settings.shared_embedding,
        share_dec_input_output_embed=pre_MNMT_settings.share_dec_input_output_embed,
        init_weight=False,
        fused_layer_norm=pre_MNMT_settings.use_fused,
    ).to(device)
    model.load_state_dict(trained_data['models']['MNMT'])
    print(f"[Info]Loading completed from {trained_model_path}")
    
    # -- Prepare DataLoader --
    eval_loader = get_eval_loader(opt)    

    # -- Prepare Translator --
    evaluator = ScoreCalculator(
        model=model,
        data_loader=eval_loader,
        references=eval_loader.dataset.MNMT_tgt_insts,
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
