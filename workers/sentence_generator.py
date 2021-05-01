import math
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from re import compile, sub
from collections import deque

## [Pytorch] ############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
#########################################################################################

## [Self-Module] ########################################################################
import my_utils.Constants as Constants
from my_utils.pytorch_utils import get_device, get_state_dict, load_params, copy_params
#########################################################################################

class ScoreCalculator:
    def __init__(self, model, data_loader, references=None, bpe=None,
                 cp_avg_num=1, beam_size=4, len_penalty=1.0):
        assert cp_avg_num >= 1
        self.model = model
        self.data_loader = data_loader
        self.references = references
        self.cp_avg_num = cp_avg_num
        if cp_avg_num > 1:
            self.queue = deque()
            params = copy_params(self.model, to_cpu=True)
            self.sum_params = [torch.zeros(p.shape) for p in params]

        self.generator = SentenceGenerator(
            model=model,
            tgt_index2word=data_loader.dataset.tgt_index2word,
            bpe=bpe,
            beam_size=beam_size,
            len_penalty=len_penalty
        )

    def _get_no_cp_avg_bleu(self, use_beam=False):
        pred_words = self.generator.generate_loader(self.data_loader, use_beam)
        state_dict = get_state_dict(self.model)
        return pred_words, state_dict

    def _cp_avg(self):
        queue_num = len(self.queue)
        assert queue_num <= self.cp_avg_num

        params = copy_params(self.model, to_cpu=True)
        self.queue.append(params)
        self.sum_params = [sum_p + p for sum_p, p in zip(self.sum_params, params)]

        if queue_num < self.cp_avg_num:
            avg_params = [sum_p / (queue_num + 1) for sum_p in self.sum_params]
        else:
            old_params = self.queue.popleft()
            self.sum_params = [sum_p - old_p for sum_p, old_p in zip(self.sum_params, old_params)]
            avg_params = [sum_p / self.cp_avg_num for sum_p in self.sum_params]

        return avg_params

    def _get_cp_avg_bleu(self, use_beam=False, return_sentences=False):
        back_up_params = copy_params(self.model)
        avg_params = self._cp_avg()
        load_params(self.model, avg_params)
        pred_words = self.generator.generate_loader(self.data_loader, use_beam)
        state_dict = get_state_dict(self.model)
        load_params(self.model, back_up_params)
        return pred_words, state_dict

    def get_bleu(self, use_beam=False, return_sentences=False):
        if self.cp_avg_num > 1:
            pred_words, state_dict = self._get_cp_avg_bleu(use_beam)
        else:
            pred_words, state_dict = self._get_no_cp_avg_bleu(use_beam)

        bleu_score = corpus_bleu(self.references, pred_words) * 100

        if return_sentences:
            return bleu_score, pred_words
        return bleu_score, state_dict


class SentenceGenerator:
    def __init__(self, model, tgt_index2word, bpe=None, beam_size=4, len_penalty=1.0):
        self.model = model
        self.tgt_index2word = tgt_index2word
        self.bpe = bpe
        self.beam_size = beam_size
        self.len_penalty = len_penalty

        self.vocab_size = len(self.tgt_index2word)
        self.bpe_re = compile("@@ |@@ ?$")


    def get_dec_pos(self, dec_seq_len, inst_num):
        # (dec_seq_len)
        dec_pos = torch.arange(1, dec_seq_len + 1, dtype=torch.long)
        # (dec_seq_len) >> (inst_num, dec_seq_len)
        dec_pos = dec_pos.unsqueeze(0).repeat(inst_num, 1)
        return dec_pos


    def generate_by_greedy(self, batch_inputs):
        self.model.eval()
        model = self.model.module if "DataParallel" in str(type(self.model)) else self.model
        device = model.get_device()
        
        # enc_outs = tuple(src_seq, enc_out)
        enc_outs = model.forward_encoder(
            *tuple(map(lambda x: x.to(device), batch_inputs))
        )

        batch_size, src_len = enc_outs[0].size()
        dec_max_len = src_len + 50

        remaining_batch_num = batch_size
        is_finished = [False for _ in range(batch_size)]
        finalized_info = [[] for _ in range(batch_size)]

        dec_seqs = torch.full((batch_size, dec_max_len + 2), Constants.PAD, dtype=torch.long)
        dec_seqs[:, 0] = Constants.BOS

        required_batch_ids = None
        for step in range(dec_max_len + 1):
            if required_batch_ids is not None:
                enc_outs = model.select_enc_outs(enc_outs, required_batch_ids.to(device))
            
            dec_pos = self.get_dec_pos(step + 1, dec_seqs.size(0))
            dec_out = model.forward_decoder(
                dec_seqs[:, :step + 1].to(device),
                dec_pos.to(device),
                *enc_outs
            )

            dec_out = dec_out[:, -1, :].cpu()
            word_ids = torch.argmax(dec_out, dim=1)

            if step >= dec_max_len:
                word_ids[:] = Constants.EOS
            
            dec_seqs[:, step + 1] = word_ids

            eos_mask = word_ids.eq(Constants.EOS)
            finished_batch_ids = [x.item() for x in eos_mask.nonzero(as_tuple=False)]

            finished_batch_num = len(finished_batch_ids)
            if finished_batch_num > 0:
                finalize_dec_seqs = dec_seqs[eos_mask][:, 1:step+2]
                
                curr_to_whole_batch_idx = []
                whole_batch_idx = 0
                for f in is_finished:
                    if not f:
                        curr_to_whole_batch_idx.append(whole_batch_idx)
                    whole_batch_idx += 1

                for i in range(finished_batch_num):
                    whole_batch_idx = curr_to_whole_batch_idx[finished_batch_ids[i]]
                    assert len(finalized_info[whole_batch_idx]) == 0                        
                    finalized_info[whole_batch_idx].append(
                        {
                            "tokens": finalize_dec_seqs[i]
                        }
                    )
                    is_finished[whole_batch_idx] = True

                remaining_batch_num -= finished_batch_num

                # --- Break Check ---
                assert remaining_batch_num >= 0    
                if remaining_batch_num == 0:
                    break
                assert step < dec_max_len

                dec_seqs = dec_seqs[~eos_mask]
                required_batch_ids = (~eos_mask).nonzero(as_tuple=False).squeeze(-1)
            else:
                required_batch_ids = None

        self.model.train()
        return finalized_info


    def generate_by_beam(self, batch_inputs):
        self.model.eval()
        model = self.model.module if "DataParallel" in str(type(self.model)) else self.model
        device = model.get_device()
        
        # enc_outs = tuple(src_seq, enc_out)
        enc_outs = model.forward_encoder(
            *tuple(map(lambda x: x.to(device), batch_inputs))
        )

        batch_size, src_len = enc_outs[0].size()
        dec_max_len = src_len + 50
        beam_size = self.beam_size

        # [batch_size]
        remaining_batch_num = batch_size
        is_finished = [False for _ in range(batch_size)]
        finalized_info = [[] for _ in range(batch_size)]

        # (batch_size) >> (batch_size, beam_size) >> (batch_size * beam_size) = (inst_num)
        required_inst_ids = torch.arange(batch_size).unsqueeze(-1).repeat(1, beam_size).view(-1)

        # (inst_num, dec_max_len + 1)
        scores = torch.zeros(batch_size * beam_size, dec_max_len + 1).float()
        # (inst_num, dec_max_len + 2)
        dec_seqs = torch.full((batch_size * beam_size, dec_max_len + 2), Constants.PAD, dtype=torch.long)
        dec_seqs[:, 0] = Constants.BOS

        # (batch_size, beam_size)
        ignore_mask = torch.full((batch_size, beam_size), False, dtype=torch.bool)   
        # (batch_size, 1)
        beam_id_to_inst_id = (torch.arange(0, batch_size) * beam_size).unsqueeze(1).long()
        # (beam_size * 2)
        beam_ranking = torch.arange(0, beam_size * 2).long()

        batch_ids = None
        for step in range(dec_max_len + 1):
            if required_inst_ids is not None:
                # --- Remove finished insts ---
                if batch_ids is not None:
                    correction = batch_ids - torch.arange(batch_ids.numel())
                    required_inst_ids.view(-1, beam_size).add_(correction.unsqueeze(-1) * beam_size)

                enc_outs = model.select_enc_outs(enc_outs, required_inst_ids.to(device))

            # --- Forward decoder ---
            inst_num = dec_seqs.size(0)
            dec_pos = self.get_dec_pos(step + 1, inst_num)
            dec_out = model.forward_decoder(
                dec_seqs[:, :step + 1].to(device),
                dec_pos.to(device),
                *enc_outs
            )

            # --- Extracting last word score ---
            # (inst_num, dec_seq_len, vocab_size) >> (inst_num, vocab_size)
            dec_out = dec_out[:, -1, :].cpu()
            lprobs = F.log_softmax(dec_out, dim=-1)
            lprobs[:, Constants.PAD] = -math.inf
            lprobs[:, Constants.BOS] = -math.inf
            scores = scores.type_as(lprobs)
            
            # --- Case of maximum length ---
            if step >= dec_max_len:
                lprobs[:, :Constants.EOS] = -math.inf
                lprobs[:, Constants.EOS + 1:] = -math.inf

            # --- Get topk information ---
            # (batch_size, beam_size * 2)
            topk_scores, topk_word_ids, topk_beam_ids = self.beam_step(
                step,
                lprobs.view(batch_size, -1, self.vocab_size),
                scores.view(batch_size, beam_size, -1)[:, :, :step]
            )

            # --- Prepare EOS mask ---
            topk_inst_ids = topk_beam_ids.add(beam_id_to_inst_id)
            eos_mask = topk_word_ids.eq(Constants.EOS) & topk_scores.ne(-math.inf)
            eos_mask[:, :beam_size][ignore_mask] = False
            
            # --- Finalize sentences with EOS ---
            # (eos_ists_num)
            eos_inst_ids = torch.masked_select(topk_inst_ids[:, :beam_size], mask=eos_mask[:, :beam_size])
            finished_batch_ids = []
            if eos_inst_ids.numel() > 0:
                eos_scores = torch.masked_select(topk_scores[:, :beam_size], mask=eos_mask[:, :beam_size])
                finished_batch_ids = self.finalize_hypos(
                    step,
                    eos_inst_ids,
                    eos_scores,
                    dec_seqs,
                    scores,
                    finalized_info,
                    is_finished,
                    beam_size,
                    dec_max_len
                )
                remaining_batch_num -= len(finished_batch_ids)

            # --- Break Check ---
            assert remaining_batch_num >= 0    
            if remaining_batch_num == 0:
                break
            assert step < dec_max_len

            # --- Remove finished batches ---
            if len(finished_batch_ids) > 0:
                new_batch_size = batch_size - len(finished_batch_ids)

                batch_mask = torch.full((batch_size,), True, dtype=torch.bool)
                batch_mask[finished_batch_ids] = False
                batch_ids = batch_mask.nonzero(as_tuple=False).squeeze(-1)

                scores = scores.view(batch_size, -1)[batch_ids].view(new_batch_size * beam_size, -1)
                dec_seqs = dec_seqs.view(batch_size, -1)[batch_ids].view(new_batch_size * beam_size, -1)
                ignore_mask = ignore_mask[batch_ids]
                beam_id_to_inst_id.resize_(new_batch_size, 1)
                eos_mask = eos_mask[batch_ids]

                topk_scores = topk_scores[batch_ids]
                topk_word_ids = topk_word_ids[batch_ids]
                topk_beam_ids = topk_beam_ids[batch_ids]
                topk_inst_ids = topk_beam_ids.add(beam_id_to_inst_id)
            
                batch_size = new_batch_size
            else:
                batch_ids = None

            # --- Prepare next step ---
            # (batch_size, beam_size * 2)
            eos_mask[:, :beam_size] = ~((~ignore_mask) & (~eos_mask[:, :beam_size]))
            # (batch_size, beam_size * 2)
            next_beam_ranking = torch.add(
                eos_mask.type_as(beam_ranking) * (beam_size * 2),
                beam_ranking
            )
            # (batch_size, beam_size)
            active_beam_ranking, active_beam_ids = torch.topk(
                next_beam_ranking, k=beam_size, dim=1, largest=False
            )
            # (batch_size, beam_size)
            ignore_mask = active_beam_ranking.ge(beam_size*2)[:, :beam_size]
            assert (~ignore_mask).any(dim=1).all()
            # (batch_size, beam_size)
            active_inst_ids = torch.gather(topk_inst_ids, dim=1, index=active_beam_ids)
            active_inst_ids = active_inst_ids.view(-1)

            # --- Copy the previous tokens and scores ---
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], dim=0, index=active_inst_ids
                )
            dec_seqs[:, :step+1] = torch.index_select(
                dec_seqs[:, :step+1], dim=0, index=active_inst_ids
            )

            # --- Add next token and score ---
            scores.view(batch_size, beam_size, -1)[:, :, step] = torch.gather(
                topk_scores, dim=1, index=active_beam_ids
            )
            dec_seqs.view(batch_size, beam_size, -1)[:, :, step+1] = torch.gather(
                topk_word_ids, dim=1, index=active_beam_ids
            )

            required_inst_ids = active_inst_ids

        # --- Sorting info in beam size by score ---
        for batch_idx in range(len(finalized_info)):
            beam_list = [BeamSort(info["score"].item(), info) for info in finalized_info[batch_idx]]
            beam_list.sort()
            beam_list.reverse()
            finalized_info[batch_idx] = [x.info for x in beam_list]

        self.model.train()
        return finalized_info


    def beam_step(self, step, lprobs, scores):
        batch_size, beam_size, vocab_size = lprobs.size()

        if step == 0:
            lprobs = lprobs[:, ::beam_size, :].contiguous()
        else:
            lprobs = lprobs + scores[:, :, step -1].unsqueeze(-1)

        topk_scores, topk_indices = torch.topk(
            lprobs.view(batch_size, -1),
            k=beam_size*2
        )
        topk_beam_ids = topk_indices // vocab_size
        topk_word_ids = topk_indices.fmod(vocab_size)
        return topk_scores, topk_word_ids, topk_beam_ids


    def finalize_hypos(self, step, eos_inst_ids, eos_scores, dec_seqs, all_scores,
                       finalized_info, is_finished, beam_size, max_len):
        # --- Finalize dec_seqs ---
        finalize_dec_seqs = dec_seqs.index_select(0, eos_inst_ids)[:, 1:step+2]
        finalize_dec_seqs[:, step] = Constants.EOS

        # --- Finalize scores ---
        finalize_scores = all_scores.index_select(0, eos_inst_ids)[:, :step+1]
        finalize_scores[:, step] = eos_scores
        # --- Convert from cumulative to per-position scores ---
        finalize_scores[:, 1:] = finalize_scores[:, 1:] - finalize_scores[:, :-1]

        # --- Apply length penalty ---
        eos_scores /= (step + 1) ** self.len_penalty
        #eos_scores /= ((5 + (step + 1)) / (5 + 1)) ** self.len_penalty

        curr_to_whole_batch_idx = []
        whole_batch_idx = 0
        for f in is_finished:
            if not f:
                curr_to_whole_batch_idx.append(whole_batch_idx)
            whole_batch_idx += 1

        maybe_finished_batches = {}
        eos_inst_num = eos_inst_ids.numel()
        for i in range(eos_inst_num):
            idx = eos_inst_ids[i]
            score = eos_scores[i]
            curr_batch_idx = idx // beam_size
            whole_batch_idx = curr_to_whole_batch_idx[curr_batch_idx]

            key = str(whole_batch_idx) + '_' + str(curr_batch_idx.item())
            if key not in maybe_finished_batches:
                maybe_finished_batches[key] = None

            if len(finalized_info[whole_batch_idx]) < beam_size:
                finalized_info[whole_batch_idx].append(
                    {
                        "tokens": finalize_dec_seqs[i],
                        "score": score,
                        "pos_scores": finalize_scores[i],
                    }
                )

        finished_batch_ids = []
        for key in maybe_finished_batches.keys():
            whole_batch_idx, curr_batch_idx = key.split('_')
            whole_batch_idx = int(whole_batch_idx)
            curr_batch_idx = int(curr_batch_idx)
            if not is_finished[whole_batch_idx]:
                if len(finalized_info[whole_batch_idx])==beam_size or step == max_len:
                    is_finished[whole_batch_idx] = True
                    finished_batch_ids.append(curr_batch_idx)
            
        return finished_batch_ids


    @torch.no_grad()
    def generate_loader(self, loader, use_beam=False, return_indices=False):
        pred_indices = []
        pred_words = []

        pbar = tqdm(loader, ncols=90, mininterval=0.5, ascii=True)
        for batch_datas in pbar:
            if use_beam:
                finalized_info = self.generate_by_beam(batch_datas)
            else:
                finalized_info = self.generate_by_greedy(batch_datas)

            for cnt, info in enumerate(finalized_info):
                info = info[0] # extruct best info
                tokens = info["tokens"][:-1] # remove EOS
                tokens = [idx.item() for idx in tokens]
                words = [self.tgt_index2word[idx] for idx in tokens]
                if self.bpe is not None:
                    seq = ' '.join(words)
                    seq = self.bpe_re.sub('', seq)
                    words = seq.split(' ')
                #if len(words) == 1:
                    #print(words)
                    #print(batch_datas[0][cnt])
                pred_indices.append(tokens)
                pred_words.append(words)
        
        if return_indices:
            return pred_words, pred_indices
        return pred_words


class BeamSort:
    def __init__(self, score, info):
        self.score = score
        self.info = info

    def __lt__(self, other):
        return self.score <= other.score
