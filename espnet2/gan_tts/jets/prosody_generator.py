# Copyright 2022 Dan Lim
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Generator module in JETS."""
from typing import Optional, Tuple, Callable

import torch

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, make_pad_mask
from espnet2.gan_tts.jets.alignments import (
    average_by_duration,
    viterbi_decode,
)
from espnet2.gan_tts.jets.generator import JETSGenerator
from prosody.transfer_prosody import transfer_prosody


class JETSProsodyGenerator(JETSGenerator):

    def inference(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        feats_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        d_factor: Optional[torch.Tensor] = None,
        p_factor: Optional[torch.Tensor] = None,
        e_factor: Optional[torch.Tensor] = None,
        d_split_factor: Optional[torch.Tensor] = None,
        d_mod_fns: Optional[Callable] = None,
        p_mod_fns: Optional[Callable] = None,
        e_mod_fns: Optional[Callable] = None,
        is_vowels: Optional[torch.Tensor] = None,
        is_consonants: Optional[torch.Tensor] = None,
        pitch_range: Optional[Tuple[float, float]] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        p_reference: Optional[torch.Tensor] = None,
        e_reference: Optional[torch.Tensor] = None,
        p_reference_weight: float = 1.0,
        e_reference_weight: float = 0.5,
        e_threshold: float = -1,
        sil_threshold: float = -1,
        avg_energy_window: int = 0,
        change_durations: bool = False,
        use_teacher_forcing: bool = False,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference.

        Args:
            text (Tensor): Input text index tensor (B, T_text,).
            text_lengths (Tensor): Text length tensor (B,).
            feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            feats_lengths (Tensor): Feature length tensor (B,).
            pitch (Tensor): Pitch tensor (B, T_feats, 1)
            energy (Tensor): Energy tensor (B, T_feats, 1)
            sids (Optional[Tensor]): Speaker index tensor (B,) or (B, 1).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, spk_embed_dim).
            lids (Optional[Tensor]): Language index tensor (B,) or (B, 1).
            use_teacher_forcing (bool): Whether to use teacher forcing.

        Returns:
            Tensor: Generated waveform tensor (B, T_wav).
            Tensor: Duration tensor (B, T_text).

        """
        # forward encoder
        x_masks = self._source_mask(text_lengths)
        hs, _ = self.encoder(text, x_masks)  # (B, T_text, adim)

        # integrate with GST
        if self.use_gst:
            style_embs = self.gst(feats)
            hs = hs + style_embs.unsqueeze(1)

        # integrate with SID and LID embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        h_masks = make_pad_mask(text_lengths).to(hs.device)

        preds = None
        if use_teacher_forcing:
            # forward alignment module and obtain duration, averaged pitch, energy
            log_p_attn = self.alignment_module(hs, feats, h_masks)
            d_outs, _ = viterbi_decode(log_p_attn, text_lengths, feats_lengths)
            p_outs = average_by_duration(
                d_outs, pitch.squeeze(-1), text_lengths, feats_lengths
            ).unsqueeze(-1)
            e_outs = average_by_duration(
                d_outs, energy.squeeze(-1), text_lengths, feats_lengths
            ).unsqueeze(-1)
        else:
            # forward duration predictor and variance predictors
            d_outs = self.duration_predictor(hs, h_masks)
            d_outs = d_outs.exp() - self.duration_predictor.offset  # (B, T_text)
            p_outs = self.pitch_predictor(hs, h_masks.unsqueeze(-1))  # (B, T_text, 1)
            e_outs = self.energy_predictor(hs, h_masks.unsqueeze(-1))  # (B, T_text, 1)
            if verbose:
                print('Duration pred:', d_outs.squeeze())
                print('Pitch pred:', p_outs.squeeze())
                print('Energy pred:', e_outs.squeeze())
                preds = (d_outs.squeeze(), p_outs.squeeze(), e_outs.squeeze())
            if d_factor is not None:
                d_outs *= d_factor
            if p_factor is not None:
                p_outs += p_factor.unsqueeze(-1)
            if e_factor is not None:
                e_outs += e_factor.unsqueeze(-1)
            if d_split_factor is not None:
                assert d_split_factor.dtype in {torch.short, torch.int, torch.long}
                pitch_pred = p_outs.squeeze(-1)
                energy_pred = e_outs.squeeze(-1)
                d_outs = self._expand_embed_duration(d_outs, d_split_factor, d_mod_fns)
                p_outs = self._expand_embed_duration(pitch_pred, d_split_factor, p_mod_fns).unsqueeze(-1)
                e_outs = self._expand_embed_duration(energy_pred, d_split_factor, e_mod_fns).unsqueeze(-1)
                hs = self._expand_embed_duration(hs, d_split_factor)
                text_lengths = d_split_factor.sum(dim=-1)
            elif p_reference is not None and e_reference is not None:
                duration_pred = d_outs.squeeze()
                pitch_pred = p_outs.squeeze()
                energy_pred = e_outs.squeeze()
                duration_new, pitch_new, energy_new = transfer_prosody(
                    duration_pred, pitch_pred, energy_pred, pitch_range, energy_range,
                    p_reference, e_reference, e_threshold, sil_threshold,
                    is_vowels, is_consonants, avg_energy_window, change_durations
                )
                pitch_new = pitch_new[None].unsqueeze(-1)
                energy_new = energy_new[None].unsqueeze(-1)
                d_outs = duration_new[None]
                p_outs = p_outs * (1-p_reference_weight) + pitch_new * p_reference_weight
                e_outs = e_outs * (1-e_reference_weight) + energy_new * e_reference_weight

            d_outs = torch.clamp(torch.round(d_outs), min=0).long()
            if verbose:
                print('Duration (new):', d_outs.squeeze())
                print('Pitch (new):', p_outs.squeeze())
                print('Energy (new):', e_outs.squeeze())

        p_embs = self.pitch_embed(p_outs.transpose(1, 2)).transpose(1, 2)
        e_embs = self.energy_embed(e_outs.transpose(1, 2)).transpose(1, 2)

        hs = hs + e_embs + p_embs

        # upsampling
        if feats_lengths is not None:
            h_masks = make_non_pad_mask(feats_lengths).to(hs.device)
        else:
            h_masks = None
        d_masks = make_non_pad_mask(text_lengths).to(d_outs.device)
        hs = self.length_regulator(hs, d_outs, h_masks, d_masks)  # (B, T_feats, adim)

        # if d_factor is not None and d_interpolate:
        #     hs = hs.transpose(2, 1)  # (B, adim, T_feats) so that we can interpolate on the T_feats dimension
        #     old_durations = d_outs.squeeze(-1)
        #     print(old_durations)
        #     new_durations = (d_factor * old_durations).round().int()
        #     print(new_durations)
        #     longest_dur = new_durations.sum(dim=1).max()
        #     new_hs = torch.zeros((hs.size(0), hs.size(1), longest_dur), dtype=hs.dtype, device=hs.device)
        #     old_indices = torch.zeros(
        #         (old_durations.size(0), old_durations.size(1) + 1),
        #         dtype=old_durations.dtype, device=old_durations.device
        #     )
        #     old_indices[:, 1:] = old_durations.cumsum(dim=1)
        #     new_indices = torch.clone(old_indices)
        #     new_indices[:, 1:] = new_durations.cumsum(dim=1)
        #
        #     for i in range(old_durations.size(0)):
        #         for j in range(new_durations.size(1)):
        #             old_start = old_indices[i, j]
        #             old_end = old_indices[i, j+1]
        #             new_start = new_indices[i, j]
        #             new_end = new_indices[i, j+1]
        #             if new_start < new_end:
        #                 segment = hs[i:i+1, :, old_start:old_end]
        #                 segment = torch.nn.functional.interpolate(
        #                     segment, size=new_end-new_start, mode='linear', align_corners=True
        #                 )
        #                 new_hs[i:i+1, :, new_start:new_end] = segment
        #     hs = new_hs.transpose(2, 1)  # back to (B, T_feats_longest, adim)

        # forward decoder
        if feats_lengths is not None:
            h_masks = self._source_mask(feats_lengths)
        else:
            h_masks = None
        zs, _ = self.decoder(hs, h_masks)  # (B, T_feats, adim)

        # forward generator
        wav = self.generator(zs.transpose(1, 2))

        return wav.squeeze(1), d_outs  #, preds

    def _expand_embed_duration(self, preds, d_split_factor, mod_fns=None):
        '''
        To achieve subphoneme control, we use mod_fns to split the phoneme duration into subphoneme durations
        and set a different pitch/energy for each subphoneme.

        Example with batch size 1: Yes! (with falling tone) -> Y EH1 S
        ```
        hs = [[h0, h1, h2]]  # h is a vector with dimension adim 
        duration_pred = [[3.1, 6.0, 5.2]]
        pitch_pred = [[0.2, 0.4, -1.0]]
        energy_pred = [[0.1, 1.5, -0.1]]
        ```

        If we want the split the phoneme 1 (EH1) into 3 parts, starting at pitch = 2 and dropping to pitch = -2, we let
        ```
        d_split_factor = [[1, 3, 1]]
        mod_fns = {(0, 1): mod_fn_generator([2, -2], combine_fn=None)}  # {(batch_index, phoneme_index): mod_fn}

        pitch_new = [[0.2, 2.0, 0.0, -2.0, -1.0]]
        ```

        However, we need to maintain the same length of h, duration and energy if we expand phonemes to subphonemes.
        For duration, we have to pass `mod_fns` such that it splits durations by `d_split_factor`. 
        ```
        duration_new = [3.1, 2.0, 2.0, 2.0, 5.2]
        ```
        
        For h and energy, with `mod_fns=None`, we simply repeat them.
        ```
        hs_new = [[h0, h1, h1, h1, h2]]
        energy_new = [[0.1, 1.5, 1.5, 1.5, -0.1]] 
        ```

        :param preds:
        :param d_split_factor:
        :param mod_fns:
        :return:
        '''
        max_t_text = d_split_factor.sum(dim=-1).max().item()
        if preds.ndim == 3:  # expand h
            preds_new = torch.zeros((preds.size(0), max_t_text, preds.size(2)), dtype=preds.dtype, device=preds.device)
            for h, pred_new, d_split in zip(preds, preds_new, d_split_factor):
                start = 0
                end = 0
                for h_vec, split in zip(h, d_split):
                    end += split.item()
                    pred_new[start:end, :] = h_vec
                    start = end
        else:  # expand duration, pitch, energy
            if mod_fns is None:
                mod_fns = {}
            preds_new = torch.zeros((preds.size(0), max_t_text), dtype=preds.dtype, device=preds.device)
            for batch_i, (pred, pred_new, d_split) in enumerate(zip(preds, preds_new, d_split_factor)):
                start = 0
                end = 0
                for phone_i, (pred_val, split) in enumerate(zip(pred, d_split)):
                    end += split.item()
                    if (batch_i, phone_i) in mod_fns:
                        pred_new[start:end] = mod_fns[(batch_i, phone_i)](pred_val, split)
                    else:
                        pred_new[start:end] = pred_val
                    start = end
        return preds_new
