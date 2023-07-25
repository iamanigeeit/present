#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""FS2 Inference with prosody."""

from typing import Dict, Optional, Callable, Tuple, Sequence

import torch
import torch.nn.functional as F

from espnet2.tts.fastspeech2.fastspeech2 import FastSpeech2
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from prosody.transfer_prosody import transfer_prosody
from prosody.utils.utils import expand_embed_duration


class FS2Prosody(FastSpeech2):

    def inference(
        self,
        text: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        spembs: torch.Tensor = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        use_teacher_forcing: bool = False,
        d_factor: Optional[torch.Tensor] = None,
        p_factor: Optional[torch.Tensor] = None,
        e_factor: Optional[torch.Tensor] = None,
        d_split_factor: Optional[torch.Tensor] = None,
        d_mod_fns: Optional[Callable] = None,
        p_mod_fns: Optional[Callable] = None,
        e_mod_fns: Optional[Callable] = None,
        transfer_prosody_args: Optional[dict] = None,
        force_prosody_args: Optional[dict] = None,
        p_ref_weight: Optional[float] = None,
        e_ref_weight: Optional[float] = None,
        change_durations: bool = False,
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T_text,).
            feats (Optional[Tensor): Feature sequence to extract style (N, idim).
            durations (Optional[Tensor): Groundtruth of duration (T_text + 1,).
            spembs (Optional[Tensor): Speaker embedding vector (spk_embed_dim,).
            sids (Optional[Tensor]): Speaker ID (1,).
            lids (Optional[Tensor]): Language ID (1,).
            pitch (Optional[Tensor]): Groundtruth of token-avg pitch (T_text + 1, 1).
            energy (Optional[Tensor]): Groundtruth of token-avg energy (T_text + 1, 1).
            alpha (float): Alpha to control the speed.
            use_teacher_forcing (bool): Whether to use teacher forcing.
                If true, groundtruth of duration, pitch and energy will be used.

        Returns:
            Dict[str, Tensor]: Output dict including the following items:
                * feat_gen (Tensor): Output sequence of features (T_feats, odim).
                * duration (Tensor): Duration sequence (T_text + 1,).
                * pitch (Tensor): Pitch sequence (T_text + 1,).
                * energy (Tensor): Energy sequence (T_text + 1,).

        """
        x, y = text, feats
        spemb, d, p, e = spembs, durations, pitch, energy

        # add eos at the last of sequence
        x = F.pad(x, [0, 1], "constant", self.eos)

        # setup batch axis
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs, ys = x.unsqueeze(0), None
        if y is not None:
            ys = y.unsqueeze(0)
        if spemb is not None:
            spembs = spemb.unsqueeze(0)

        if use_teacher_forcing:
            # use groundtruth of duration, pitch, and energy
            ds, ps, es = d.unsqueeze(0), p.unsqueeze(0), e.unsqueeze(0)
            _, outs, d_outs, p_outs, e_outs = self._forward(
                xs,
                ilens,
                ys,
                ds=ds,
                ps=ps,
                es=es,
                spembs=spembs,
                sids=sids,
                lids=lids,
            )  # (1, T_feats, odim)
        else:
            _, outs, d_outs, p_outs, e_outs = self._forward(
                xs,
                ilens,
                ys,
                spembs=spembs,
                sids=sids,
                lids=lids,
                is_inference=True,
                alpha=alpha,
                d_factor=d_factor,
                p_factor=p_factor,
                e_factor=e_factor,
                d_split_factor=d_split_factor,
                d_mod_fns=d_mod_fns,
                p_mod_fns=p_mod_fns,
                e_mod_fns=e_mod_fns,
                transfer_prosody_args=transfer_prosody_args,
                force_prosody_args=force_prosody_args,
                p_ref_weight=p_ref_weight,
                e_ref_weight=e_ref_weight,
                change_durations=change_durations,
                verbose=verbose,
            )  # (1, T_feats, odim)

        return dict(
            feat_gen=outs[0],
            duration=d_outs[0],
            pitch=p_outs[0],
            energy=e_outs[0],
        )

    def _forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        ys: Optional[torch.Tensor] = None,
        olens: Optional[torch.Tensor] = None,
        ds: Optional[torch.Tensor] = None,
        ps: Optional[torch.Tensor] = None,
        es: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        is_inference: bool = False,
        alpha: float = 1.0,
        d_factor: Optional[torch.Tensor] = None,
        p_factor: Optional[torch.Tensor] = None,
        e_factor: Optional[torch.Tensor] = None,
        d_split_factor: Optional[torch.Tensor] = None,
        d_mod_fns: Optional[Callable] = None,
        p_mod_fns: Optional[Callable] = None,
        e_mod_fns: Optional[Callable] = None,
        transfer_prosody_args: Optional[dict] = None,
        force_prosody_args: Optional[dict] = None,
        p_ref_weight: Optional[float] = None,
        e_ref_weight: Optional[float] = None,
        change_durations: bool = False,
        verbose: bool = False,
    ) -> Sequence[torch.Tensor]:
        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)  # (B, T_text, adim)

        # integrate with GST
        if self.use_gst:
            style_embs = self.gst(ys)
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

        # forward duration predictor and variance predictors
        h_masks = make_pad_mask(ilens).to(xs.device)

        if self.stop_gradient_from_pitch_predictor:
            p_outs = self.pitch_predictor(hs.detach(), h_masks.unsqueeze(-1))
        else:
            p_outs = self.pitch_predictor(hs, h_masks.unsqueeze(-1))
        if self.stop_gradient_from_energy_predictor:
            e_outs = self.energy_predictor(hs.detach(), h_masks.unsqueeze(-1))
        else:
            e_outs = self.energy_predictor(hs, h_masks.unsqueeze(-1))

        if is_inference:
            d_outs = self.duration_predictor(hs, h_masks)  # (B, T_text)
            d_outs = d_outs.exp() - self.duration_predictor.offset  # (B, T_text)
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
                d_outs = expand_embed_duration(d_outs, d_split_factor, d_mod_fns)
                p_outs = expand_embed_duration(pitch_pred, d_split_factor, p_mod_fns).unsqueeze(-1)
                e_outs = expand_embed_duration(energy_pred, d_split_factor, e_mod_fns).unsqueeze(-1)
                hs = expand_embed_duration(hs, d_split_factor)
            elif p_ref_weight is not None and e_ref_weight is not None:
                if transfer_prosody_args is not None:
                    duration_pred = d_outs.squeeze()
                    pitch_pred = p_outs.squeeze()
                    energy_pred = e_outs.squeeze()
                    duration_new, pitch_new, energy_new = transfer_prosody(
                        duration_pred, pitch_pred, energy_pred, **transfer_prosody_args
                    )
                    d_outs = duration_new[None]
                    pitch_new = pitch_new[None].unsqueeze(-1)
                    energy_new = energy_new[None].unsqueeze(-1)
                elif force_prosody_args is not None:
                    if 'duration_new' in force_prosody_args:
                        d_outs = force_prosody_args['duration_new']
                    if 'pitch_new' in force_prosody_args:
                        pitch_new = force_prosody_args['pitch_new']
                    else:
                        pitch_new = p_outs
                    if 'energy_new' in force_prosody_args:
                        energy_new = force_prosody_args['energy_new']
                    else:
                        energy_new = e_outs
                else:
                    raise ValueError

                p_outs = p_outs * (1.0 - p_ref_weight) + pitch_new * p_ref_weight
                e_outs = e_outs * (1.0 - e_ref_weight) + energy_new * e_ref_weight

            d_outs = torch.clamp(torch.round(d_outs), min=0).long()

            p_embs = self.pitch_embed(p_outs.transpose(1, 2)).transpose(1, 2)
            e_embs = self.energy_embed(e_outs.transpose(1, 2)).transpose(1, 2)
            hs = hs + e_embs + p_embs
            hs = self.length_regulator(hs, d_outs)
            if verbose:
                print('Duration (new):', d_outs.squeeze())
                print('Pitch (new):', p_outs.squeeze())
                print('Energy (new):', e_outs.squeeze())
        else:
            d_outs = self.duration_predictor(hs, h_masks)
            # use groundtruth in training
            p_embs = self.pitch_embed(ps.transpose(1, 2)).transpose(1, 2)
            e_embs = self.energy_embed(es.transpose(1, 2)).transpose(1, 2)
            hs = hs + e_embs + p_embs
            hs = self.length_regulator(hs, ds)  # (B, T_feats, adim)


        # forward decoder
        if olens is not None and not is_inference:
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None
        zs, _ = self.decoder(hs, h_masks)  # (B, T_feats, adim)
        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )  # (B, T_feats, odim)

        # postnet -> (B, T_feats//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        return before_outs, after_outs, d_outs, p_outs, e_outs