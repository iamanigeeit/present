import os
import re
from collections import OrderedDict

import soundfile as sf
import numpy as np
import torch
from phonemizer.backend import BACKENDS
from phonemizer.separator import Separator

from prosody.utils.utils import print_table, flatten_list
from prosody.utils.text import ARPA_VOWELS, ARPA_CONSONANTS

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

OVERALL_D_FACTOR = 1  # Hungarian speaking speed is faster

VOWELS_MAP = {
    'aː': 'AH2',
    'ˈaː': 'AA2',
    'ˌaː': 'AH1',

    'ɑ': ('AA0', '0.5'),
    'ˈɑ': ('AA1', '0.5'),
    'ˌɑ': ('AA2', '0.5'),
    'ˈɑː': 'AA1',

    'eː': 'EY0',
    'ˈeː': 'EY1',
    'ˌeː': 'EY2',
    'ɛ': 'EH0',
    'ˈɛ': 'EH1',
    'ˌɛ': 'EH2',

    'i': ('IY0', '0.5'),
    'ˈi': ('IY1', '0.5'),
    'ˌi': ('IY2', '0.5'),
    'iː': 'IY0',
    'ˈiː': 'IY1',
    'ˌiː': 'IY2',

    'o': ('OW0', '0.5'),
    'ˈo': ('OW1', '0.5'),
    'ˌo': ('OW2', '0.5'),
    'oː': 'OW0',
    'ˈoː': 'OW1',
    'ˌoː': 'OW2',
    #
    # 'ø': ('UH0', '1'),
    # 'ˈø': ('UH0', '1'),
    # 'ˌø': ('UH0', '1'),
    # 'øː': ('UH0', '1.5'),
    # 'ˈøː': ('UH0', '1.5'),
    # 'ˌøː': ('UH0', '1.5'),
    
    'ø': ('UH0', '1'),
    'ˈø': ('UH1', '1'),
    'ˌø': ('UH2', '1'),
    'øː': ('UH0', '1.5'),
    'ˈøː': ('UH1', '1.5'),
    'ˌøː': ('UH2', '1.5'),

    'u': ('UW0', '0.5'),
    'ˈu': ('UW1', '0.5'),
    'ˌu': ('UW2', '0.5'),
    'uː': ('UW0', '1'),
    'ˈuː': ('UW1', '1'),
    'ˌuː': ('UW2', '1'),

    'y': ('UH0 Y', '0 1'),
    'ˈy': ('UH1 Y', '0 1'),
    'ˌy': ('UH2 Y', '0 1'),
    'yː': ('UH0 Y', '1 0'),
    'ˈyː': ('UH1 Y', '1 0'),
    'ˌyː': ('UH2 Y', '1 0'),
}


CONSONANTS_MAP = {
    'b': ('B', '0.7'),
    'bː': ('B B', '0.5 0.5'),
    'c': ('T Y', '0.4 0.4'),
    'cː': ('T Y', '0.7 0.7'),
    'd': ('D', '0.7'),
    'dː': ('D D', '0.5 0.5'),
    'dzː': ('D Z', '0.7 0.7'),
    'dʒ': ('D Z JH', '0.3 0.3 0.3'),
    'f': ('F', '0.7'),
    'ɡ': ('G', '0.7'),
    'ɡː': ('G', '1.4'),
    'h': ('HH', '0.7'),
    'j': ('Y', '0.7'),
    'ɟ': ('G Y', '0.7 0'),
    'ɟː': ('D Y', '0.7 0.4'),
    'k': ('K', '0.7'),
    'kː': ('K K', '0.7 0.7'),
    'l': ('L', '0.7'),
    'ʎ': ('L', '0.7'),  # only occurs before j anyway
    'm': ('M', '0.7'),
    'n': ('N', '0.7'),
    'ɲ': ('N Y', '0.8 0.2'),
    'p': ('P', '0.7'),
    'pː': ('P P', '0.7 0.7'),
    'r': ('R', '0.7'),
    's': ('S', '0.7'),
    'ʃ': ('SH', '0.7'),
    't': ('T', '0.7'),
    'tː': ('T T', '0.5 0.5'),
    'ts': ('T S', '0.4 0.4'),
    'tsː': ('T S', '0.7 0.7'),
    'tʃ': ('CH', '0.7'),
    'tʃː': ('CH CH', '0 1'),
    'v': ('V', '0.7'),
    'z': ('Z', '0.7'),
    'ʒ': ('ZH', '0.7'),
}


PHONES_MAP = VOWELS_MAP | CONSONANTS_MAP | {'|': '|'}  # word separator is important!

CUSTOM_ARPA_SUBS = OrderedDict()  # old_arpa_str: (new_arpa_str, duration_str, energy_str)
# # '=x' in duration / energy means RETAIN the d_factor or e_factor from position x

# Make vowels sound out clearly instead of merging with the next vowel
VOWELS_LIST = sorted(ARPA_VOWELS)
# UNSTRESSED_VOWELS = [v for v in VOWELS_LIST if v.endswith('0')]
# STRESSED_VOWELS = [v for v in VOWELS_LIST if v.endswith('1')]
# W_POST_VOWELS = set()
# Y_POST_VOWELS = set()
# W_PRE_VOWELS = set()
# Y_PRE_VOWELS = set()
# for v in VOWELS_LIST:
#     vowel = v[:2]
#     if vowel in ['UW', 'OW']:
#         W_POST_VOWELS.add(vowel)
#     elif vowel in ['IY', 'AY', 'EY', 'EH']:
#         Y_POST_VOWELS.add(vowel)
# for v in VOWELS_LIST:
#     vowel = v[:2]
#     if vowel == 'UW':
#         W_PRE_VOWELS.add(vowel)
#     elif vowel == 'IY':
#         Y_PRE_VOWELS.add(vowel)
#
# for unsd_vowel1 in UNSTRESSED_VOWELS:
#     for stressed_vowel in STRESSED_VOWELS:
#         uv1 = unsd_vowel1[:2]
#         sv = stressed_vowel[:2]
#         if uv1 in W_POST_VOWELS or sv in W_PRE_VOWELS:
#             bridge1 = 'W'
#             pause1 = '0.4'
#         elif uv1 in Y_POST_VOWELS or sv in Y_PRE_VOWELS:
#             bridge1 = 'Y'
#             pause1 = '0.4'
#         else:
#             bridge1 = ','
#             pause1 = '0'
#         CUSTOM_ARPA_SUBS[f'{unsd_vowel1} {stressed_vowel}'] = (
#             f'{unsd_vowel1} {bridge1} {stressed_vowel}',
#             f'=0 {pause1} =1',
#             '=0 0 1',
#             '=0 0 0.5'
#         )
#
#         if sv in W_POST_VOWELS or uv1 in W_PRE_VOWELS:
#             bridge1 = 'W'
#             pause1 = '0.4'
#         elif sv in Y_POST_VOWELS or uv1 in Y_PRE_VOWELS:
#             bridge1 = 'Y'
#             pause1 = '0.4'
#         else:
#             bridge1 = ','
#             pause1 = '0'
#         CUSTOM_ARPA_SUBS[f'{stressed_vowel} {unsd_vowel1}'] = (
#             f'{stressed_vowel} {bridge1} {unsd_vowel1}',
#             f'=0 {pause1} =1',
#             '1 0 =1',
#             '0.5 0 =1'
#         )
#
#         for unsd_vowel2 in UNSTRESSED_VOWELS:
#             uv2 = unsd_vowel2[:2]
#             if sv in W_POST_VOWELS or uv2 in W_PRE_VOWELS:
#                 bridge2 = 'W'
#                 pause2 = '0.4'
#             elif sv in Y_POST_VOWELS or uv2 in Y_PRE_VOWELS:
#                 bridge2 = 'Y'
#                 pause2 = '0.4'
#             else:
#                 bridge2 = ','
#                 pause2 = '0'
#             CUSTOM_ARPA_SUBS[f'{unsd_vowel1} {stressed_vowel} {unsd_vowel2}'] = (
#                 f'{unsd_vowel1} {bridge1} {stressed_vowel} {bridge2} {unsd_vowel2}',
#                 f'1 {pause1} =1 {pause2} 1',
#                 '=0 0 1 0 =2',
#                 '1 0 0.5 0 1'
#             )
#
for arpa_vowel in VOWELS_LIST + ['W', 'Y']:
    # Prevent initial vowel from sticking to previous word
    CUSTOM_ARPA_SUBS[f'| {arpa_vowel}'] = (
        f', {arpa_vowel}', '0.1 =1',
        # '0 =1', '0 =1'
    )
    CUSTOM_ARPA_SUBS[f'{arpa_vowel} |'] = (
        f'{arpa_vowel} ,', '=0 0',
        # '0 =1', '0 =1'
    )

for v1 in VOWELS_LIST:
    for v2 in VOWELS_LIST:
        CUSTOM_ARPA_SUBS[f'{v1} | {v2}'] = (
            f'{v1} , {v2}', '=0 0.2 =1'
        )

# ɟ at word end should be more like G HH
CUSTOM_ARPA_SUBS['G Y |'] = (
        'G HH ,', '0.4 0.4 0',
        # '0 =1', '0 =1'
    )

# Make g at word end clearer
CUSTOM_ARPA_SUBS['G |'] = (
        'G |', '1 0',
        # '0 =1', '0 =1'
    )

# This comes from Hungarian <lly>
CUSTOM_ARPA_SUBS['Y Y Y'] = (
    'L Y', '0 1'
)


for v in VOWELS_LIST + ['L', 'R']:
    for c in ['P', 'T', 'K', 'B', 'D', 'G']:
        # Separate P T K before/after vowels
        # CUSTOM_ARPA_SUBS[f'{v} | {c} '] = (
        #     f'{v} , {c}', '=0 0.1 =1',  # '=0 0 =1 =2', '=0 0 =1 =2'
        # )
        CUSTOM_ARPA_SUBS[f'{c} | {v} '] = (
            f'{c} , {v}', '=0 0 =1',  # '=0 0 =1 =2', '=0 0 =1 =2'
        )

# for v1 in VOWELS_LIST:
#     for v2 in VOWELS_LIST:
#         # Prevent devoicing of P T K between vowels
#         CUSTOM_ARPA_SUBS[f'{v1} P {v2}'] = (
#             f'{v1} P P {v2}', '=0 0.3 0.3 =2',
#             # '=0 0 =1 =2', '=0 0 =1 =2'
#         )
#         CUSTOM_ARPA_SUBS[f'{v1} T {v2}'] = (
#             f'{v1} T T {v2}', '=0 0.3 0.3 =2',
#             # '=0 0 =1 =2', '=0 0 =1 =2'
#         )
#         CUSTOM_ARPA_SUBS[f'{v1} K {v2}'] = (
#             f'{v1} K K {v2}', '=0 0.3 0.3 =2',
#             # '=0 0 =1 =2', '=0 0 =1 =2'
#         )

# error where consonant + AA1 R is treated as consonant | AA1 R (due to common English <... are ...>)
for arpa_consonant in sorted(ARPA_CONSONANTS):
    CUSTOM_ARPA_SUBS[f'{arpa_consonant} AA1 R |'] = (
        f', {arpa_consonant} AA2 R', '0 =0 =1 =2' #, '0 0 0.5 0', '0 =0 0.5 0'
    )
#
# # error where N UW is treated as N Y UW (due to English <new>?)
# for i in range(3):
#     CUSTOM_ARPA_SUBS[f'N UW{i}'] = (
#         (f'N UW{i} W', '=0 0 =1', '=0 0 =1', '=0 0 =1')
#     )
#
# error where final e gets swallowed by consonant
# CUSTOM_ARPA_SUBS['EH0 |'] = (
#     'EY0 |', '=0 =1',
#     # '=0 =1', '0 =1'
# )
for arpa_vowel in VOWELS_LIST:
    CUSTOM_ARPA_SUBS[f'EH0 | {arpa_vowel}'] = (
        f'EH2 , {arpa_vowel}', '0.7 0 =2',
        # '=0 0 =2', '0 0 =2'
    )


G2P = BACKENDS['espeak'](
    language='hu',
    preserve_punctuation=True,
    with_stress=True,
)
SEP = Separator(word='|', phone=' ')
PUNCS = ',.?!'
PUNC_REPLACE = {
    ';:—': ',',
    '…': '.',
    '¡¿"«»“”': '',
}

# def custom_ipa_subs(phones_tuple):
#     phones = list(phones_tuple)
#     return phones

ALLOWED_IPA = set(''.join(PHONES_MAP.keys()))


def phonemize(text):
    phonestr: str = G2P.phonemize([text.replace('-', ' ')], separator=SEP)[0]
    for punc, new_punc in PUNC_REPLACE.items():
        phonestr = re.sub(rf'[{punc}]', new_punc, phonestr)
    phonestr = phonestr.replace('|', ' | ')
    # for old_ipa, new_ipa in IPA_REPLACE.items():
    #     phonestr = phonestr.replace(old_ipa, new_ipa)

    phones = phonestr.strip().split()

    clean = []
    for phone in phones:
        split_pos = [0]
        for pos, ipa in enumerate(phone):
            if ipa in PUNCS:
                split_pos.append(pos)
                split_pos.append(pos+1)
        split_pos.append(len(phone))
        if split_pos:
            for start, end in zip(split_pos[:-1], split_pos[1:]):
                if start < end:
                    clean.append(phone[start:end])
    # ipas = [IPA_REPLACE[x] if x in IPA_REPLACE else x for x in clean]
    # ipas = custom_ipa_subs(tuple(ipas))

    return clean


class HungarianArpaSpeech:

    def __init__(
            self,
            tts_inference_fn,
            token_id_converter,
            phones_map=None,
            custom_arpa_subs=None,
    ):
        self.tts_inference_fn = tts_inference_fn
        self.token_id_converter = token_id_converter
        self.phones_to_arpa, self.phones_to_duration = {}, {}
        # self.phones_to_pitch, self.phones_to_energy = {}, {}
        self.update_arpa(PHONES_MAP if phones_map is None else phones_map)
        self.custom_arpa_subs, self.subs_minlen, self.subs_maxlen = {}, 9999, 0
        self.update_custom_arpa_subs(CUSTOM_ARPA_SUBS if custom_arpa_subs is None else custom_arpa_subs)

    def update_arpa(self, phones_map, verbose=False):
        for phone, mapping in phones_map.items():
            if isinstance(mapping, str):
                mapping = (mapping,)
            arpa_str = mapping[0]
            arpa = arpa_str.split(' ')
            if phone in self.phones_to_arpa:
                del self.phones_to_arpa[phone]
                if phone in self.phones_to_duration:
                    del self.phones_to_duration[phone]
                # if phone in self.phones_to_pitch:
                #     del self.phones_to_pitch[phone]
                # if phone in self.phones_to_energy:
                #     del self.phones_to_energy[phone]
            self.phones_to_arpa[phone] = arpa
            if len(mapping) > 1:
                duration_str = mapping[1]
                durations = [float(x) for x in duration_str.split(' ')]
                assert len(arpa) == len(durations), f'Mismatched arpa and duration: {arpa_str} {duration_str}'
                self.phones_to_duration[phone] = durations
                # if len(mapping) > 2:
                #     pitch_str = mapping[2]
                #     pitches = [float(x) for x in pitch_str.split(' ')]
                #     assert len(arpa) == len(pitches), f'Mismatched arpa and pitch: {arpa_str} {pitch_str}'
                #     self.phones_to_pitch[phone] = pitches
                #     if len(mapping) > 3:
                #         energy_str = mapping[3]
                #         energies = [float(x) for x in energy_str.split(' ')]
                #         assert len(arpa) == len(energies), f'Mismatched arpa and energy: {arpa_str} {energy_str}'
                #         self.phones_to_energy[phone] = energies
        if verbose:
            print(f'Updated {", ".join(phones_map.keys())}')

    def update_custom_arpa_subs(self, custom_arpa_subs, verbose=False):
        # for old_arpa_str, (new_arpa_str, duration_str, pitch_str, energy_str) in custom_arpa_subs.items():
        for old_arpa_str, (new_arpa_str, duration_str) in custom_arpa_subs.items():
            old_arpa = tuple(old_arpa_str.split())
            new_arpa = new_arpa_str.split(' ')
            durations = [int(x[1:]) if x.startswith('=') else float(x) for x in duration_str.split(' ')]
            assert len(new_arpa) == len(durations), f'Mismatched arpa and duration: {new_arpa_str} {duration_str}'
            # pitches = [int(x[1:]) if x.startswith('=') else float(x) for x in pitch_str.split(' ')]
            # assert len(new_arpa) == len(pitches), f'Mismatched arpa and pitch: {new_arpa_str} {pitch_str}'
            # energies = [int(x[1:]) if x.startswith('=') else float(x) for x in energy_str.split(' ')]
            # assert len(new_arpa) == len(energies), f'Mismatched arpa and energy: {new_arpa_str} {energy_str}'
            self.custom_arpa_subs[old_arpa] = (
                new_arpa, durations,
                # pitches, energies
            )

        for old_arpa in self.custom_arpa_subs:
            sub_len = len(old_arpa)
            if sub_len < self.subs_minlen:
                self.subs_minlen = sub_len
            if sub_len > self.subs_maxlen:
                self.subs_maxlen = sub_len

        if verbose:
            print(f'Updated {", ".join(custom_arpa_subs.keys())}')

    def convert_phones_arpa(self, phones):
        arpas = []
        d_factor = []
        # p_factor = []
        # e_factor = []
        for phone in phones:
            if phone in PUNCS:
                arpas.append(phone)
                d_factor.append(1.0)
                # p_factor.append(0.0)
                # e_factor.append(0.0)
                continue

            arpa = self.phones_to_arpa[phone]
            arpas.extend(arpa)
            if phone in self.phones_to_duration:
                d_fac = self.phones_to_duration[phone]
            else:
                d_fac = [1.0] * len(arpa)
            d_factor.extend(d_fac)
            # if phone in self.phones_to_pitch:
            #     p_factor.extend(self.phones_to_pitch[phone])
            # else:
            #     p_factor.extend([0.0] * len(arpa))
            # if phone in self.phones_to_energy:
            #     e_factor.extend(self.phones_to_energy[phone])
            # else:
            #     e_factor.extend([0.0] * len(arpa))  # e_factor is added, not multiplied
        # e_factor = [0.0] * len(d_factor)
        # if arpas[0] == 'H':
        #     e_factor[0] = 1
        if self.subs_maxlen:  # apply custom_arpa_subs
            arpas.append('|')
            d_factor.append(1.0)
            # e_factor.insert(0, 0.0)
            # p_factor.insert(0, 0.0)
            for sub_len in range(self.subs_maxlen, self.subs_minlen - 1, -1):
                arpas_tup = tuple(arpas)
                i = 0
                while i < len(arpas_tup) - sub_len:
                    sub_arpa = arpas_tup[i:i+sub_len]
                    if sub_arpa in self.custom_arpa_subs:
                        # new_arpa, durations, pitches, energies = self.custom_arpa_subs[sub_arpa]
                        new_arpa, durations = self.custom_arpa_subs[sub_arpa]
                        arpas[i] = new_arpa
                        arpas[i+1:i+sub_len] = [[]] * (sub_len - 1)
                        d_factor[i] = [d_factor[i + d] if isinstance(d, int) else d for d in durations]
                        d_factor[i+1:i+sub_len] = [[]] * (sub_len - 1)
                        # p_factor[i] = [p_factor[i + p] if isinstance(p, int) else p for p in pitches]
                        # p_factor[i + 1:i + sub_len] = [[]] * (sub_len - 1)
                        # e_factor[i] = [e_factor[i + e] if isinstance(e, int) else e for e in energies]
                        # e_factor[i+1:i+sub_len] = [[]] * (sub_len - 1)
                        i += sub_len - 1
                    i += 1
                arpas = flatten_list(arpas)
                d_factor = flatten_list(d_factor)
                # p_factor = flatten_list(p_factor)
                # e_factor = flatten_list(e_factor)

        arpas = np.array(arpas)
        not_wordseps = arpas != '|'
        arpas = arpas[not_wordseps]
        d_factor = np.array(d_factor, dtype=np.float32)[not_wordseps]
        p_factor = np.zeros(d_factor.shape, dtype=np.float32)
        e_factor = np.zeros(d_factor.shape, dtype=np.float32)
        # p_factor = np.array(p_factor, dtype=np.float32)[not_wordseps]
        # e_factor = np.array(e_factor, dtype=np.float32)[not_wordseps]

        return arpas, d_factor, p_factor, e_factor

    def gen_inputs(self, hungarian, verbose=False, overall_d_factor=OVERALL_D_FACTOR, device=DEVICE):
        if isinstance(hungarian, str):
            ipas = phonemize(hungarian)
        elif isinstance(hungarian, list):
            ipas = hungarian
        else:
            raise NotImplementedError('hungarian must be string of Hungarian text or list of regularized Hungarian phones')
        arpas, d_factor, p_factor, e_factor = self.convert_phones_arpa(ipas)
        if verbose:
            print_table(arpas=arpas, d_factor=d_factor, p_factor=p_factor, e_factor=e_factor)
        d_factor = torch.tensor(d_factor, device=device).unsqueeze(0) * overall_d_factor
        p_factor = torch.tensor(p_factor, device=device).unsqueeze(0)
        e_factor = torch.tensor(e_factor, device=device).unsqueeze(0)
        return arpas, d_factor, p_factor, e_factor


    def gen_audio(
            self,
            hungarian,
            save_dir,
            inputs=None,
            phones_map=None,
            custom_arpa_subs=None,
            infer_overrides=None,
            arpa_in_filename=False,
            custom_filename='',
            verbose=False,
            device=DEVICE,
    ):
        if phones_map is not None:
            self.update_arpa(phones_map, verbose=verbose)
        if custom_arpa_subs is not None:
            self.update_custom_arpa_subs(custom_arpa_subs, verbose=verbose)

        if inputs is None:
            arpas, d_factor, p_factor, e_factor = self.gen_inputs(hungarian, verbose=verbose, device=device)
        else:
            arpas, d_factor, p_factor, e_factor = inputs

        infer_kwargs = {
            'd_factor': d_factor,
            'p_factor': p_factor,
            'e_factor': e_factor,
            'd_split_factor': None,
            'd_mod_fns': None,
            'p_mod_fns': None,
            'e_mod_fns': None,
        }
        if infer_overrides is not None:
            for x in infer_overrides.keys():
                infer_kwargs[x] = infer_overrides[x]
        phone_ids = torch.tensor(self.token_id_converter.tokens2ids(arpas), dtype=torch.int32, device=device)
        with torch.no_grad():
            wav_modified, _ = self.tts_inference_fn(
                    text=phone_ids.unsqueeze(0), text_lengths=torch.tensor([len(phone_ids)], device=device),
                    verbose=verbose,
                    **infer_kwargs,
                )
            if custom_filename:
                filename = custom_filename
            else:
                filename = ''.join(hungarian) if isinstance(hungarian, list) else hungarian
                if arpa_in_filename:
                    filename = f'{filename}-{"".join(arpas)}'
                filename = f'{filename}.wav'
            save_path = save_dir / filename
            os.makedirs(save_path.parent, exist_ok=True)
            sf.write(save_path, wav_modified.squeeze().cpu().numpy(), 22050, "PCM_16")
        return arpas, d_factor, p_factor, e_factor

