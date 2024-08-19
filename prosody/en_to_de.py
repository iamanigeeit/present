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

LONG_D_FACTOR = 1.5
LONG_MARK = 'ː'

VOWELS_MAP = {
    'a': 'AH1',
    'ˌa': ('AA2', '0.5'),
    'ˈa': ('AA1', '0.5'),
    'ɐ': 'AH1',  # special cases to be handled in CUSTOM_ARPA_SUBS

    'aɪ': ('AY0', '0.7'),
    'ˌaɪ': ('AY2', '0.7'),
    'ˈaɪ': ('AY1', '0.7'),

    'aʊ': 'AA0 W',
    'ˌaʊ': 'AA2 W',
    'ˈaʊ': 'AA1 W',

    'ɑ': ('AA0', '0.7'),
    'ˌɑ': ('AA2', '0.7'),
    'ˈɑ': ('AA1', '0.7'),

    'e': ('EY0', '0.2'),
    'ˌe': ('EY2', '0.5'),
    'ˈe': ('EY1', '0.5'),

    'ɛ': ('EH2', '0.7'),
    'ˌɛ': 'EH2',
    'ˈɛ': 'EH1',

    'ə': ('AH0', '1.5', '1'),
    'ˌə': ('AH0', '1.5', '2'),

    'ɜ': ('AH1', '0.5'),

    'i': ('IY0', '0.7'),
    'ˌi': ('IY2', '0.7'),
    'ˈi': ('IY1', '0.7'),

    'ɪ': 'IH0',
    'ˌɪ': 'IH2',
    'ˈɪ': 'IH1',

    'o': ('OW0', '0.7'),
    'ˌo': ('OW2', '0.7'),
    'ˈo': ('OW1', '0.7'),

    'ø': 'UW0',
    'ˌø': 'UW2',
    'ˈø': 'UW1',

    'œ': ('W EH0', '0 1'),
    'ˌœ': ('W EH2', '0 1'),
    'ˈœ': ('W EH1', '0 1'),

    'ɔ': 'AO0',
    'ˌɔ': 'AO2',
    'ˈɔ': 'AO1',

    'ɔø': 'OY0',
    'ˌɔø': 'OY1',
    'ˈɔø': 'OY2',

    'u': ('W UW0', '0 0.7'),
    'ˌu': ('W UW2', '0 0.7'),
    'ˈu': ('W UW1', '0 0.7'),

    'ʊ': ('W UH0', '0 1'),
    'ˌʊ': ('W UH2', '0 1'),
    'ˈʊ': ('W UH1', '0 1'),

    'y': ('Y UH0', '0 1', '0 0'),
    'ˌy': ('Y UH2', '0 1', '0 0'),
    'ˈy': ('Y UH1', '0 1', '0 0'),
}

CONSONANTS_MAP = {
    'b': 'B',
    'ç': ('H SH S', '0 1 0'),
    'd': 'D',
    'dʒ': 'JH',
    'f': 'F',
    'ɡ': 'G',
    'h': 'HH',
    'j': 'Y',
    'k': 'K',
    'l': 'L',
    'm': 'M',
    'n': 'N',
    'ŋ': 'NG',
    'p': 'P',
    'pf': ('P F', '0.5 1'),
    'ʁ': 'R',
    's': 'S',
    'ʃ': 'SH',
    't': 'T',
    'ts': 'T S',
    'tʃ': 'CH',
    'v': 'V',
    'x': ('HH K HH', '1 0 1', '0 0 0'),
    'z': 'Z',
    'ʒ': 'ZH',
}

PHONES_MAP = VOWELS_MAP | CONSONANTS_MAP | {'|': '|'}  # word separator is important!

CUSTOM_ARPA_SUBS = OrderedDict()  # old_arpa_str: (new_arpa_str, duration_str, energy_str)
# '=x' in duration / energy means RETAIN the d_factor or e_factor from position x

# Makes /ə/ sound out clearly instead of merging with the next vowel / consonant
CUSTOM_ARPA_SUBS[f'AH0 |'] = (f'AH0 ,', '=0 0', '=0 0')
CUSTOM_ARPA_SUBS[f'AH1 |'] = (f'AH1 ,', '=0 0', '=0 0')
CUSTOM_ARPA_SUBS[f'AH2 |'] = (f'AH2 ,', '=0 0', '=0 0')

for arpa_vowel in sorted(ARPA_VOWELS) + ['W', 'Y']:
    # Prevent initial vowel from sticking to previous word
    CUSTOM_ARPA_SUBS[f'| {arpa_vowel}'] = (f', {arpa_vowel}', '0 =1', '0 =1')

#     CUSTOM_ARPA_SUBS[f'AH0 | {arpa_vowel}'] = (f'AH0 , {arpa_vowel}', '2 0 =2', '1 0 =2')
#     CUSTOM_ARPA_SUBS[f'AH0 {arpa_vowel}'] = (f'AH0 , {arpa_vowel}', '2 0.2 =1', '1 0 =1')
#     CUSTOM_ARPA_SUBS[f'AH1 | {arpa_vowel}'] = (f'AH1 , {arpa_vowel}', '1 0 =2', '=0 0 =2')
#     CUSTOM_ARPA_SUBS[f'AH2 | {arpa_vowel}'] = (f'AH2 , {arpa_vowel}', '1 0 =2', '=0 0 =2')

    # For vowel + /ɐ/: use AH0 for /ɐ/ and stop it from sticking to next word
    CUSTOM_ARPA_SUBS[f'{arpa_vowel} AH1 |'] = (f'{arpa_vowel} AH0 ,', '=0 1 0', '=0 =1 0')
    # if arpa_vowel.endswith('0') and arpa_vowel != 'AH0':
    if arpa_vowel != 'AH0':
        CUSTOM_ARPA_SUBS[f'{arpa_vowel} AH1'] = (f'{arpa_vowel} AH0', '=0 1', '=0 =1')

# Force /t/ to be pronounced between vowels and not as /d/
for pre_vowel in sorted(ARPA_VOWELS):
    for post_vowel in sorted(ARPA_VOWELS):
        CUSTOM_ARPA_SUBS[f'{pre_vowel} T {post_vowel}'] = (f'{pre_vowel} T {post_vowel}', '=0 1.5 =2', '=0 =1 =2')

# Force final /x/ to be separate from the next word
CUSTOM_ARPA_SUBS[f'HH K HH |'] = (f'HH K HH ,', '=0 =1 =2 0', '=0 =1 =2 0')

# weird error where AH0 is pronounced as EY before full stop
CUSTOM_ARPA_SUBS['AH0 | .'] = ('AH1 | .', '0.5 0 2', '=0 0 0')
# error where consonant + AH0 N is treated as consonant | AH0 N (due to common English ... and...)
for arpa_consonant in sorted(ARPA_CONSONANTS):
    CUSTOM_ARPA_SUBS[f'{arpa_consonant} AH0 N |'] = (f'{arpa_consonant} AH0 N ,', '1 1 1 0', '0 0 0 0')

# # error where R AH0 comes out as R AH1 and not R ER0
# CUSTOM_ARPA_SUBS['R AH0 |'] = ('R ER0', '1 1', '0 -0.5')


G2P = BACKENDS['espeak'](
    language='de',
    preserve_punctuation=True,
    with_stress=True
)
SEP = Separator(word='|', phone=' ')
PUNCS = ',.?!'
PUNC_REPLACE = {
    ';:—': ',',
    '…': '.',
    '¡¿"«»“”': '',
}
# For phoneme accuracy
IPA_REPLACE = {
    'ɾ': 'ɐ',
    'r': 'ʁ',
}


def custom_ipa_subs(phones_tuple):
    phones = list(phones_tuple)
    # Consonant + r should be ʁ not ɐ
    i = 0
    while i < len(phones):
        if phones[i] in CONSONANTS_MAP:
            if phones[i+1] == 'ɐ':
                if phones[i+2].replace(LONG_MARK, '') in VOWELS_MAP:
                    phones[i+1] = 'ʁ'
                    i += 2
        i += 1
    return phones


ALLOWED_IPA = {LONG_MARK} | set(''.join(PHONES_MAP.keys()))


def phonemize(text):
    phonestr: str = G2P.phonemize([text], separator=SEP)[0]
    for punc, new_punc in PUNC_REPLACE.items():
        phonestr = re.sub(rf'[{punc}]', new_punc, phonestr)
    # correction for espeak issue where ʊɾ becomes ??
    phonestr = re.sub(rf'([ˌˈ ])\?\?', r'\1ʊ ɾ', phonestr)  # hopefully no weird ?? floating around
    phones = phonestr.replace('|', ' | ').strip().split()

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
    ipas = [IPA_REPLACE[x] if x in IPA_REPLACE else x for x in clean]
    ipas = custom_ipa_subs(tuple(ipas))

    # Sometimes there are weird errors like random number 1
    # Or even (en) marks even though it's German like "innewohnte"
    # Those have to be manually removed
    return ipas


class GermanArpaSpeech:

    def __init__(
            self,
            tts_inference_fn,
            token_id_converter,
            phones_map=None,
            custom_arpa_subs=None,
    ):
        self.tts_inference_fn = tts_inference_fn
        self.token_id_converter = token_id_converter
        self.phones_to_arpa, self.phones_to_duration, self.phones_to_energy = {}, {}, {}
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
                if phone in self.phones_to_energy:
                    del self.phones_to_energy[phone]
            self.phones_to_arpa[phone] = arpa
            if len(mapping) > 1:
                duration_str = mapping[1]
                durations = [float(x) for x in duration_str.split(' ')]
                assert len(arpa) == len(durations), f'Mismatched arpa and duration: {arpa_str} {duration_str}'
                self.phones_to_duration[phone] = durations
                if len(mapping) > 2:
                    energy_str = mapping[2]
                    energies = [float(x) for x in energy_str.split(' ')]
                    assert len(arpa) == len(energies), f'Mismatched arpa and energy: {arpa_str} {energy_str}'
                    self.phones_to_energy[phone] = energies
        if verbose:
            print(f'Updated {", ".join(phones_map.keys())}')

    def update_custom_arpa_subs(self, custom_arpa_subs, verbose=False):
        for old_arpa_str, (new_arpa_str, duration_str, energy_str) in custom_arpa_subs.items():
            old_arpa = tuple(old_arpa_str.split())
            new_arpa = new_arpa_str.split(' ')
            durations = [int(x[1:]) if x.startswith('=') else float(x) for x in duration_str.split(' ')]
            assert len(new_arpa) == len(durations), f'Mismatched arpa and duration: {new_arpa_str} {duration_str}'
            energies = [int(x[1:]) if x.startswith('=') else float(x) for x in energy_str.split(' ')]
            assert len(new_arpa) == len(energies), f'Mismatched arpa and energy: {new_arpa_str} {energy_str}'
            self.custom_arpa_subs[old_arpa] = (new_arpa, durations, energies)

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
        e_factor = []
        for phone in phones:
            if phone in PUNCS or phone == '|':
                arpas.append(phone)
                d_factor.append(1.0)
                e_factor.append(0.0)
                continue

            long = False
            if phone.endswith(LONG_MARK):
                phone = phone[:-1]
                long = True

            arpa = self.phones_to_arpa[phone]
            arpas.extend(arpa)
            if phone in self.phones_to_duration:
                d_fac = self.phones_to_duration[phone]
            else:
                d_fac = [1.0] * len(arpa)
            if long:
                d_fac = [d * LONG_D_FACTOR for d in d_fac]
            d_factor.extend(d_fac)
            if phone in self.phones_to_energy:
                e_factor.extend(self.phones_to_energy[phone])
            else:
                e_factor.extend([0.0] * len(arpa))  # e_factor is added, not multiplied

        if self.subs_maxlen:  # apply custom_arpa_subs
            for sub_len in range(self.subs_maxlen, self.subs_minlen - 1, -1):
                # print_table(arpas=arpas, d_factor=d_factor, e_factor=e_factor)
                arpas_tup = tuple(arpas)
                i = 0
                while i < len(arpas_tup):
                    sub_arpa = arpas_tup[i:i+sub_len]
                    if sub_arpa in self.custom_arpa_subs:
                        new_arpa, durations, energies = self.custom_arpa_subs[sub_arpa]
                        arpas[i] = new_arpa
                        arpas[i+1:i+sub_len] = [[]] * (sub_len - 1)
                        d_factor[i] = [d_factor[i+d] if isinstance(d, int) else d for d in durations]
                        d_factor[i+1:i+sub_len] = [[]] * (sub_len - 1)
                        e_factor[i] = [e_factor[i+e] if isinstance(e, int) else e for e in energies]
                        e_factor[i+1:i+sub_len] = [[]] * (sub_len - 1)
                        i += sub_len - 1
                    i += 1
                arpas = flatten_list(arpas)
                d_factor = flatten_list(d_factor)
                e_factor = flatten_list(e_factor)

        arpas = np.array(arpas)
        not_wordseps = arpas != '|'
        arpas = arpas[not_wordseps]
        d_factor = np.array(d_factor)[not_wordseps]
        e_factor = np.array(e_factor)[not_wordseps]

        return arpas, d_factor, e_factor

    def gen_inputs(self, german, verbose=False, device=DEVICE):
        if isinstance(german, str):
            ipas = phonemize(german)
        elif isinstance(german, list):
            ipas = german
        else:
            raise NotImplementedError('german must be string of German text or list of regularized German phones')
        arpas, d_factor, e_factor = self.convert_phones_arpa(ipas)
        if verbose:
            print_table(arpas=arpas, d_factor=d_factor, e_factor=e_factor)
        d_factor = torch.tensor(d_factor, device=device).unsqueeze(0)
        e_factor = torch.tensor(e_factor, device=device).unsqueeze(0)
        return arpas, d_factor, e_factor


    def gen_audio(
            self,
            german,
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
            arpas, d_factor, e_factor = self.gen_inputs(german, verbose=verbose, device=device)
        else:
            arpas, d_factor, e_factor = inputs

        infer_kwargs = {
            'd_factor': d_factor,
            'p_factor': None,
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
                filename = ''.join(german) if isinstance(german, list) else german
                if arpa_in_filename:
                    filename = f'{filename}-{"".join(arpas)}'
                filename = f'{filename}.wav'
            save_path = save_dir / filename
            os.makedirs(save_path.parent, exist_ok=True)
            sf.write(save_path, wav_modified.squeeze().cpu().numpy(), 22050, "PCM_16")
        return arpas, d_factor, e_factor

