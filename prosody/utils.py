import math
import re
from pathlib import Path
import itertools
from tabulate import tabulate
import numpy as np
import torch
import unicodedata
from builtins import str as unicode
from g2p_en.expand import normalize_numbers
from nltk.corpus import cmudict
from prosody.aligner import G2PAligner
import librosa
import soundfile as sf
import g2p_en
G2P = g2p_en.G2p()
ALIGNER = G2PAligner(Path(__file__).resolve().parent / 'g2p_dict.txt')
WORD_CHARS = r"a-z'\-"
PUNCS = '.,?!'
MODIFIER_MARKS = '_^*~'
REMOVE_REGEX = re.compile(rf"[^ A-Z{WORD_CHARS}{PUNCS}{MODIFIER_MARKS}]")
WORD_REGEX = re.compile(rf"[{WORD_CHARS}{MODIFIER_MARKS}]+|[{PUNCS}]")
NONWORD_REGEX = re.compile(rf"[^{WORD_CHARS}]")
MARKS_REGEX = re.compile(rf"[{MODIFIER_MARKS}]")

CMUDICT_WORDS = set(word for word, _ in cmudict.entries())

ARPA_VOWELS = {'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
          'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
          'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2',
          'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
          'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2',
          'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2'}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def preprocess(text):
    assert text.count('*') % 2 == 0, 'Closing asterisk missing'
    assert not text.startswith('~'), 'Cannot start with tilde'
    # Copied directly from g2p_en
    text = unicode(text)
    text = normalize_numbers(text)
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents
    text = REMOVE_REGEX.sub('', text)
    text = text.replace("i.e.", "that is")
    text = text.replace("e.g.", "for example")
    # Locate caps, convert to *
    text = re.sub('[A-Z]{2,}', lambda m: f'*{m.group().lower()}*', text)
    text = text.lower()
    # Locate extended letters, convert to ~
    text = mark_repeats(text)
    # Save position of modifier marks
    nospace_text = text.replace(' ', '')
    bare_text = MARKS_REGEX.sub('', nospace_text)
    mark_positions = {mark: [] for mark in MODIFIER_MARKS}
    text_len = len(bare_text)
    nospace_pos = 0
    bare_pos = 0
    while bare_pos < text_len:
        if bare_text[bare_pos] != nospace_text[nospace_pos]:
            mark = nospace_text[nospace_pos]
            mark_positions[mark].append(bare_pos)
        else:
            bare_pos += 1
        nospace_pos += 1
    nomarks_text = MARKS_REGEX.sub('', text)
    words = WORD_REGEX.findall(nomarks_text)
    return words, mark_positions


def mark_repeats(text):
    words = WORD_REGEX.findall(text)
    edits = []
    for word in words:
        repeat_matches = list(re.finditer(r'(.)\1{2,}', word))
        if repeat_matches:
            pos = 0
            end = 0
            segments = []
            i_to_shorten = []
            repeat_lens = []
            for m in repeat_matches:
                start, end = m.regs[0]
                if start > pos:
                    segments.append(word[pos:start])
                segments.append(word[start:end])
                i_to_shorten.append(len(segments) - 1)
                repeat_lens.append(end - start)
                pos = end
            if end < len(word):
                segments.append(word[end:])
            num_i = len(i_to_shorten)
            perms = []
            for j in range(num_i, -1, -1):
                tmp_list = j * [2] + (num_i - j) * [1]
                perms.extend(set(itertools.permutations(tmp_list)))
            for perm in perms:
                tmp_segments = segments.copy()
                for n, i in zip(perm, i_to_shorten):
                    tmp_segments[i] = tmp_segments[i][0] * n
                candidate_word = ''.join(tmp_segments)
                if NONWORD_REGEX.sub('', candidate_word) in CMUDICT_WORDS:
                    for n, repeat_len, i in zip(perm, repeat_lens, i_to_shorten):
                        tmp_segments[i] = tmp_segments[i] + '~' * (repeat_len - n)
                    replace_word = ''.join(tmp_segments)
                    edits.append((word, replace_word))
                    break
    for word, replace_word in edits:
        text = text.replace(word, replace_word)
    return text


def phonemize(words):
    phonemes = G2P(' '.join(words))
    word_phones = []
    start, end = 0, 1
    while end < len(phonemes):
        if phonemes[end] == ' ':
            word_phones.append(phonemes[start:end])
            start = end + 1
        end += 1
    word_phones.append(phonemes[start:end])
    return word_phones


def get_alignment_data(words, word_phones):
    alignments = []
    for word, pron in zip(words, word_phones):
        if NONWORD_REGEX.match(word):
            alignments.extend([(word, tuple(pron))])
        else:
            alignments.extend(ALIGNER(word, pron))
    graphemes = ''.join(words)
    phonemes = [phone for phones in word_phones for phone in phones]
    g2p_pos = []
    graph_pos = 0
    phone_start = 0
    for graph, phone in alignments:
        phone_end = phone_start + len(phone)
        for i in range(len(graph)):
            g2p_pos.append((phone_start, phone_end))
        graph_pos += len(graph)
        phone_start = phone_end
    return alignments, graphemes, phonemes, g2p_pos


DEFAULT_FACTOR_CONFIG = {
    'duration_unit': 0.5,
    'vowel_pitch_up': 1.0,
    'nonvowel_pitch_up': 0.5,
    'vowel_pitch_down': -1.0,
    'nonvowel_pitch_down': -0.5,
    'vowel_energy_up': 2.0,
    'nonvowel_energy_up': 1.0,
    'question': True,
    'exclamation': False,
}

def get_scale_factors(phonemes, g2p_pos, mark_positions,
                      has_eos_token=False, factor_config=DEFAULT_FACTOR_CONFIG, device=DEVICE):
    factor_len = len(phonemes) + has_eos_token
    d_factor = [1.0] * factor_len
    p_factor = [0.0] * factor_len
    e_factor = [0.0] * factor_len

    for longer_pos in mark_positions['~']:
        phone_start, phone_end = g2p_pos[longer_pos]
        for i in range(phone_start-1, phone_end-1):  # tildes come AFTER the grapheme
            d_factor[i] += factor_config['duration_unit']

    for high_p_pos in mark_positions['^']:
        phone_pos, _ = g2p_pos[high_p_pos]
        while phone_pos < len(phonemes):
            if phonemes[phone_pos] in ARPA_VOWELS:
                p_factor[phone_pos] += factor_config['vowel_pitch_up']
                break
            else:
                p_factor[phone_pos] += factor_config['nonvowel_pitch_up']
                phone_pos += 1

    for low_p_pos in mark_positions['_']:
        phone_pos, _ = g2p_pos[low_p_pos]
        while phone_pos < len(phonemes):
            if phonemes[phone_pos] in ARPA_VOWELS:
                p_factor[phone_pos] += factor_config['vowel_pitch_down']
                break
            else:
                p_factor[phone_pos] += factor_config['nonvowel_pitch_down']
                phone_pos += 1

    emph_pos = mark_positions['*']
    if len(emph_pos) % 2 == 1:
        emph_pos.append(len(g2p_pos) - 1)
    for graph_start, graph_end in zip(emph_pos[0::2], emph_pos[1::2]):
        phone_start = g2p_pos[graph_start][0]
        phone_end = g2p_pos[graph_end][1]
        for phone_pos in range(phone_start, phone_end):
            if phonemes[phone_pos] in ARPA_VOWELS:
                e_factor[phone_pos] += factor_config['vowel_energy_up']
            else:
                e_factor[phone_pos] += factor_config['nonvowel_energy_up']

    for i, phoneme in enumerate(phonemes):
        if phoneme == '?' and factor_config['question']:
            # Mark nearest consonants as +1, nearest vowel as +0.5, second nearest vowel -0.5
            phone_pos = i-1
            vowels_found = 0
            while phone_pos >= 0:
                phoneme = phonemes[phone_pos]
                if phoneme in PUNCS:
                    break
                elif phoneme in ARPA_VOWELS:
                    if vowels_found:
                        p_factor[phone_pos] += factor_config['vowel_pitch_down'] / 2.0
                        break
                    else:
                        vowels_found = 1
                        p_factor[phone_pos] += factor_config['vowel_pitch_up'] / 2.0
                else:
                    if not vowels_found:
                        p_factor[phone_pos] += factor_config['vowel_pitch_up']
                phone_pos -= 1

        elif phoneme == '!' and factor_config['exclamation']:
            phone_pos = i-1
            vowels_found = 0
            while phone_pos >= 0:
                phoneme = phonemes[phone_pos]
                if phoneme in PUNCS:
                    break
                elif phoneme in ARPA_VOWELS:
                    vowels_found += 1
                    d_factor[phone_pos] += factor_config['duration_unit']
                    p_factor[phone_pos] += factor_config['vowel_pitch_up'] / 2.0 * vowels_found
                    if vowels_found >= 2 and '1' in phoneme:
                        break
                else:
                    p_factor[phone_pos] += factor_config['nonvowel_pitch_up'] / 2.0
                phone_pos -= 1

    d_factor = torch.tensor(d_factor, device=device)
    p_factor = torch.tensor(p_factor, device=device)
    e_factor = torch.tensor(e_factor, device=device)
    d_split_factor = None
    d_div = d_factor > 2.0
    if d_div.any():
        d_split_factor = d_div + 1.0
    return d_factor, p_factor, e_factor, d_split_factor


def extract_normalized_feats(audio_file, pretrained_tts, silence_db=0, device=DEVICE):
    pretrained_model = pretrained_tts.model
    with open(audio_file, 'rb') as f:
        audio, sample_rate = sf.read(f)
        target_sr = pretrained_model.feats_extract.fs
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
        if silence_db:
            audio, _ = librosa.effects.trim(y=audio, top_db=silence_db)
    speech = torch.tensor(audio, dtype=getattr(torch, pretrained_tts.train_args.train_dtype), device=device).unsqueeze(0)
    speech_lengths = torch.tensor([len(audio)], device=device)
    feats, feats_lengths = pretrained_model.feats_extract(speech, speech_lengths)
    pitch, pitch_lengths = pretrained_model.pitch_extract(speech, speech_lengths, feats_lengths=feats_lengths)
    energy, energy_lengths = pretrained_model.energy_extract(speech, speech_lengths, feats_lengths=feats_lengths)
    pitch = pretrained_model.pitch_normalize(pitch.squeeze(-1), pitch_lengths)[0].unsqueeze(-1)
    energy = pretrained_model.energy_normalize(energy.squeeze(-1), energy_lengths)[0].unsqueeze(-1)
    return speech, speech_lengths, feats, feats_lengths, pitch, pitch_lengths, energy, energy_lengths

def get_inputs(text, tokens2ids_fn, has_eos_token=False, device=DEVICE):
    words, mark_positions = preprocess(text)
    word_phones = phonemize(words)
    alignments, graphemes, phonemes, g2p_pos = get_alignment_data(words, word_phones)
    phone_ids = torch.tensor(tokens2ids_fn(phonemes), dtype=torch.int32, device=device)
    d_factor, p_factor, e_factor, d_split_factor = get_scale_factors(phonemes, g2p_pos, mark_positions, has_eos_token)
    return phonemes, phone_ids, d_factor, p_factor, e_factor, d_split_factor

def phone_time_stretch(wav, phone_durations, d_factor, frame_rate=256):
    segments = []
    start_pos = 0
    end_pos = 0
    for d, phone_duration in zip(d_factor, phone_durations):
        end_pos += phone_duration * frame_rate
        segment = wav[start_pos:end_pos]
        if d != 1.0:
            segment = librosa.effects.time_stretch(segment, rate=1.0/d, n_fft=frame_rate)
        segments.append(segment)
    return np.concatenate(segments)


def phone_pitch_shift(wav, phone_durations, p_factor, sampling_rate=22050, frame_rate=256):
    segments = []
    start_pos = 0
    end_pos = 0
    for p, phone_duration in zip(p_factor, phone_durations):
        end_pos += phone_duration * frame_rate
        segment = wav[start_pos:end_pos]
        if p:
            segment = librosa.effects.pitch_shift(segment, sr=sampling_rate, n_steps=p*2.0)
        segments.append(segment)
    return np.concatenate(segments)

def phone_change_energy(wav, phone_durations, e_factor, frame_rate=256):
    start_pos = 0
    end_pos = 0
    segments = []
    for e, phone_duration in zip(e_factor, phone_durations):
        end_pos += e * frame_rate
        segment = wav[start_pos:end_pos]
        amplitude_factor = math.sqrt(2 / (1 + math.exp(-e)))
        segments.append(segment * amplitude_factor)
        start_pos = end_pos
    return np.concatenate(segments)


def mod_fn_generator(reference_points, combine_fn=None):
    def mod_fn(pred_val, split):
        refs = len(reference_points)
        count = split.item()
        if refs == 0:
            return pred_val
        elif refs == 1 or refs == count:
            mod_arr = reference_points
        else:
            if count == 1:
                mod_arr = [sum(reference_points) / refs]
            else:
                mod_arr = np.interp(np.arange(count) / (count-1) * (refs-1), list(range(refs)), reference_points)
        mod_tensor = torch.tensor(mod_arr, device=pred_val.device)
        if combine_fn is None:
            return mod_tensor
        else:
            return combine_fn(mod_tensor, pred_val)
    return mod_fn


def duration_even_split(min_duration=0.0, max_duration=100.0):
    def mod_fn(pred_val, split):
        duration = pred_val.item()
        if duration > 0.0:
            if duration < min_duration:
                duration = min_duration
            elif duration > max_duration:
                duration = max_duration
        spl = split.item()
        first = duration // spl
        if spl == 2:
            last = duration - first
            mod_arr = [first, last]
        else:
            last = first
            middle = duration - first - last
            mod_arr = [first] + [middle / (spl - 2)] * (spl - 2) + [last]
        return torch.tensor(mod_arr, device=pred_val.device)
    return mod_fn


def get_d_mod_fns(d_split_factor, duration_split_fn, min_duration=0, max_duration=100):
    d_mod_fns = {}
    for batch_i, d_split in enumerate(d_split_factor):
        phone_i = 0
        while phone_i < len(d_split):
            split = d_split[phone_i]
            add_i = []
            while split != 1:
                add_i.append(phone_i)
                phone_i += 1
                try:
                    split = d_split[phone_i]
                except IndexError:
                    break
            if add_i:
                for i in add_i:
                    d_mod_fns[(batch_i, i)] = duration_split_fn(min_duration / len(add_i), max_duration / len(add_i))
            else:
                phone_i += 1
    return d_mod_fns


def round_floats(float_list):
    for x in float_list:
        n = format(x, '.2f').rstrip('.0')
        yield n if n else '0'

def round_and_join(float_list):
    return '|'.join(round_floats(float_list))

def print_numseq(**kwargs):
    rows = {}
    for name, row in kwargs.items():
        if isinstance(row, torch.Tensor):
            r = torch.flatten(row).cpu().tolist()
            if torch.is_floating_point(row):
                rows[name] = list(round_floats(r))
            else:
                rows[name] = r
        else:
            rows[name] = row
    if 'd_split_factor' in rows:
        d_split_factor = rows['d_split_factor']
        if 'duration_new' in rows:
            duration_new = rows['duration_new']
            duration = []
            start = 0
            for fac in d_split_factor:
                end = start + fac
                duration.append(round_and_join(duration_new[start:end]))
                start = end
            rows['duration_new'] = duration
        if 'pitch_new' in rows:
            pitch_new = rows['pitch_new']
            pitch = []
            start = 0
            for fac in d_split_factor:
                end = start + fac
                pitch.append(round_and_join(pitch_new[start:end]))
                start = end
            rows['pitch_new'] = pitch
        if 'pitch_values' in rows:
            flattened = []
            pitch_values = rows['pitch_values']
            for i, pitch_value in enumerate(pitch_values):
                if pitch_value is None:
                    flattened.append('')
                else:
                    flattened.append(round_and_join(pitch_value))
            rows['pitch_values'] = flattened
        # if 'p_mod_fns' in rows:
        #     p_mod_fns = rows['p_mod_fns']
        #     p_mods = []
        #     for i, fac in enumerate(d_split_factor):
        #         if (0, i) in p_mod_fns:
        #             p_mod = p_mod_fns[(0, i)](torch.Tensor([0.0]), torch.Tensor([fac])).cpu().tolist()
        #             p_mod = round_and_join(p_mod)
        #         else:
        #             p_mod = None
        #         p_mods.append(p_mod)
        #     rows['p_mod_fns'] = p_mods
    table = []
    for name, row in rows.items():
        table.append([name] + row)
    print(tabulate(table, headers=list(range(-1, len(row)+1)), tablefmt='plain', stralign='center'))

def find_dataset_ranges(wav_paths, pretrained_tts, device=DEVICE):
    # glob('/home/perry/PycharmProjects/LJSpeech-1.1/wavs/*.wav')
    max_pitch, min_pitch = 0, 0
    max_energy, min_energy = 0, 0
    for w in wav_paths:
        _, _, _, _, pitch, _, energy, _ = extract_normalized_feats(w, pretrained_tts, device=device)
        new_max_pitch = pitch.max().item()
        if new_max_pitch > max_pitch:
            max_pitch = new_max_pitch
        new_min_pitch = pitch.min().item()
        if new_min_pitch < min_pitch:
            min_pitch = new_min_pitch
        new_max_energy = energy.max().item()
        if new_max_energy > max_energy:
            max_energy = new_max_energy
        new_min_energy = energy.min().item()
        if new_min_energy < min_energy:
            min_energy = new_min_energy
    return (min_pitch, max_pitch), (min_energy, max_energy)

