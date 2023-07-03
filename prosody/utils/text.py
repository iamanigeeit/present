import re
from pathlib import Path
import itertools
import torch
import unicodedata
from builtins import str as unicode
import g2p_en
from g2p_en.expand import normalize_numbers
from nltk.corpus import cmudict
from prosody.aligner import G2PAligner
from prosody.utils.utils import generate_table

ALIGNER = G2PAligner(Path(__file__).resolve().parent / 'g2p_dict.txt')
WORD_CHARS = r"a-z'\-"
PUNCS = '.,?!'
MODIFIER_MARKS = '_^*~'
REMOVE_REGEX = re.compile(rf"[^ A-Z{WORD_CHARS}{PUNCS}{MODIFIER_MARKS}]")
WORD_REGEX = re.compile(rf"[{WORD_CHARS}{MODIFIER_MARKS}]+|[{PUNCS}]")
NONWORD_REGEX = re.compile(rf"[^{WORD_CHARS}]")
MARKS_REGEX = re.compile(rf"[{MODIFIER_MARKS}]")

G2P = g2p_en.G2p()
CMUDICT_WORDS = set(word for word, _ in cmudict.entries())

ARPA_VOWELS = {'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2',
          'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2',
          'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2',
          'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2',
          'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2',
          'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2'}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

class TextEffectProcessor:

    def __init__(self, tokens2ids_fn,
                 aligner=ALIGNER,
                 wordset=CMUDICT_WORDS,
                 g2p=G2P,
                 factor_config=DEFAULT_FACTOR_CONFIG,
                 device=DEVICE
                 ):
        self.aligner = aligner
        self.tokens2ids_fn = tokens2ids_fn
        self.wordset = wordset
        self.G2P = g2p
        self.device = device
        
        # Settings
        self.duration_unit = factor_config['duration_unit']
        self.vowel_pitch_up = factor_config['vowel_pitch_up']
        self.nonvowel_pitch_up = factor_config['nonvowel_pitch_up']
        self.vowel_pitch_down = factor_config['vowel_pitch_down']
        self.nonvowel_pitch_down = factor_config['nonvowel_pitch_down']
        self.vowel_energy_up = factor_config['vowel_energy_up']
        self.nonvowel_energy_up = factor_config['nonvowel_energy_up']
        self.question = factor_config['question']
        self.exclamation = factor_config['exclamation']
        
        # States
        self.alignments = []
        self.mark_positions = {}

    def __str__(self):
        if self.alignments:
            grapheme_list = [a[0] for a in self.alignments] + ['']
            phoneme_list = ['+'.join(a[1]) for a in self.alignments] + ['']
            graph_positions = []
            for i, grapheme in enumerate(grapheme_list):
                graph_positions.extend([i] * len(grapheme))
            low_tones = [''] * len(graph_positions)
            for mark_pos in self.mark_positions['_']:
                low_tones[graph_positions[mark_pos]] = '_'
            high_tones = [''] * len(graph_positions)
            for mark_pos in self.mark_positions['^']:
                high_tones[graph_positions[mark_pos]] = '^'
            emphases = [''] * len(graph_positions)
            emph_pos = self.mark_positions['*']
            for start, stop in zip(emph_pos[0::2], emph_pos[1::2]):
                for i in range(graph_positions[start], graph_positions[stop]):
                    emphases[i] = '*'
            longers = [''] * len(graph_positions)
            for mark_pos in self.mark_positions['~']:
                longers[graph_positions[mark_pos]-1] += '~'
            return generate_table(
                grapheme_list=grapheme_list, phoneme_list=phoneme_list,
                low_tones=low_tones, high_tones=high_tones, emphases=emphases, longers=longers
            )
        else:
            return 'TextEffectProcessor: no alignment on text yet'

    def preprocess(self, text):
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
        text = self.mark_repeats(text)
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
        self.mark_positions = mark_positions
        return words, mark_positions

    def mark_repeats(self, text):
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
                    if NONWORD_REGEX.sub('', candidate_word) in self.wordset:
                        for n, repeat_len, i in zip(perm, repeat_lens, i_to_shorten):
                            tmp_segments[i] = tmp_segments[i] + '~' * (repeat_len - n)
                        replace_word = ''.join(tmp_segments)
                        edits.append((word, replace_word))
                        break
        for word, replace_word in edits:
            text = text.replace(word, replace_word)
        return text


    def phonemize(self, words):
        phonemes = self.G2P(' '.join(words))
        word_phones = []
        start, end = 0, 1
        while end < len(phonemes):
            if phonemes[end] == ' ':
                word_phones.append(phonemes[start:end])
                start = end + 1
            end += 1
        word_phones.append(phonemes[start:end])
        return word_phones


    def get_alignment_data(self, words, word_phones):
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
        self.alignments = alignments
        return alignments, graphemes, phonemes, g2p_pos
    
    def get_scale_factors(self, phonemes, g2p_pos, has_eos_token=False):
        factor_len = len(phonemes) + has_eos_token
        d_factor = [1.0] * factor_len
        p_factor = [0.0] * factor_len
        e_factor = [0.0] * factor_len

        for longer_pos in self.mark_positions['~']:
            phone_start, phone_end = g2p_pos[longer_pos]
            for i in range(phone_start-1, phone_end-1):  # tildes come AFTER the grapheme
                d_factor[i] += self.duration_unit

        for high_p_pos in self.mark_positions['^']:
            phone_pos, _ = g2p_pos[high_p_pos]
            while phone_pos < len(phonemes):
                if phonemes[phone_pos] in ARPA_VOWELS:
                    p_factor[phone_pos] += self.vowel_pitch_up
                    break
                else:
                    p_factor[phone_pos] += self.nonvowel_pitch_up
                    phone_pos += 1

        for low_p_pos in self.mark_positions['_']:
            phone_pos, _ = g2p_pos[low_p_pos]
            while phone_pos < len(phonemes):
                if phonemes[phone_pos] in ARPA_VOWELS:
                    p_factor[phone_pos] += self.vowel_pitch_down
                    break
                else:
                    p_factor[phone_pos] += self.nonvowel_pitch_down
                    phone_pos += 1

        emph_pos = self.mark_positions['*']
        for graph_start, graph_end in zip(emph_pos[0::2], emph_pos[1::2]):
            phone_start = g2p_pos[graph_start][0]
            phone_end = g2p_pos[graph_end][0]
            for phone_pos in range(phone_start, phone_end):
                if phonemes[phone_pos] in ARPA_VOWELS:
                    e_factor[phone_pos] += self.vowel_energy_up
                else:
                    e_factor[phone_pos] += self.nonvowel_energy_up

        for i, phoneme in enumerate(phonemes):
            if phoneme == '?' and self.question:
                # Mark nearest consonants as +1, nearest vowel as +0.5, second nearest vowel -0.5
                phone_pos = i-1
                vowels_found = 0
                while phone_pos >= 0:
                    phoneme = phonemes[phone_pos]
                    if phoneme in PUNCS:
                        break
                    elif phoneme in ARPA_VOWELS:
                        if vowels_found:
                            p_factor[phone_pos] += self.vowel_pitch_down / 2.0
                            break
                        else:
                            vowels_found = 1
                            p_factor[phone_pos] += self.vowel_pitch_up / 2.0
                    else:
                        if not vowels_found:
                            p_factor[phone_pos] += self.vowel_pitch_up
                    phone_pos -= 1

            elif phoneme == '!' and self.exclamation:
                phone_pos = i-1
                vowels_found = 0
                while phone_pos >= 0:
                    phoneme = phonemes[phone_pos]
                    if phoneme in PUNCS:
                        break
                    elif phoneme in ARPA_VOWELS:
                        vowels_found += 1
                        d_factor[phone_pos] += self.duration_unit
                        p_factor[phone_pos] += self.vowel_pitch_up / 2.0 * vowels_found
                        if vowels_found >= 2 and '1' in phoneme:
                            break
                    else:
                        p_factor[phone_pos] += self.nonvowel_pitch_up / 2.0
                    phone_pos -= 1

        d_factor = torch.tensor(d_factor, device=self.device).unsqueeze(0)
        p_factor = torch.tensor(p_factor, device=self.device).unsqueeze(0)
        e_factor = torch.tensor(e_factor, device=self.device).unsqueeze(0)
        d_split_factor = d_factor.int()
        d_factor /= d_split_factor
        return d_factor, p_factor, e_factor, d_split_factor


    def get_inputs(self, text, has_eos_token=False, print_alignment=False):
        words, _ = self.preprocess(text)
        word_phones = self.phonemize(words)
        _, _, phonemes, g2p_pos = self.get_alignment_data(words, word_phones)
        phone_ids = torch.tensor(self.tokens2ids_fn(phonemes), dtype=torch.int32, device=self.device)
        if print_alignment:
            print(self)
        d_factor, p_factor, e_factor, d_split_factor = self.get_scale_factors(phonemes, g2p_pos, has_eos_token)
        return phonemes, phone_ids, d_factor, p_factor, e_factor, d_split_factor


