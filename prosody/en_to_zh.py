import itertools
from pathlib import Path
import regex as re
import numpy as np
import torch
import soundfile as sf
from pywordseg import Wordseg
from prosody.utils.text import ARPA_VOWELS, PUNCS
from prosody.utils.utils import get_p_mod_fns, get_d_mod_fns, duration_even_split, print_table
from prosody.pinyin import (hans_to_pinyins, regularize_pinyin, NULL_INITIAL, VALID_PINYIN_REGEX,
                            INITIALS as INITIALS_LIST, RIMES as RIMES_LIST)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


INITIALS_TO_ARPA_DURATIONS = {
    'ʔ': (',',
          '0.2'),

    'b': ('B P',
          '1 0'),
    'p': ('P HH',
          '1 0.5'),
    'm': 'M',
    'f': 'F',

    'd': ('D T',
          '1 0'),
    't': ('T HH',
          '1 0.5'),
    'n': 'N',
    'l': (', L',
          '0 1'),

    'g': ('G K',
          '1 0'),
    'k': ('K HH',
          '1 0.5'),
    'h': 'HH',

    'j': ('T S SH',
          '0.5 0.5 0'),
    'q': ('T CH HH',
          '0 1 0.5'),
    'x': ('SH S',
          '1 0'),

    'zh': ('T SH',
           '1 0'),
    'ch': ('CH HH',
           '1 0.5'),
    'sh': 'SH',
    'r': 'R',

    'z': ('T S',
          '0.5 0.5'),
    'c': ('T S HH',
          '0.3 0.3 0.7'),
    's': 'S',

    'y': 'Y',
    'w': 'W',
}

RIMES_TO_ARPA_DURATIONS = {
    'ɨ': ('Z UH1',
          '0.5 2'),
    'a': ('AH1 AA1 ,',
          '1.5  0 0'),
    'ai': ('AY1', '2'),
    'an': ('AH1 N',
           '1.5 0.5'),
    'ang': ('AH1 NG',
            '1.5 0.5'),
    'ao': ('AA1 AW1',
           '0.5 1.5'),

    'e': ('UH1 AH1 ,',
          '0 1.5 0'),
    'ei': ('EY1', '1.5'),
    'en': ('UH1 N',
           '1.5 0.5'),
    'eng': ('UH1 NG',
            '1.5 0.5'),

    'er': ('ER1',
           '1.5'),

    'i': ('IY1 ,',
          '1.5 0'),
    'ia': ('Y AH1',
           '0.5 1.5'),
    'iang': ('Y AH1 NG',
             '0.5 1 0.5'),
    'iao': ('Y AW1',
            '0.5 1'),
    'ie': ('Y EH1',
           '0.5 1'),
    'ien': ('Y EH1 N',
            '0.5 1 0.5'),
    'in': ('IH1 IY1 N',
           '1 0 0.5'),
    'ing': ('IH1 IY1 NG',
            '1 0 1'),
    'iong': ('Y OW1 NG',
             '0.5 1 1'),
    'iu': ('Y OW1',
           '0.5 1'),

    'o': ('AO1 W',
          ' 1  0'),
    'ou': ('OW1', '1.5'),
    'ong': ('OW1 NG',
            '1.5 0.5'),

    'u': ('W UW1 ,',
          '0  1.5  0'),
    'ua': ('W AA1',
           '0.5 1'),
    'uai': ('W AA1 Y',
            '0.5 1 0.5'),
    'uan': ('W AH1 N',
            '0.5 1 0.5'),
    'uang': ('W AH1 NG',
             '1 1 1'),
    'ui': ('W EY1',
           '0.5 1'),
    'un': ('W UH1 N',
           '0.5 1 0.5'),
    'uo': ('UH1 AO1',
           '0.8 0.8'),

    'v': ('UH1 Y',
          '0.5 1.5'),
    've': ('Y UW1 EH1 ,',
           '0 0.5 1 0'),
    'vn': ('UH1 Y N',
           '1 0.5 0.5'),
    'ven': ('W EH1 N',
            '0.5 1 0.5'),
}

CUSTOM_ARPA_DURATIONS = {
    'ʔa': (', AH1 AA1 ,',
           '0.2 1.5 1 0'),
    'ʔe': (', UH1 AH1 ,',
           '0.2 0.5 1 0'),

    'cɨ': ('T    S  HH UH1 R',
           '0.5 0.5 0.5 0  1'),  # vowel should be Z UH but HH + Z fails
    'chɨ': ('CH HH R R',
            '1 0.5 1 1'),

    # 'da': ('T D AA1 ,',
    #        '1 0 1 0'),
    # 'di': ('T T D IY1 ,',
    #        '1 0 0 1.5 0'),
    # 'die': ('T T D Y EH1',
    #         '1 0 0 1  1'),
    # 'din': ('T T D IH1 IY1 N',
    #         '1 0 0  1   0  1'),
    # 'ding': ('T T D IH1 IY1 NG',
    #          '1 0 0  1   0  1'),
    # 'diou': ('T T D Y OW1',
    #          '1 0 0 1 1'),
    # 'du': ('T D UW1 W',
    #        '1 0 1 0'),
    # 'dun': ('T T D W UH1 N',
    #         '0 1 0 1 1 1'),
    #
    # 'gu': ('K G W UW1',
    #        '1 0 0 1'),

    'hong': ('HH OW1 UW1 NG',
             '1 0.5 0.5 1'),
    'hu': ('HH HH UW1',
           '1 0 1.5'),

    'lve': ('L UH1 Y EH1',
            '1 0.5 0.5 0.5'),

    'neng': ('N AH1 UH1 NG',
             '1 0 1 1'),
    'nong': ('N UH1 OW1 NG',
             '1 0 1 1'),
    'nve': ('N UH1 Y EH1',
            '1 0.5 0.5 0.5'),
    # 'qv': ('T SH HH W IY1 ,',
    #        '0.5 0.5 0.5 0 1 0'),
    # 'qvn': ('T   SH HH UW1 IY1 N ,',
    #         '0.5 0.5 0  0   1  1 0'),
    'rɨ': ('R R',
            '1 1'),
    'ren': ('R AH0 N',
            '1 1.5 1'),

    'shɨ': 'SH R R',

    'tuo': ('TH T HH UH1 AO1',
            '0 1 0.5 0.5 0.5'),

    'wa': 'W AA1',
    'wo': ('W UH1 AO1',
           '0.8 0.8 0.8'),
    'wu': ('W UW1 W',
           '1 1 0'),
    'xve': ('SH W EH1 ,',
            '1 0.5 1 0'),
    'xven': ('SH W EH1 N',
             '1 0.5 1 1'),
    'ye': 'Y EH1',
    'yen': 'Y EH1 N',
    'yve': ('Y UH1 EH1',
            '1 0.5 0.5'),
    'yven': ('Y UH1 EH1 N',
             '1 0.5 0.5 1'),
    # 'yvn': ('Y UH1 Y N',
    #         '1 1 0 1'),
    'zheng': ('T SH AH1 UH1 NG',
              '1 0   0   1  1'),

    'zhɨ': ('CH R UH1 R',
            '1 0.5 1 0'),
}

PINYIN_TO_ARPA_DURATIONS = INITIALS_TO_ARPA_DURATIONS | \
                           RIMES_TO_ARPA_DURATIONS | \
                           CUSTOM_ARPA_DURATIONS

TONE_DURATION_SPLIT = {
    1: 2,
    2: 3,
    3: 3,
    4: 3,
    5: 1
}

TONE_CONTOURS = {
    1: [2.0],
    2: [-1.0, 1.0],
    3: [-1.0, -2.0, -1.0],
    4: [2.0, -1.0],
    5: [0.0],
}

TONE_COMBINE_FNS = {
    1: None,
    2: None,
    3: None,
    4: None,
    5: None,
}

TONE5_D_FACTOR = 0.7
WORD_PAUSE = 0.2
MAX_PITCH_CHANGE = 2.5

# These are the phonemes to split duration on
SPLIT_VOWELS = {'R'} | ARPA_VOWELS
# These are the phonemes to apply tone contour on
NUCLEUS_VOWELS = {'W', 'Y', 'Z'} | SPLIT_VOWELS
# These are BOTH initials and vowels. Skip contour if used as initial
INITIAL_VOWELS = {'R', 'W', 'Y'}

PUNC_CONVERSION = {
    '，': ',',
    '。': '.',
    '？': '?',
    '！': '!'
}


INITIALS = set(INITIALS_TO_ARPA_DURATIONS)
RIMES = set(RIMES_TO_ARPA_DURATIONS)
assert INITIALS == set(INITIALS_LIST)
assert RIMES == set(RIMES_LIST)
PINYIN_WITH_TONE_REGEX = re.compile(
    rf"({'|'.join(INITIALS)})({'|'.join(RIMES)})([1-5])"
)


class PinyinArpaSpeech:

    def __init__(
            self,
            tts_inference_fn,
            token_id_converter,
            pinyin_to_arpa_durations=PINYIN_TO_ARPA_DURATIONS,
            tone_duration_split=TONE_DURATION_SPLIT,
            tone_contours=TONE_CONTOURS,
            tone_combine_fns=TONE_COMBINE_FNS,
            tone5_d_factor=TONE5_D_FACTOR,
            max_pitch_change=MAX_PITCH_CHANGE,
            segment_chinese=True,
            nucleus_tone_only=True,
            tone_interpolation=False,
    ):
        self.tts_inference_fn = tts_inference_fn
        self.token_id_converter = token_id_converter
        self.pinyin_to_arpa, self.pinyin_to_duration = {}, {}
        self.update_arpa_durations(pinyin_to_arpa_durations)
        self.tone_duration_split = tone_duration_split
        self.tone_contours = tone_contours
        self.tone_combine_fns = tone_combine_fns
        self.tone5_d_factor = tone5_d_factor
        self.max_pitch_change = max_pitch_change
        self.nucleus_tone_only = nucleus_tone_only
        self.tone_interpolation = tone_interpolation
        self.check_consistency()
        self.segment_chinese = segment_chinese
        if segment_chinese:
            self.wordseg = Wordseg(batch_size=1, embedding='elmo', device=DEVICE)

    def check_consistency(self):
        pinyin_set = INITIALS | RIMES
        arpa_duration_set = set(self.pinyin_to_arpa)
        assert pinyin_set.issubset(arpa_duration_set), f'Missing pinyin: {pinyin_set - arpa_duration_set}'
        for pinyin, arpa in self.pinyin_to_arpa.items():
            if pinyin in self.pinyin_to_duration:
                duration = self.pinyin_to_duration[pinyin]
                assert len(arpa) == len(duration), pinyin

    def update_arpa_durations(self, pinyin_to_arpa_durations, verbose=False):
        for pinyin, arpa_duration in pinyin_to_arpa_durations.items():
            if arpa_duration is None:  # reset to default
                self.update_arpa_durations({pinyin: PINYIN_TO_ARPA_DURATIONS[pinyin]})
            elif isinstance(arpa_duration, tuple):
                arpa_str, duration_str = arpa_duration
                arpa = arpa_str.strip().split()
                duration = [float(d) for d in duration_str.strip().split()]
                assert len(arpa) == len(duration), f'Mismatched arpa and duration: {arpa_str} {duration_str}'
                self.pinyin_to_arpa[pinyin] = arpa
                self.pinyin_to_duration[pinyin] = duration
            elif isinstance(arpa_duration, str):
                arpa = arpa_duration.split()
                self.pinyin_to_arpa[pinyin] = arpa
                if pinyin in self.pinyin_to_duration:
                    del self.pinyin_to_duration[pinyin]
                # no duration, defaults to 1x
            else:
                raise ValueError('pinyin_to_arpa_durations must have str, tuple or None as values')
        if verbose:
            print(f'Updated {", ".join(pinyin_to_arpa_durations.keys())}')

    def update(
            self,
            pinyin_to_arpa_durations=None,
            tone_duration_split=None,
            tone_contours=None,
            tone5_d_factor=None,
            max_pitch_change=None,
            nucleus_tone_only=None,
            tone_interpolation=None,
    ):
        if pinyin_to_arpa_durations:
            self.update_arpa_durations(pinyin_to_arpa_durations, verbose=True)
        if tone_duration_split:
            self.tone_duration_split.update(tone_duration_split)
        if tone_contours:
            self.tone_contours.update(tone_contours)
        if tone5_d_factor is not None:
            self.tone5_d_factor = tone5_d_factor
        if max_pitch_change is not None:
            self.max_pitch_change = max_pitch_change
        if nucleus_tone_only is not None:
            self.nucleus_tone_only = nucleus_tone_only
        if tone_interpolation is not None:
            self.tone_interpolation = tone_interpolation

        self.check_consistency()

    def convert_hanzi_with_pinyin(self, chinese):
        orig_pinyins = re.sub(r'[^ a-zü0-9]', '', chinese).strip().split()
        reg_pinyins = regularize_pinyin(orig_pinyins)
        if self.segment_chinese:
            hans = re.sub(r'[ a-zü0-9]', '', chinese)
            pinyins = self.align_segments(hans, reg_pinyins)
        else:
            pinyins = reg_pinyins
        return pinyins

    def align_segments(self, hans, reg_pinyins):
        assert len(re.sub(r'\P{Han}', '', hans)) == len(reg_pinyins), \
            f'Mismatched hans and pinyins length: {hans} {reg_pinyins}'
        # hans can have punctuation, reg_pinyins cannot
        start = 0
        end = 0
        pinyins = []
        words = self.wordseg.cut([hans])[0]
        if not words:  # weird error where sometimes wordseg returns nothing
            self.wordseg = Wordseg(batch_size=1, embedding='elmo', device=DEVICE)
            words = self.wordseg.cut([hans])[0]
        for word in words:
            if re.search(r'\p{Han}', word) is None:  # punctuation
                pinyins.append(word)
            else:
                end += len(word)
                pinyins.extend(reg_pinyins[start:end])
                pinyins.append(' ')
                start = end
        return pinyins

    def convert_hanzi_to_pinyin(self, hans):
        with_sandhi = hans_to_pinyins(hans)
        reg_pinyins = regularize_pinyin(with_sandhi)
        if self.segment_chinese:
            reg_pinyins = [p for p in reg_pinyins if VALID_PINYIN_REGEX.fullmatch(p) is not None]
            pinyins = self.align_segments(hans, reg_pinyins)
        else:
            pinyins = reg_pinyins
        return pinyins

    def find_py_units(self, pinyin_unit):
        if pinyin_unit in self.pinyin_to_arpa:
            unit_arpas = self.pinyin_to_arpa[pinyin_unit]
            if pinyin_unit in self.pinyin_to_duration:
                unit_d_factor = self.pinyin_to_duration[pinyin_unit]
            else:
                unit_d_factor = [1.0] * len(unit_arpas)
            return unit_arpas, unit_d_factor
        else:
            return [], []

    def build_arpa_duration(self, initial, rime):
        # return word_arpas, word_d_factor
        unit_arpas, unit_d_factor = self.find_py_units(initial + rime)
        if unit_arpas:
            return unit_arpas, unit_d_factor
        else:
            init_arpas, init_d_factor = self.find_py_units(initial)
            rime_arpas, rime_d_factor = self.find_py_units(rime)
            return init_arpas + rime_arpas, init_d_factor + rime_d_factor

    def convert_pinyins_to_arpa(self, pinyin_list, batch_i=0):
        tones = []
        arpas = []
        arpa_lens = []
        d_factor = []
        for pinyin in pinyin_list:
            pinyin = pinyin.strip()
            if pinyin == '':
                arpas.append(',')
                tones.append(0)
                arpa_lens.append(1)
                d_factor.append(WORD_PAUSE)
            elif pinyin in PUNC_CONVERSION:
                arpas.append(PUNC_CONVERSION[pinyin])
                tones.append(0)
                arpa_lens.append(1)
                d_factor.append(1.0)
            elif pinyin in PUNCS:
                arpas.append(pinyin)
                tones.append(0)
                arpa_lens.append(1)
                d_factor.append(1.0)
            else:
                m = PINYIN_WITH_TONE_REGEX.fullmatch(pinyin)
                if m is None:
                    raise ValueError(f'Unrecognized character or pinyin: {pinyin}')
                else:
                    initial, rime, tone = m.groups()
                    tones.append(int(tone))
                    word_arpas, word_d_factor = self.build_arpa_duration(initial, rime)
                    arpas.append(word_arpas)
                    arpa_lens.append(len(word_arpas))
                    d_factor.extend(word_d_factor)

        # arpas is now a list of word_arpas and punctuation e.g. [[T S AW1], [AH1 N], '.']
        joined_arpas = [arpa for word_arpas in arpas for arpa in word_arpas]
        all_tones = [0] * len(joined_arpas)
        arpa_offsets = list(itertools.accumulate(arpa_lens, initial=0))
        d_split_factor = [1] * len(d_factor)

        # Set d_split_factor
        for word_arpas, tone, offset, pinyin in zip(arpas, tones, arpa_offsets[:-1], pinyin_list):
            if isinstance(word_arpas, str):  # this is punctuation
                continue

            start_i = int(word_arpas[0] in INITIAL_VOWELS)  # do not split on initial R, W, Y
            for i in range(start_i, len(word_arpas)):
                arpa = word_arpas[i]
                if arpa in SPLIT_VOWELS and d_factor[offset + i]:
                    d_split_factor[offset + i] = self.tone_duration_split[tone]

        # Set pitch_values and p_mod_fns
        pitch_values = [[]] * len(d_factor)
        combine_fns = [None] * len(d_factor)
        last_pitch = 0.0
        for word_i, (word_arpas, tone, word_start, word_end) in enumerate(
                zip(arpas, tones, arpa_offsets[:-1], arpa_offsets[1:])):

            if isinstance(word_arpas, str):  # punctuation should take last pitch
                pitch_values[word_start] = [last_pitch]
                continue

            nucleus_start = 0
            while word_arpas[nucleus_start] not in NUCLEUS_VOWELS:
                nucleus_start += 1
            nucleus_end = nucleus_start + 1
            while nucleus_end < len(word_arpas) and word_arpas[nucleus_end] in NUCLEUS_VOWELS:
                nucleus_end += 1
            nucleus_start += word_start
            nucleus_end += word_start

            if self.nucleus_tone_only:
                tone_start = nucleus_start
                tone_end = nucleus_end
            else:
                tone_start = word_start
                tone_end = word_end

            for i in range(tone_start, tone_end):
                all_tones[i] = tone

            combine_fn = self.tone_combine_fns[tone]
            if tone == 5:
                for i in range(tone_start, tone_end):
                    d_factor[i] = self.tone5_d_factor * d_factor[i]
                    pitch_values[i] = self.tone_contours[5]
                    combine_fns[i] = combine_fn
            else:
                contour = self.tone_contours[tone]
                refs = len(contour)
                if refs == 1:
                    for i in range(tone_start, tone_end):
                        pitch_values[i] = contour[:]  # slicing or modifying pitch_values[i] will affect contour!!
                        combine_fns[i] = combine_fn
                else:
                    has_initial = int(tone_start < nucleus_start)  # for self.nucleus_tone_only = False
                    has_final = int(tone_end > nucleus_end)
                    split_offsets = list(
                        itertools.accumulate(d_split_factor[nucleus_start:nucleus_end], initial=has_initial)
                    )
                    splits = split_offsets[-1]
                    splits += has_final
                    assert splits > 1, (
                        f'Total d_split_factor across phonemes having tones must be >1 to interpolate pitch\n'
                        f'Pinyins: {" ".join(pinyin_list)}\n'
                        f'Word: {" ".join(word_arpas)}'
                    )
                    word_pitches = np.interp(
                        np.arange(splits) / (splits - 1) * (refs - 1), list(range(refs)), contour
                    ).tolist()
                    # print(has_initial, has_final, split_offsets, splits, word_arpas, pitch_values)
                    for i in range(tone_start, nucleus_start):
                        pitch_values[i] = word_pitches[:1]
                        combine_fns[i] = combine_fn
                    for i, split_start, split_end in zip(
                            range(nucleus_start, nucleus_end), split_offsets[:-1], split_offsets[1:]):
                        pitch_values[i] = word_pitches[split_start:split_end]
                        combine_fns[i] = combine_fn
                    for i in range(nucleus_end, tone_end):
                        pitch_values[i] = word_pitches[-1:]
                        combine_fns[i] = combine_fn

            last_pitch = pitch_values[tone_end-1][-1]

            if self.nucleus_tone_only and not self.tone_interpolation:
                # Set initials / finals tone according to start/end of nucleus tone
                start_pitch = pitch_values[nucleus_start][0]
                end_pitch = pitch_values[nucleus_end - 1][-1]
                for i in range(word_start, nucleus_start):
                    pitch_values[i] = [start_pitch]
                    combine_fns[i] = combine_fn
                for i in range(nucleus_end, word_end):
                    pitch_values[i] = [end_pitch]
                    combine_fns[i] = combine_fn

        # Fill in Nones by interpolating if initials / finals have no tone
        if self.nucleus_tone_only and self.tone_interpolation:
            last_pitch = None
            i = 0
            pitch_value = pitch_values[0]
            while i < len(pitch_values):
                i_to_fill = []
                while not pitch_value:
                    i_to_fill.append(i)
                    i += 1
                    try:
                        pitch_value = pitch_values[i]
                    except IndexError:
                        break
                if i_to_fill:
                    if pitch_value:
                        next_pitch = pitch_value[0]
                    else:
                        next_pitch = last_pitch
                    if last_pitch is None:
                        if next_pitch is None:
                            next_pitch = 0.0
                        last_pitch = next_pitch
                    pitch_increment = (next_pitch - last_pitch) / (len(i_to_fill) + 1)
                    for step, fill_i in enumerate(i_to_fill, 1):
                        pitch_values[fill_i] = [step * pitch_increment + last_pitch]
                else:
                    last_pitch = pitch_value[-1]
                    i += 1
                    pitch_value = pitch_values[i]
        else:
            try:
                for i, pitch_value in enumerate(pitch_values):
                    assert pitch_value
            except AssertionError:
                arpa_list = [a for arpa in arpas for a in arpa]
                print(f'Pitch missing at position {i}\n'
                      f'Pinyin: {" ".join(pinyin_list)}\n')
                print_table(arpas=arpa_list, pitch=pitch_values)
                raise

        # Smooth abrupt changes
        if self.max_pitch_change:
            for word_start, word_end in zip(arpa_offsets[1:-1], arpa_offsets[2:]):
                last_pitch = pitch_values[word_start - 1][-1]
                curr_pitch = pitch_values[word_start][0]
                pitch_change = abs(curr_pitch - last_pitch)
                if pitch_change > self.max_pitch_change:
                    adjustment = (self.max_pitch_change - pitch_change) * int(curr_pitch > last_pitch)
                    pitch_len = sum(len(pitch_value) for pitch_value in pitch_values) - 1
                    count = pitch_len
                    for pitch_value in pitch_values[word_start:word_end]:
                        for i, pitch in enumerate(pitch_value):
                            pitch_value[i] = pitch + adjustment * count / pitch_len
                            count -= 1

        p_mod_fns = get_p_mod_fns(pitch_values, combine_fns, batch_i)
        return joined_arpas, all_tones, d_factor, d_split_factor, pitch_values, p_mod_fns

    def convert_hanzi_to_arpa(self, hans, batch_i=0):
        pinyins = self.convert_hanzi_to_pinyin(hans)
        return self.convert_pinyins_to_arpa(pinyins, batch_i)

    def gen_inputs(self, chinese, verbose=False, add_full_stop=False, device=DEVICE):
        if isinstance(chinese, str):
            if re.search('[a-z][0-5]', chinese) is None:
                if add_full_stop:
                    chinese.append('.')
                arpas, tones, d_factor, d_split_factor, pitch_values, p_mod_fns = self.convert_hanzi_to_arpa(chinese)
            else:  # includes pinyin
                pinyins = self.convert_hanzi_with_pinyin(chinese)
                if add_full_stop:
                    pinyins.append('.')
                arpas, tones, d_factor, d_split_factor, pitch_values, p_mod_fns = self.convert_pinyins_to_arpa(pinyins)
        elif isinstance(chinese, list):
            if add_full_stop:
                chinese.append('.')
            arpas, tones, d_factor, d_split_factor, pitch_values, p_mod_fns = self.convert_pinyins_to_arpa(chinese)
        else:
            raise NotImplementedError('chinese must be string of characters or list of regularized pinyin')
        if verbose:
            print_table(arpas=arpas, d_factor=d_factor, d_split_factor=d_split_factor, tones=tones,
                        pitch_values=pitch_values)
        d_factor = torch.tensor(d_factor, device=device).unsqueeze(0)
        d_split_factor = torch.tensor(d_split_factor, dtype=torch.int32, device=device).unsqueeze(0)
        return arpas, tones, d_factor, d_split_factor, pitch_values, p_mod_fns

    def gen_audio(
            self,
            chinese,
            save_dir,
            inputs=None,
            pac_update_dict=None,
            infer_overrides=None,
            overall_d_factor=1.0,
            fix_durations=True,
            vowel_duration=(9.0, 15.0),
            arpa_in_filename=True,
            custom_filename='',
            disable_tones=False,
            disable_durations=False,
            add_full_stop=False,
            verbose=False,
            device=DEVICE,
    ):
        if pac_update_dict is None:
            pac_update_dict = {}
        if pac_update_dict:
            self.update(**pac_update_dict)
        if inputs is None:
            arpas, tones, d_factor, d_split_factor, pitch_values, p_mod_fns = self.gen_inputs(
                chinese, add_full_stop=add_full_stop, verbose=verbose
            )
        else:
            arpas, tones, d_factor, d_split_factor, pitch_values, p_mod_fns = inputs

        if disable_durations:
            d_override = None
            d_factor = None
            d_split_factor = None
            d_mod_fns = None
        else:
            if fix_durations:
                d_override = overall_d_factor * d_factor
                if d_split_factor is not None:
                    d_override /= d_split_factor
                d_factor = None
                d_mod_fns = None
            else:
                d_factor = d_factor * overall_d_factor
                d_override = None
                min_duration = vowel_duration[0] * overall_d_factor
                max_duration = vowel_duration[1] * overall_d_factor
                if d_split_factor is None:
                    d_mod_fns = get_d_mod_fns(
                        d_split_factor,
                        duration_split_fn=duration_even_split,
                        min_duration=min_duration,
                        max_duration=max_duration
                    )
                else:
                    d_mod_fns = None

        if disable_tones:
            p_mod_fns = None

        infer_kwargs = {
            'd_factor': d_factor,
            'd_override': d_override,
            'p_factor': None,
            'e_factor': None,
            'd_split_factor': d_split_factor,
            'd_mod_fns': d_mod_fns,
            'p_mod_fns': p_mod_fns,
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
                filename = ' '.join(chinese) if isinstance(chinese, list) else chinese
                if arpa_in_filename:
                    filename = f'{filename}-{"".join(arpas)}'
                filename = f'{filename}.wav'
            sf.write(Path(save_dir) / filename, wav_modified.squeeze().cpu().numpy(), 22050, "PCM_16")
        return arpas, tones, d_factor, d_split_factor, pitch_values, p_mod_fns
