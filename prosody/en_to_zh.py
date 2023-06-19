import itertools
import re
import numpy as np
import torch
import soundfile as sf
from pypinyin import lazy_pinyin, Style, load_phrases_dict
from prosody.utils import mod_fn_generator, ARPA_VOWELS, PUNCS, get_d_mod_fns, duration_even_split, print_numseq
from prosody.pinyin import regularize_pinyin, NULL_INITIAL

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ADD_PINYIN_PHRASES = {
    '很长': [['hěn'], ['cháng']],
    '多长': [['duō'], ['cháng']],
    '真长': [['zhēn'], ['cháng']],
    '越长': [['yuè'], ['cháng']],
    '这个': [['zhè'], ['ge']],
    '那个': [['nà'], ['ge']],
    '哪个': [['něi'], ['ge']],
}
load_phrases_dict(ADD_PINYIN_PHRASES)


INITIALS_TO_ARPA_DURATIONS = {
    'ʔ': (',',
          '0.2'),

    'b': ('P B',
          '1 0'),
    'p': ('P HH',
          '1.5 0.5'),
    'm': 'M',
    'f': 'F',

    'd': ('T D ',
          '1 0'),
    't': ('T HH',
          '1.5 0.5'),
    'n': 'N',
    'l': (', L',
          '0 0.5'),

    'g': ('K G',
          '1 0'),
    'k': ('K HH',
          '1.5 0'),
    'h': 'HH',

    'j': ('T Z',
          '1 0'),
    'q': ('T CH HH',
          '0.5 0.5 0.5'),
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
          '0.5 0.5 0.5'),
    's': 'S',

    'y': 'Y',
    'w': 'W'
}

RIMES_TO_ARPA_DURATIONS = {
    'ɨ': 'UH1',
    'a': ('AH1 AA1 ,',
          '1.5  0 0'),
    'ai': 'AY1',
    'an': ('AH1 N',
           '1 1'),
    'ang': ('AH1 NG',
            '1 1'),
    'ao': 'AA1 W',

    'e': ('UH1 AH1 ,',
          '0 1 0'),
    'ei': 'EY1',
    'en': 'UH1 N',
    'eng': 'UH1 NG',

    'er': ('AH1 R',
           '1.5 1'),

    'i': ('IY1 ,',
          '1.5 0'),
    'ia': 'Y AH1',
    'iang': 'Y AH1 NG',
    'iao': 'Y AW1',
    'ie': 'Y EH1',
    'ien': 'Y EH1 N',
    'in': ('IH1 IY1 N',
           '1 0 1'),
    'ing': ('IH1 IY1 NG',
            '1 0 1'),
    'iong': 'Y OW1 NG',
    'iu': 'Y OW1',

    'o': ('AO1 W',
          ' 1  0'),
    'ou': 'OW1',
    'ong': 'OW1 NG',

    'u': ('W UW1 ,',
          '0  1  0'),
    'ua': 'W AA1',
    'uai': 'W AA1 Y',
    'uan': 'W AH1 N',
    'uang': 'W AH1 NG',
    'ui': 'W EY1',
    'un': 'W UH1 N',
    'uo': ('UH1 AO1',
           '0.5 0.5'),

    'v': ('UW1 IY1 ,',
          '0 1 0'),
    've': ('W EH1',
           '1 1'),
    'vn': ('UW1 IY1 N',
           '0  1 1'),
    'ven': ('W EH1 N',
            '1 1 1'),
}

CUSTOM_ARPA_DURATIONS = {
    'ʔa': (', AH1 AA1 ,',
           '0.2 1.5 1 0'),
    'ʔe': (', UH1 AH1 ,',
           '0.2 0.5 1 0'),

    'chɨ': ('CH HH R R',
            '1 0.5 1 1'),

    'da': ('T D AA1 ,',
           '1 0 1 0'),
    'di': ('T T D IY1 ,',
           '1 0 0 1.5 0'),
    'die': ('T T D Y EH1',
            '1 0 0 1  1'),
    'din': ('T T D IH1 IY1 N',
            '1 0 0  1   0  1'),
    'ding': ('T T D IH1 IY1 NG',
             '1 0 0  1   0  1'),
    'diou': ('T T D Y OW1',
             '1 0 0 1 1'),
    'du': ('T D UW1 W',
           '1 0 1 0'),
    'duen': ('T T D W UH1 N',
             '0 1 0 1 1 1'),

    'gu': ('K G W UW1',
           '1 0 0 1'),

    'houng': ('HH OW1 UW1 NG',
              '1 0.5 0.5 1'),
    'hu': ('HH HH UW1',
           '1 0 1'),

    'lv': ('L HH W IY1 ,',
           '0.5 0 0 1 0'),

    'neng': ('N AH1 UH1 NG',
             '1 0 1 1'),
    'noung': ('N UH1 OW1 NG',
              '1 0 1 1'),
    'qv': ('T SH HH W IY1 ,',
           '0.5 0.5 0.5 0 1 0'),
    'qvn': ('T SH HH UW1 IY1 N ,',
            '0.5 0.5 0 0 1 1 0'),
    'rɨ': ('ZH R',
           '1  3'),
    'shɨ': 'SH R R',

    'tuo': ('TH T HH UH1 AO1',
            '0 1 0.5 0.5 0.5'),

    'wa': 'W AA1',
    'wu': ('W UW1 W',
           '1 1 0'),
    'xve': 'SH W EH1',
    'xven': 'SH W EH1 N',
    'ye': 'Y EH1',
    'yen': 'Y EH1 N',
    'yve': ('Y UH1 EH1',
            '1 0.5 0.5'),
    'yven': ('Y UH1 EH1 N',
             '1 0.5 0.5 1'),
    'yvn': ('Y UH1 Y N',
            '1 1 0 1'),
    'zheng': ('T SH AH1 UH1 NG',
              '1 0 0 1 1'),
    'zhɨ': ('CH R R',
            '0.5 1 1'),
    'zɨ': ('T Z UH1',
           '1 0.5 0.5'),
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
    2: [-1.0, 2.0],
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
MAX_PITCH_CHANGE = 2.5

SPLIT_VOWELS = {'R'} | ARPA_VOWELS
NUCLEUS_VOWELS = {'W', 'Y'} | SPLIT_VOWELS
PUNC_CONVERSION = {
    '，': ',',
    '。': '.',
    '？': '?',
    '！': '!'
}

INITIALS = set(INITIALS_TO_ARPA_DURATIONS)
RIMES = set(RIMES_TO_ARPA_DURATIONS)
PINYIN_WITH_TONE_REGEX = re.compile(
    rf"({'|'.join(INITIALS)})({'|'.join(RIMES)})?([1-5])"
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
            nucleus_tone_only=False,
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
        self.check_consistency()

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

        self.check_consistency()

    @staticmethod
    def convert_hanzi_pinyin(hans):
        # pypinyin doesn't handle the sandhi properly!
        pinyin_list = lazy_pinyin(hans, Style.TONE3, neutral_tone_with_five=True, tone_sandhi=False)
        num_words = len(pinyin_list)
        # For simplicity convert alternate tone-3 sandhi so that 3-3-3-3-3 becomes 2-3-2-3-2
        # The rules are more complicated but no one implements it correctly
        with_sandhi = []
        i = 0
        while i < num_words:
            pinyin = pinyin_list[i]
            if pinyin.endswith('3'):
                add = []
                while i < num_words and pinyin_list[i].endswith('3'):
                    add.append(pinyin_list[i])
                    i += 1
                for j in range(len(add) - 2, -1, -2):
                    add[j] = add[j][:-1] + '2'
                with_sandhi.extend(add)
            else:
                with_sandhi.append(pinyin)
                i += 1
        # Adjust for 不 and 一
        for i, han in enumerate(hans[:-1]):
            if han == '不' and with_sandhi[i + 1].endswith('4'):
                bu_pinyin = with_sandhi[i]
                with_sandhi[i] = bu_pinyin[:-1] + '2'
            elif han == '一':
                if i == 0 or hans[i - 1] not in '〇零一二三四五六七八九十':
                    yi_pinyin = with_sandhi[i]
                    if with_sandhi[i + 1].endswith('4'):
                        with_sandhi[i] = yi_pinyin[:-1] + '2'
                    else:
                        with_sandhi[i] = yi_pinyin[:-1] + '4'
        # Regularize the pinyin (bo -> buo, ju -> jv, lian -> lien, rui -> ruei, sun -> suen, zun -> zuen)
        regularized = [regularize_pinyin(pinyin) for pinyin in with_sandhi]
        return regularized


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

    def convert_pinyin_arpa(self, pinyin_list, batch_i=0):
        tones = []
        arpas = []
        arpa_lens = []
        d_factor = []
        for pinyin in pinyin_list:
            if pinyin in PUNC_CONVERSION:
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

        # arpas is now a list of word_arpas and punctuation e.g. [[T S AW1], [AH1, N], '.']
        joined_arpas = [arpa for word_arpas in arpas for arpa in word_arpas]
        all_tones = [0] * len(joined_arpas)
        arpa_offsets = list(itertools.accumulate(arpa_lens, initial=0))
        d_split_factor = [1] * len(d_factor)

        # Set d_split_factor
        for word_arpas, tone, offset, pinyin in zip(arpas, tones, arpa_offsets[:-1], pinyin_list):
            if isinstance(word_arpas, str):  # this is punctuation
                continue

            for i, arpa in enumerate(word_arpas):
                if arpa in SPLIT_VOWELS and d_factor[offset+i] and not (arpa == 'R' and pinyin.startswith('r')):
                    # Split if vowel and on duration > 0; but do not split pinyin r-
                    d_split_factor[offset + i] = self.tone_duration_split[tone]

        # Set pitch_values and p_mod_fns
        pitch_values = [[]] * len(d_factor)
        combine_fns = [None] * len(d_factor)
        last_pitch = 0.0
        for word_i, (word_arpas, tone, offset) in enumerate(zip(arpas, tones, arpa_offsets[:-1])):

            if isinstance(word_arpas, str):  # punctuation should take last pitch
                pitch_values[offset] = [last_pitch]
                continue

            nucleus_start = 0
            while word_arpas[nucleus_start] not in NUCLEUS_VOWELS:
                nucleus_start += 1
            nucleus_end = nucleus_start + 1
            while nucleus_end < len(word_arpas) and word_arpas[nucleus_end] in NUCLEUS_VOWELS:
                nucleus_end += 1
            nucleus_start += offset
            nucleus_end += offset

            if self.nucleus_tone_only:
                tone_start = nucleus_start
                tone_end = nucleus_end
            else:
                tone_start = offset
                tone_end = arpa_offsets[word_i + 1]

            for i in range(tone_start, tone_end):
                all_tones[i] = tone

            if tone == 5:
                for i in range(tone_start, tone_end):
                    d_factor[i] = self.tone5_d_factor * d_factor[i]
                    pitch_values[i] = self.tone_contours[5]
                    combine_fns[i] = self.tone_combine_fns[5]
                last_pitch = pitch_values[i][-1]
                continue

            contour = self.tone_contours[tone]
            combine_fn = self.tone_combine_fns[tone]
            refs = len(contour)
            if refs == 1:
                for i in range(tone_start, tone_end):
                    pitch_values[i] = contour[:]  # take the slice or modifying pitch_values[i] will affect contour!!
                    combine_fns[i] = combine_fn
            else:
                has_initial = int(tone_start < nucleus_start)  # for self.nucleus_tone_only = False
                has_final = int(tone_end > nucleus_end)
                split_offsets = list(
                    itertools.accumulate(d_split_factor[nucleus_start:nucleus_end], initial=has_initial)
                )
                splits = split_offsets[-1]
                splits += has_final
                assert splits > 1, 'Total d_split_factor across phonemes having tones must be >1 to interpolate pitch'
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
            last_pitch = pitch_values[i][-1]

        # Fill in Nones by interpolating if initials / finals have no tone
        if self.nucleus_tone_only:
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
            for pitch_value in pitch_values:
                assert pitch_value, str(pitch_values)

        # Smooth abrupt changes
        if self.max_pitch_change:
            for word_start, word_end in zip(arpa_offsets[1:-1], arpa_offsets[2:]):
                last_pitch = pitch_values[word_start-1][-1]
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

        p_mod_fns = self.get_p_mod_fns(pitch_values, combine_fns, batch_i)
        return joined_arpas, all_tones, d_factor, d_split_factor, pitch_values, p_mod_fns

    def convert_hanzi_arpa(self, hans, batch_i=0):
        return self.convert_pinyin_arpa(self.convert_hanzi_pinyin(hans), batch_i)

    @staticmethod
    def get_p_mod_fns(pitch_values, combine_fns, batch_i=0):
        p_mod_fns = {}
        for i, (pitch_value, combine_fn) in enumerate(zip(pitch_values, combine_fns)):
            if pitch_value[0] is not None:
                p_mod_fns[(batch_i, i)] = mod_fn_generator(pitch_value, combine_fn)
        return p_mod_fns


    def gen_inputs(self, chinese, device=DEVICE):
        if isinstance(chinese, str):
            arpas, tones, d_factor, d_split_factor, pitch_values, p_mod_fns = self.convert_hanzi_arpa(chinese)
        elif isinstance(chinese, list):
            arpas, tones, d_factor, d_split_factor, pitch_values, p_mod_fns = self.convert_pinyin_arpa(chinese)
        else:
            raise NotImplementedError('chinese must be string of characters or list of regularized pinyin')
        print_numseq(arpas=arpas, d_factor=d_factor, d_split_factor=d_split_factor, tones=tones, pitch_values=pitch_values)
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
            vowel_duration=(9.0, 15.0),
            arpa_in_filename=True,
            device=DEVICE,
    ):
        if pac_update_dict is None:
            pac_update_dict = {}
        if pac_update_dict:
            self.update(**pac_update_dict)
        if inputs is None:
            arpas, tones, d_factor, d_split_factor, pitch_values, p_mod_fns = self.gen_inputs(chinese)
        else:
            arpas, tones, d_factor, d_split_factor, pitch_values, p_mod_fns = inputs

        d_factor = d_factor * overall_d_factor
        min_duration = vowel_duration[0] * overall_d_factor
        max_duration = vowel_duration[1] * overall_d_factor
        d_mod_fns = get_d_mod_fns(
            d_split_factor, duration_split_fn=duration_even_split, min_duration=min_duration, max_duration=max_duration
        )
        infer_kwargs = {
            'd_split_factor': d_split_factor,
            'p_factor': None,
            'd_factor': d_factor,
            'e_factor': None,
            'd_mod_fns': d_mod_fns,
            'p_mod_fns': p_mod_fns,
            'e_mod_fns': None,
        }
        if infer_overrides is not None:
            for x in infer_overrides.keys():
                infer_kwargs[x] = infer_overrides[x]
        phone_ids = torch.tensor(self.token_id_converter.tokens2ids(arpas), dtype=torch.int32, device=device)
        with torch.no_grad():
            wav_modified, _, _ = self.tts_inference_fn(
                    text=phone_ids.unsqueeze(0), text_lengths=torch.tensor([len(phone_ids)], device=device),
                    verbose=True,
                    **infer_kwargs,
                )
            filename = ' '.join(chinese) if isinstance(chinese, list) else chinese
            if arpa_in_filename:
                filename = f'{filename}-{"".join(arpas)}'
            sf.write(save_dir / f'{filename}.wav', wav_modified.squeeze().cpu().numpy(), 22050, "PCM_16")
        return arpas, tones, d_factor, d_split_factor, pitch_values, p_mod_fns


def all_tones(pinyin):
    with_tones = []
    if isinstance(pinyin, str):
        pinyin_list = pinyin.split(' ')
    else:
        pinyin_list = pinyin
    for p in pinyin_list:
        if p[0] in 'aoe':
            p = NULL_INITIAL + p
        with_tones.extend(p + str(x) for x in range(1,5))
    return with_tones

