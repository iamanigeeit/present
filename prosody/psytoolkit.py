import shutil

import os
from shutil import copyfile
from prosody.utils.utils import clean_filename
from pathlib import Path
import re
from inspect import cleandoc

HOSTING_URL = 'https://present2024.web.app'
HOME = Path.home()
PROJECTS_DIR = HOME / 'PycharmProjects'
FIREBASE_DIR = HOME / 'firebase_present2024' / 'public'


def make_durations_string():
    sentences = [
        "Suuuuuuure, I'm coming!",
        'This is a craaaaaaazy way to do it.',
        'Its too faaaaaaar.',
        'Waaaaaaa, this is a big prize.',
        'That could be a lo~~~~~~ng way!',
        "Cooooooool, let's go.",
        'Then we will dieeeeeee.',
        'Yesterdays lunch was reaaaaaaally good.',
        "Nooooooo, i won't give it to you.",
        'Will be fiiiiiiine.'
    ]
    s = cleandoc(
        '''
        l: {suffix}_{clean_text}
        a: {hosting_url}/durations/{filename}
        t: range
        q: 
        - {{min=1,max=5,by=0.25,start=3,left=Totally Unnatural,right=Totally Natural}}
        '''
    )
    for sentence in sentences:
        clean_text = clean_filename(sentence)
        for suffix in ['psola', 'present', 'jets']:
            filename = f'{clean_text}.{suffix}.wav'
            print(s.format(suffix=suffix, clean_text=clean_text, hosting_url=HOSTING_URL, filename=filename) + '\n')


def make_questions_string():
    questions = [
        'What are you working on?',
        'Well, how does it look?',
        'Is that Mister Edna Kent?',
        'Did you bring some lunch with you?',
        'Who taught you to put on make-up?',
        'Excuse me, could you tell me where you have got that music book?',
        'What do I need to take with me?',
        'Can you come to my office this afternoon at three o clock?',
        'Can I help you?',
        'You mean the boy who felt carsick just now?',
    ]
    s = cleandoc(
        '''
        l: {suffix}_{clean_text}
        a: {hosting_url}/questions/{filename}
        t: range
        q: 
        - {{min=1,max=5,by=0.25,start=3,left=Totally Unnatural,right=Totally Natural}}
        '''
    )
    for qn in questions:
        clean_text = clean_filename(qn)
        for suffix in ['dailytalk', 'present', 'jets']:
            filename = f'{clean_text}.{suffix}.wav'
            print(s.format(suffix=suffix, clean_text=clean_text, hosting_url=HOSTING_URL, filename=filename) + '\n')


long_surprise_texts = [
    'Why do all your coffee mugs have numbers on the bottom?',
    'Okay, so you were trying to play bad this whole time.',
    "Rach, I'm sorry, but you didn't give me any contracts!"
]
short_surprise_texts = [
    "They're not listening to me?",
    'Wow, you guys, this is big.',
    "It's nine o'clock in the morning!",
]
long_sad_texts = [
    "Oh come on, what am I gonna do, its been hours and it won't stop crying.",
    'I guess I just figured that somewhere down the road, we would be on again.',
    'No, I had to return it to the costume place.'
]
short_sad_texts = [
    'Can we please turn this off?',
    "I mean, no, you're right.",
    "Well it's very unsettling."
]
input_texts = long_surprise_texts + short_surprise_texts + long_sad_texts + short_sad_texts

short_surprise_ref = [
    PROJECTS_DIR / 'ESD_22050_en/0016/Surprise/test/0016_001449.wav',
    PROJECTS_DIR / 'ESD_22050_en/0017/Surprise/test/0017_001447.wav',
    PROJECTS_DIR / 'ESD_22050_en/0018/Surprise/test/0018_001421.wav',
]
long_surprise_ref = [
    PROJECTS_DIR / 'ESD_22050_en/0016/Surprise/test/0016_001450.wav',
    PROJECTS_DIR / 'ESD_22050_en/0017/Surprise/test/0017_001428.wav',
    PROJECTS_DIR / 'ESD_22050_en/0018/Surprise/test/0018_001440.wav'
]
short_sad_ref = [
    PROJECTS_DIR / 'ESD_22050_en/0016/Sad/test/0016_001091.wav',
    PROJECTS_DIR / 'ESD_22050_en/0017/Sad/test/0017_001086.wav',
    PROJECTS_DIR / 'ESD_22050_en/0018/Sad/test/0018_001092.wav',
]
long_sad_ref = [
    PROJECTS_DIR / 'ESD_22050_en/0016/Sad/test/0016_001090.wav',
    PROJECTS_DIR / 'ESD_22050_en/0017/Sad/test/0017_001078.wav',
    PROJECTS_DIR / 'ESD_22050_en/0018/Sad/test/0018_001084.wav'
]
ref_wavs = short_surprise_ref + long_surprise_ref + short_sad_ref + long_sad_ref


def make_transfer_string():
    s = '''
l: Prosody_transfer_{i}
t: info
q: Reference audio {i}
a: {hosting_url}/xfer/reference/{ref_filename}

l: Prosody_transfer_{i}_qn
a: {hosting_url}/xfer/generspeech/{filename}.wav {hosting_url}/xfer/present/{filename}.wav
t: range
q: Compared to reference audio {i}...
- {{min=-3,max=3,by=0.5,start=0,left=1st has much closer style to reference,right=2nd has much closer style to reference}}
    '''
    for i, (text, ref) in enumerate(zip(input_texts, ref_wavs), 1):
        ref_filename = ref.parts[-1]
        # copyfile(ref, f'/home/perry/firebase_asru/public/xfer/reference/{ref_filename}')
        filename = f'{clean_filename(text)}_ref{ref.stem}'
        print(s.format(filename=filename, ref_filename=ref_filename, hosting_url=HOSTING_URL, i=i))


MANDARIN_UTTS = '''
    SSB11260212.wav	中国人现在是美国房地产遥遥领先的最大外国买家|zhong1 guo2 ren2 xian4 zai4 shi4 mei3 guo2 fang2 di4 chan3 yao2 yao2 ling3 xian1 de5 zui4 da4 wai4 guo2 mai3 jia4
    SSB11260231.wav	比较节能的产品才会有补贴|bi3 jiao4 jie2 neng2 de5 chan2 pin3 cai2 hui4 you2 bu3 tie1
    SSB11350152.wav	一年以来持续攀升的待售面积也出现首次下降|yi4 nian2 yi3 lai2 chi2 xu4 pan1 sheng1 de5 dai4 shou4 mian4 ji1 ye3 chu1 xian4 shou3 ci4 xia4 jiang4
    SSB13220045.wav	这头大象你是不可能说得动的|zhe4 tou2 da4 xiang4 ni3 shi4 bu4 ke3 neng2 shuo1 de5 dong4 de5
    SSB14520196.wav	一般跳绳双脚离地高度为二点五公分到五公分|yi4 ban1 tiao4 sheng2 shuang1 jiao3 li2 di4 gao1 du4 wei2 er4 dian2 wu3 gong1 fen1 dao4 wu3 gong1 fen1
    SSB17390336.wav	未来这两幅相邻地块很可能同时开发|wei4 lai2 zhe4 liang3 fu2 xiang1 lin2 di4 kuai4 hen2 ke3 neng2 tong2 shi2 kai1 fa1
    SSB17390210.wav	中秋节不回来就算了|zhong1 qiu1 jie2 bu4 hui2 lai2 jiu4 suan4 le5
    SSB17590214.wav	告诉我他对我抱有极大的信心|gao4 su4 wo3 ta1 dui4 wo3 bao4 you3 ji2 da4 de5 xin4 xin1
    SSB16300072.wav	意识到束缚才能享受无期徒刑中清醒的快乐|yi4 shi2 dao4 shu4 fu4 cai2 neng2 xiang3 shou4 wu2 qi1 tu2 xing2 zhong1 qing1 xing3 de5 kuai4 le4
    SSB07020392.wav	第一届全国青年运动会射击比赛进入第四天|di4 yi1 jie4 quan2 guo2 qing1 nian2 yun4 dong4 hui4 she4 ji1 bi3 sai4 jin4 ru4 di4 si4 tian1
    SSB07020429.wav	我们之间的差异再也不像我们的长相这样差距如此大|wo3 men5 zhi1 jian1 de5 cha1 yi2 zai4 ye3 bu2 xiang4 wo3 men5 de5 zhang3 xiang4 zhe4 yang4 cha1 ju4 ru2 ci3 da4
    SSB13020205.wav	什么时候开始考虑出国留学|shen2 me5 shi2 hou4 kai1 shi2 kao3 lv4 chu1 guo2 liu2 xue2
    SSB07020005.wav	你猜咱们这位慈善的买卖人现在开着什么|ni3 cai1 zan2 men5 zhe4 wei4 ci2 shan4 de5 mai3 mai4 ren2 xian4 zai4 kai1 zhe5 shen3 me5
    SSB10550102.wav	没吃台湾麻辣锅就等于没来过台湾|mei2 chi1 tai2 wan1 ma2 la4 guo1 jiu4 deng3 yu2 mei2 lai2 guo4 tai2 wan1
    SSB10550022.wav	该超市的一工作人员不断地安慰顾客|gai1 chao1 shi4 de5 yi4 gong1 zuo4 ren2 yuan2 bu2 duan4 de5 an1 wei4 gu4 ke4
'''

to_copy = {
    'aishell': PROJECTS_DIR / 'datasets/data_aishell3/test/wav',
    'present': PROJECTS_DIR / 'present/prosody/outputs/tts_train_jets_raw_phn_tacotron_g2p_en_no_space/aishell3',
    'nopitch': PROJECTS_DIR / 'present/prosody/outputs/tts_train_jets_raw_phn_tacotron_g2p_en_no_space/aishell3_nopitch',
}

def make_mandarin_string():
    s = cleandoc(
        '''
        l: {dirname}_{i}
        a: {hosting_url}/{dirname}/{filename}
        t: range
        q: 
        - {{min=1,max=5,by=0.25,start=3,left=Totally Unnatural,right=Totally Natural}}
        '''
    )

    for dirname in to_copy:
        os.makedirs(FIREBASE_DIR / dirname, exist_ok=True)
    utts = []
    for line in MANDARIN_UTTS.strip().splitlines():
        filename = line.strip().split(maxsplit=1)[0]
        utts.append(filename)
    for dirname, source_dir in to_copy.items():
        for i, filename in enumerate(utts):
            # if dirname == 'aishell':
            #     source_dir /= filename[:7]
            # shutil.copy(source_dir / filename, FIREBASE_DIR / dirname)
            print(s.format(dirname=dirname, i=i, hosting_url=HOSTING_URL, filename=filename) + '\n')


def make_html_table():
    tablestr = '''
    <table>
      <thead>
          <th>Input Text</th>
          <th>AISHELL</th>
          <th>PRESENT</th>
          <th>PRESENT (no pitch)</th>
        </tr>
      </thead>
      <tbody>
      {rows}
      </tbody>
    </table>
    '''
    rowstr = '''
    <tr>
        <td>{text}</td>
        {cells}
    </tr>
    '''
    cellstr = '''<td><audio controls src="{dirname}/{filename}"/></td>
    '''
    rows = []
    for i, line in enumerate(MANDARIN_UTTS.strip().splitlines()):
        filename, chinese = line.strip().split(maxsplit=1)
        hans, _ = chinese.split('|', maxsplit=1)
        cells = []
        for dirname in to_copy:
            cell = cellstr.format(dirname=dirname, filename=filename)
            cells.append(cell)
        row = rowstr.format(text=hans, cells=''.join(cells))
        rows.append(row)
    table = tablestr.format(rows=''.join(rows))
    print(table)


# make_durations_string()
# make_questions_string()
# make_transfer_string()
make_mandarin_string()
# make_html_table()
