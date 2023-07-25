
from shutil import copyfile
from prosody.utils.utils import clean_filename
from pathlib import Path
import re
from inspect import cleandoc

hosting_url = 'https://asru2023.web.app'

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
            print(s.format(suffix=suffix, clean_text=clean_text, hosting_url=hosting_url, filename=filename) + '\n')


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
            print(s.format(suffix=suffix, clean_text=clean_text, hosting_url=hosting_url, filename=filename) + '\n')

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
        '/home/perry/PycharmProjects/ESD_22050_en/0016/Surprise/test/0016_001449.wav',
        '/home/perry/PycharmProjects/ESD_22050_en/0017/Surprise/test/0017_001447.wav',
        '/home/perry/PycharmProjects/ESD_22050_en/0018/Surprise/test/0018_001421.wav',
    ]
long_surprise_ref = [
    '/home/perry/PycharmProjects/ESD_22050_en/0016/Surprise/test/0016_001450.wav',
    '/home/perry/PycharmProjects/ESD_22050_en/0017/Surprise/test/0017_001428.wav',
    '/home/perry/PycharmProjects/ESD_22050_en/0018/Surprise/test/0018_001440.wav'
]
short_sad_ref = [
    '/home/perry/PycharmProjects/ESD_22050_en/0016/Sad/test/0016_001091.wav',
    '/home/perry/PycharmProjects/ESD_22050_en/0017/Sad/test/0017_001086.wav',
    '/home/perry/PycharmProjects/ESD_22050_en/0018/Sad/test/0018_001092.wav',
]
long_sad_ref = [
    '/home/perry/PycharmProjects/ESD_22050_en/0016/Sad/test/0016_001090.wav',
    '/home/perry/PycharmProjects/ESD_22050_en/0017/Sad/test/0017_001078.wav',
    '/home/perry/PycharmProjects/ESD_22050_en/0018/Sad/test/0018_001084.wav'
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
        ref_filename = Path(ref).parts[-1]
        # copyfile(ref, f'/home/perry/firebase_asru/public/xfer/reference/{ref_filename}')
        filename = f'{clean_filename(text)}_ref{Path(ref).stem}'
        print(s.format(filename=filename, ref_filename=ref_filename, hosting_url=hosting_url, i=i))


make_durations_string()
make_questions_string()
# make_transfer_string()
