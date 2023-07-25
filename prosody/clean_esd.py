import os
from pathlib import Path
import re

ESD_ROOT = Path('/home/perry/PycharmProjects/ESD/')


# Clean up files
def clean_file(i, encoding):
    speaker = f'{i:04d}'
    new_txt = ESD_ROOT / speaker / f'{speaker}.txt'
    old_txt = Path(str(new_txt) + '.old')
    os.rename(new_txt, old_txt)
    with open(old_txt, encoding=encoding) as old_file:
        content = old_file.read()
        content = re.sub(r'\s*\n\s+', '\n', content)
    with open(new_txt, 'w', encoding='utf-8') as new_file:
        new_file.write(content)
    print(f'Cleaned speaker {i}')


for i in [1, 2, 4, 5, 9]:
    clean_file(i, encoding='GB2312')

for i in [3, 6, 7, 8, 10, 12, 13, 14, 18, 19]:
    clean_file(i, encoding='utf-16')

for i in [15, 20]:
    clean_file(i, encoding='ascii')

for i in [16, 17]:
    clean_file(i, encoding='ISO-8859-1')


# Find anomalous fields
for i in range(1, 21):
    speaker = '{:04d}'.format(i)
    txt = ESD_ROOT / speaker / f'{speaker}.txt'
    with open(txt) as f:
        for line in f:
            if line.count('\t') != 2:
                print(line.replace('\t', '||'))


# Compare for inconsistent sentences

def compare(i, j):
    print(i, j)
    speaker1 = '{:04d}'.format(i)
    speaker2 = '{:04d}'.format(j)
    txt1 = ESD_ROOT / speaker1 / f'{speaker1}.txt'
    txt2 = ESD_ROOT / speaker2 / f'{speaker2}.txt'
    sentences1 = []
    sentences2 = []
    with open(txt1) as f:
        for line in f:
            sentence = line.split('\t')[1]
            sentences1.append(sentence)
    with open(txt2) as f:
        for line in f:
            sentence = line.split('\t')[1]
            sentences2.append(sentence)
    for line_num, (s1, s2) in enumerate(zip(sentences1, sentences2), start=1):
        if s1 != s2:
            print(line_num, s1, '||', s2)

for i in range(1, 11):
    for j in range(10, i, -1):
        compare(i, j)

for i in range(11, 21):
    for j in range(20, i, -1):
        compare(i, j)
