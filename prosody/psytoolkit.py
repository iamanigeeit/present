s = '''
l: Prosody_transfer_{i}
t: info
q: Reference audio {i}
a: https://interspeech2023.web.app/transfer/reference/{ref_filename}

l: Prosody_transfer_{i}_qn
a: https://interspeech2023.web.app/transfer/generspeech/{filename}.wav https://interspeech2023.web.app/transfer/present/{filename}.wav
t: range
q: Compared to reference audio {i}...
- {{min=-3,max=3,by=1,start=0,left=1st is much better quality,right=2nd is much better quality}}
- {{min=-3,max=3,by=1,start=0,left=1st much closer to reference,right=2nd much closer to reference}}
'''

from shutil import copyfile
from pathlib import Path
import re

def make_transfer_string(texts, refs):
    i = 1
    for text, ref in zip(texts, refs):
        text = re.sub(r'[^A-Za-z ]', '', text).replace(' ', '_')
        filename = f'{text}-ref{Path(ref).stem}'
        ref_filename = Path(ref).parts[-1]
        print(s.format(filename=filename, ref_filename=ref_filename, i=i))
        i += 1