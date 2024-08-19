import soundfile as sf
from prosody.utils.utils import *
from espnet2.bin.tts_inference import Text2Speech
import os
from pathlib import Path

LJSPEECH_DIR = Path('/home/perry/PycharmProjects/prosody/egs2/ljspeech/tts1')
VCTK_DIR = Path('/home/perry/PycharmProjects/prosody/egs2/vctk/tts1')

device = 'cuda'

cwd = os.getcwd()
os.chdir(LJSPEECH_DIR)
pretrained_dir = LJSPEECH_DIR / "exp/tts_train_jets_raw_phn_tacotron_g2p_en_no_space"
pretrained_model_file = pretrained_dir / "train.total_count.ave_5best.pth"
pretrained_tts = Text2Speech.from_pretrained(
    train_config=pretrained_dir / "config.yaml",
    model_file=pretrained_model_file,
    device=device
)
pretrained_model = pretrained_tts.model
os.chdir(cwd)

with open('/home/perry/PycharmProjects/LJSpeech-1.1/wavs/LJ050-0162.wav', 'rb') as f:
    audio = sf.read(f)[0]

speech = torch.tensor(audio, device=device).unsqueeze(0)
speech_lengths = torch.tensor([len(audio)], device=device)
feats, feats_lengths = pretrained_model.feats_extract(speech, speech_lengths)
pitch, pitch_lengths = pretrained_model.pitch_extract(speech, speech_lengths, feats_lengths=feats_lengths)
energy, energy_lengths = pretrained_model.energy_extract(speech, speech_lengths, feats_lengths=feats_lengths)