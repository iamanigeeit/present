import os
import math
import numpy as np
import torch
import librosa
import soundfile as sf
from parselmouth import praat, Data, Sound

from kaldiio import ReadHelper

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_all_feats(audio_file, pretrained_model, dtype=torch.float32, silence_db=0, device=DEVICE):
    pretrained_model.pitch_extract.use_token_averaged_f0 = False
    pretrained_model.energy_extract.use_token_averaged_energy = False
    with open(audio_file, 'rb') as f:
        audio, sample_rate = sf.read(f)
        target_sr = pretrained_model.feats_extract.fs
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sr)
        if silence_db:
            audio, _ = librosa.effects.trim(y=audio, top_db=silence_db)
    speech = torch.tensor(audio, dtype=dtype, device=device).unsqueeze(0)
    speech_lengths = torch.tensor([len(audio)], device=device)
    feats, feats_lengths = pretrained_model.feats_extract(speech, speech_lengths)
    pretrained_model.pitch_extract.use_continuous_f0 = False
    pitch, _ = pretrained_model.pitch_extract(speech, speech_lengths, feats_lengths=feats_lengths)
    voiced_frames = pitch.squeeze().nonzero().squeeze()
    pretrained_model.pitch_extract.use_continuous_f0 = True
    pitch, pitch_lengths = pretrained_model.pitch_extract(speech, speech_lengths, feats_lengths=feats_lengths)
    energy, energy_lengths = pretrained_model.energy_extract(speech, speech_lengths, feats_lengths=feats_lengths)
    return (
        speech, speech_lengths,
        feats, feats_lengths,
        pitch, pitch_lengths,
        energy, energy_lengths,
        voiced_frames
    )

def normalize_feats(pretrained_model, pitch, pitch_lengths, energy, energy_lengths):
    normed_pitch = pretrained_model.pitch_normalize(pitch.squeeze(-1), pitch_lengths)[0].unsqueeze(-1)
    normed_energy = pretrained_model.energy_normalize(energy.squeeze(-1), energy_lengths)[0].unsqueeze(-1)
    return normed_pitch, normed_energy


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


duration_str = '''File type = "ooTextFile"
Object class = "DurationTier"
xmin = {start_time} 
xmax = {end_time} 
points: size = 4 
points [1]:
    number = {bef_start_time} 
    value = 1 
points [2]:
    number = {start_time} 
    value = {d_fac} 
points [3]:
    number = {end_time} 
    value = {d_fac} 
points [4]:
    number = {aft_end_time} 
    value = 1 
'''
def td_psola_stretch(wav_path, out_path, durations, phone_i, d_fac):
    sound = Sound(wav_path)
    manipulation = praat.call(sound, "To Manipulation", 0.01, 50, 1000)
    total_frames = sum(durations)
    phone_start = sum(durations[:phone_i])
    phone_dur = durations[phone_i]
    start_time = sound.tmax * phone_start / total_frames
    end_time = sound.tmax * (phone_start + phone_dur) / total_frames
    bef_start_time = sound.tmax * (phone_start - 1) / total_frames
    aft_end_time = sound.tmax * (phone_start + phone_dur + 1) / total_frames
    with open('/tmp/tmpdur', 'w') as f:
        f.write(duration_str.format(start_time=start_time, end_time=end_time, bef_start_time=bef_start_time,
                                    aft_end_time=aft_end_time, d_fac=d_fac))
    new_duration_tier = Data.read('/tmp/tmpdur')
    praat.call([new_duration_tier, manipulation], 'Replace duration tier')
    new_sound = praat.call(manipulation, "Get resynthesis (overlap-add)")
    new_audio = new_sound.values[0]
    sf.write(out_path, new_audio, int(sound.sampling_frequency), "PCM_16")


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


def td_psola_shift(wav_path, out_path, p_mul):
    sound = Sound(wav_path)
    manipulation = praat.call(sound, "To Manipulation", 0.01, 75, 600)
    pitch_tier = praat.call(manipulation, "Extract pitch tier")
    praat.call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, p_mul)
    praat.call([pitch_tier, manipulation], "Replace pitch tier")
    new_sound = praat.call(manipulation, "Get resynthesis (overlap-add)")
    new_audio = new_sound.values[0]
    sf.write(out_path, new_audio, int(sound.sampling_frequency), "PCM_16")



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


def find_dataset_ranges(wav_paths, pretrained_tts, device=DEVICE):
    # glob('/home/perry/PycharmProjects/LJSpeech-1.1/wavs/*.wav')
    max_pitch, min_pitch = 0, 0
    max_energy, min_energy = 0, 0
    for w in wav_paths:
        _, _, _, _, pitch, _, energy, _ = extract_all_feats(w, pretrained_tts, normalize=True, device=device)
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


def read_xvectors(tts_dir, filename):
    filetype = filename.split('.')[-1]
    pwd = os.getcwd()
    os.chdir(tts_dir)
    with ReadHelper(f'{filetype}:{filename}') as reader:
        xvector_dict = {utt: xvector for utt, xvector in reader}
    os.chdir(pwd)
    return xvector_dict
