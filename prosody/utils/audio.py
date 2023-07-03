import math
import numpy as np
import torch
import librosa
import soundfile as sf

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
