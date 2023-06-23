import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def transfer_prosody(
        duration_pred, pitch_pred, energy_pred, pitch_range, energy_range,
        p_reference, e_reference, e_threshold, sil_threshold,
        is_vowels, is_consonants, avg_energy_window, change_durations,
    ):
        avg_energy = compute_avg_energy(e_reference, duration_pred, window=avg_energy_window)
        energy = avg_energy.clone()
        keep_frames = energy > e_threshold
        energy = energy[keep_frames]
        pitch = p_reference.squeeze()[keep_frames]
        pitch = adjust_range(pitch, pitch_range)
        energy = adjust_range(energy, energy_range)
        peaks = find_peaks(energy)
        num_peaks = len(peaks)
        num_vowels = sum(is_vowels).item()
        if num_peaks < num_vowels:
            repeats = compute_repeats(num_vowels, num_peaks)
            peaks, pitch, energy = repeat_peaks(repeats, peaks, num_peaks, pitch, energy)
            num_peaks = num_vowels
        elif num_peaks > num_vowels:
            peaks = retain_peaks(peaks, len(energy), duration_pred, is_vowels)
        energy_new = energy_pred.clone()
        energy_new[is_vowels] = energy[peaks]
        pitch_new = pitch_pred.clone()
        pitch_new[is_vowels] = pitch[peaks]
        duration_start = shift_pad(duration_pred.cumsum(0), shift=1)
        phone_positions = duration_start + duration_pred / 2
        phone_positions = add_start_end(phone_positions, 0, duration_pred.sum())
        vowel_indices = torch.nonzero(is_vowels).squeeze()
        vowel_indices = add_start_end(vowel_indices, -1, len(duration_pred))
        phone_positions = add_start_end(phone_positions, 0, duration_pred.sum())
        peaks = add_start_end(peaks, 0, len(energy))
        for start_vowel, end_vowel, start_peak, end_peak in zip(
                vowel_indices[:-1], vowel_indices[1:], peaks[:-1], peaks[1:]):
            start_peak = start_peak.item()
            end_peak = end_peak.item()
            peak_interval = end_peak - start_peak
            # print('\n', start_vowel.item(), 'to', end_vowel.item(), 'peak:', start_peak, end_peak)
            start_position = 0 if start_vowel == -1 else phone_positions[start_vowel + 1].item()
            end_position = phone_positions[end_vowel + 1].item()
            # print(start_position, end_position)
            interval = end_position - start_position
            for phone_i in range(start_vowel, end_vowel):  # phone index
                if is_consonants[phone_i]:
                    consonant_position = phone_positions[phone_i + 1].item()
                    consonant_frame = start_peak + (consonant_position - start_position) / interval * peak_interval
                    floor_frame = int(consonant_frame)
                    consonant_pitch = pitch[floor_frame] * (floor_frame + 1 - consonant_frame) + \
                                      pitch[floor_frame + 1] * (consonant_frame - floor_frame)
                    consonant_energy = energy[floor_frame] * (floor_frame + 1 - consonant_frame) + \
                                       energy[floor_frame + 1] * (consonant_frame - floor_frame)
                    pitch_new[phone_i] = consonant_pitch
                    energy_new[phone_i] = consonant_energy
        if change_durations:
            energy_mask = avg_energy > sil_threshold
            start = 0
            while not energy_mask[start]:
                start += 1
            end = -1
            while not energy_mask[end]:
                end -= 1
            ref_actual_len = len(avg_energy) - start + end + 1
            ref_syllable_frames = ref_actual_len / num_peaks
            pred_syllable_frames = duration_pred.sum() / num_vowels
            print(ref_actual_len, num_peaks, ref_syllable_frames, pred_syllable_frames)
            duration_new = duration_pred / pred_syllable_frames * ref_syllable_frames
        else:
            duration_new = duration_pred
        return duration_new, pitch_new, energy_new


def compute_avg_energy(e_reference, duration_pred, window):
    if window <= 0:
        kernel_size = int(torch.round(torch.mean(duration_pred.type(torch.float32))).item())
    else:
        kernel_size = window
    padding = kernel_size // 2
    moving_avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)
    avg_energy = moving_avg(e_reference.squeeze(-1)).squeeze()
    avg_energy[:padding] = avg_energy[padding]  # left-pad to first avg value, not 0
    avg_energy[-padding:] = avg_energy[-padding-1]  # right-pad to the last avg value, not 0
    if kernel_size % 2 == 0:  # remove extra padding
        avg_energy = avg_energy[:-1]
    return avg_energy


def adjust_range(ref, range_limits):
    range_min, range_max = range_limits
    ref_min, ref_max = ref.min(), ref.max()
    if ref_min < range_min:
        ref = ref - ref_min + range_min
    if ref_max > range_max:
        ref = ref - ref_max + range_max
    return ref

def compute_repeats(num_vowels, num_peaks):
    assert num_peaks > 1
    middle_repeats = (num_vowels - 1) // num_peaks - 1
    remaining_vowels = num_vowels - middle_repeats * num_peaks
    last = remaining_vowels // 2
    first = remaining_vowels - last
    return [first] + middle_repeats * [num_peaks] + [last]


def repeat_peaks(repeats, peaks, num_peaks, pitch, energy):
    base_frames = len(energy)
    first = repeats[0]
    if first < num_peaks:
        frame_count = (peaks[first-1] + peaks[first]).item() // 2 + 1
        peaks_new = [peaks[:first]]
        pitch_new = [pitch[:frame_count]]
        energy_new = [energy[:frame_count]]
    else:
        frame_count = base_frames
        peaks_new = [peaks]
        pitch_new = [pitch]
        energy_new = [energy]
    for repeat in repeats[1:-1]:
        peaks_new.append(peaks + frame_count)
        pitch_new.append(pitch)
        energy_new.append(energy)
        frame_count += base_frames
    last = repeats[-1]
    if last < num_peaks:
        start_frame = peaks[num_peaks-last-1].item() + (peaks[num_peaks-last] - peaks[num_peaks-last-1]).item() // 2
        peaks_new.append(peaks[-last:] + frame_count - start_frame)
        pitch_new.append(pitch[start_frame:])
        energy_new.append(energy[start_frame:])
    else:
        peaks_new.append(peaks + frame_count)
        pitch_new.append(pitch)
        energy_new.append(energy)
    return torch.cat(peaks_new), torch.cat(pitch_new), torch.cat(energy_new)


def retain_peaks(peaks, base_frames, duration_pred, is_vowels):
    peak_fracs = peaks / base_frames
    duration_end = duration_pred.cumsum(0)
    total_duration = duration_end[-1]
    duration_start = shift_pad(duration_end, shift=1)
    phone_positions = duration_start + duration_pred / 2
    vowel_positions = phone_positions[is_vowels]
    vowel_fracs = vowel_positions / total_duration
    peaks_chosen, _ = find_closest_peaks(
        remaining_peaks=peaks.tolist(),
        peaks_chosen=[],
        peak_fracs=peak_fracs,
        vowel_fracs=vowel_fracs,
        error=torch.tensor(0.0, device=peaks.device)
    )
    return torch.tensor(peaks_chosen, device=peaks.device)


def find_closest_peaks(remaining_peaks, peaks_chosen, peak_fracs, vowel_fracs, error):
    if len(vowel_fracs) == 0:  # skip all the remaining peaks
        return peaks_chosen, error
    if len(peak_fracs) == len(vowel_fracs):
        return peaks_chosen + remaining_peaks, error + sum(abs(peak_fracs - vowel_fracs))
    if len(peak_fracs) < len(vowel_fracs):  # This should not happen!
        return peaks_chosen + remaining_peaks, 9999999999.
    else:
        use_peaks_chosen, use_error = find_closest_peaks(
            remaining_peaks=remaining_peaks[1:],
            peaks_chosen=peaks_chosen + remaining_peaks[0:1],
            peak_fracs=peak_fracs[1:],
            vowel_fracs=vowel_fracs[1:],
            error=error + abs(peak_fracs[0] - vowel_fracs[0])
        )
        skip_peaks_chosen, skip_error = find_closest_peaks(
            remaining_peaks=remaining_peaks[1:],
            peaks_chosen=peaks_chosen,
            peak_fracs=peak_fracs[1:],
            vowel_fracs=vowel_fracs,
            error=error
        )
        if use_error < skip_error:
            return use_peaks_chosen, use_error
        else:
            return skip_peaks_chosen, skip_error


def shift_pad(x, shift, pad=0):
    y = x.roll(shift)
    if shift < 0:
        y[shift:] = pad
    else:
        y[0:shift] = pad
    return y


def find_peaks(avg_energy):
    avg_energy_left = shift_pad(avg_energy, -1)
    avg_energy_right = shift_pad(avg_energy, 1)
    return torch.nonzero((avg_energy > avg_energy_left) & (avg_energy > avg_energy_right)).squeeze()

def add_start_end(x, start_value, end_value):
    y = torch.zeros(len(x) + 2, dtype=x.dtype, device=x.device)
    y[0] = start_value
    y[1:-1] = x
    y[-1] = end_value
    return y