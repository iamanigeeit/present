from tabulate import tabulate
import numpy as np
import torch

def mod_fn_generator(reference_points, combine_fn=None):
    def mod_fn(pred_val, split):
        refs = len(reference_points)
        count = split.item()
        if refs == 0:
            return pred_val
        elif refs == 1 or refs == count:
            mod_arr = reference_points
        else:
            if count == 1:
                mod_arr = [sum(reference_points) / refs]
            else:
                mod_arr = np.interp(np.arange(count) / (count-1) * (refs-1), list(range(refs)), reference_points)
        mod_tensor = torch.tensor(mod_arr, device=pred_val.device)
        if combine_fn is None:
            return mod_tensor
        else:
            return combine_fn(mod_tensor, pred_val)
    return mod_fn


def get_p_mod_fns(pitch_values, combine_fns=None, batch_i=0):
    p_mod_fns = {}
    if combine_fns is None or callable(combine_fns):
        combine_fn = combine_fns
        for i, pitch_value in enumerate(pitch_values):
            if pitch_value:
                p_mod_fns[(batch_i, i)] = mod_fn_generator(pitch_value, combine_fn)
    else:
        for i, (pitch_value, combine_fn) in enumerate(zip(pitch_values, combine_fns)):
            if pitch_value:
                p_mod_fns[(batch_i, i)] = mod_fn_generator(pitch_value, combine_fn)
    return p_mod_fns

def duration_even_split(min_duration=0.0, max_duration=100.0):
    def mod_fn(pred_val, split):
        duration = pred_val.item()
        if duration > 0.0:
            if duration < min_duration:
                duration = min_duration
            elif duration > max_duration:
                duration = max_duration
        spl = split.item()
        first = duration // spl
        if spl == 2:
            last = duration - first
            mod_arr = [first, last]
        else:
            last = first
            middle = duration - first - last
            mod_arr = [first] + [middle / (spl - 2)] * (spl - 2) + [last]
        return torch.tensor(mod_arr, device=pred_val.device)
    return mod_fn


def get_d_mod_fns(d_split_factor, duration_split_fn, min_duration=0, max_duration=100):
    d_mod_fns = {}
    for batch_i, d_split in enumerate(d_split_factor):
        phone_i = 0
        while phone_i < len(d_split):
            split = d_split[phone_i]
            add_i = []
            while split != 1:
                add_i.append(phone_i)
                phone_i += 1
                try:
                    split = d_split[phone_i]
                except IndexError:
                    break
            if add_i:
                for i in add_i:
                    d_mod_fns[(batch_i, i)] = duration_split_fn(min_duration / len(add_i), max_duration / len(add_i))
            else:
                phone_i += 1
    return d_mod_fns


def round_floats(float_list):
    for x in float_list:
        n = format(x, '.2f').rstrip('.0')
        yield n if n else '0'

def round_and_join(float_list):
    return '|'.join(round_floats(float_list))

def print_table(**kwargs):
    print(generate_table(**kwargs))

def generate_table(**kwargs):
    rows = {}
    for name, row in kwargs.items():
        if isinstance(row, torch.Tensor):
            r = torch.flatten(row).cpu().tolist()
            if torch.is_floating_point(row):
                rows[name] = list(round_floats(r))
            else:
                rows[name] = r
        elif isinstance(row, dict):
            rows.update(row)
        else:
            rows[name] = row
    if 'd_split_factor' in rows:
        d_split_factor = rows['d_split_factor']
        if 'duration_new' in rows:
            duration_new = rows['duration_new']
            duration = []
            start = 0
            for fac in d_split_factor:
                end = start + fac
                duration.append(round_and_join(duration_new[start:end]))
                start = end
            rows['duration_new'] = duration
        if 'pitch_new' in rows:
            pitch_new = rows['pitch_new']
            pitch = []
            start = 0
            for fac in d_split_factor:
                end = start + fac
                pitch.append(round_and_join(pitch_new[start:end]))
                start = end
            rows['pitch_new'] = pitch
        if 'pitch_values' in rows:
            flattened = []
            pitch_values = rows['pitch_values']
            for i, pitch_value in enumerate(pitch_values):
                if pitch_value is None:
                    flattened.append('')
                else:
                    flattened.append(round_and_join(pitch_value))
            rows['pitch_values'] = flattened
        # if 'p_mod_fns' in rows:
        #     p_mod_fns = rows['p_mod_fns']
        #     p_mods = []
        #     for i, fac in enumerate(d_split_factor):
        #         if (0, i) in p_mod_fns:
        #             p_mod = p_mod_fns[(0, i)](torch.Tensor([0.0]), torch.Tensor([fac])).cpu().tolist()
        #             p_mod = round_and_join(p_mod)
        #         else:
        #             p_mod = None
        #         p_mods.append(p_mod)
        #     rows['p_mod_fns'] = p_mods
    table = []
    for name, row in rows.items():
        table.append([name] + row)
    return tabulate(table, headers=list(range(-1, len(row)+1)), tablefmt='plain', stralign='center')
