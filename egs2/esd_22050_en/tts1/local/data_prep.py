import os
import sys
from pathlib import Path
import argparse

script = sys.argv[0]

esd_dir = 'downloads/ESD'
train_dirs = [
    "Angry/train", "Angry/evaluation", "Angry/test",
    "Happy/train", "Happy/evaluation", "Happy/test",
    "Neutral/train", "Neutral/evaluation", "Neutral/test"
]
valid_dirs = ["Sad/evaluation", "Surprise/evaluation"]
test_dirs = ["Sad/test", "Surprise/test"]
data_dir = "data"
lang = ['en']


def list_str(comma_str):
    return comma_str.split(',')


def get_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--esd_dir",
        default='downloads/ESD',
        type=str,
    )
    parser.add_argument(
        "--train_dirs",
        default=["train"],
        # default=train_dirs,
        type=list_str,
    )
    parser.add_argument(
        "--valid_dirs",
        type=list_str,
        default=["evaluation"],
        # default=valid_dirs,
    )
    parser.add_argument(
        "--test_dirs",
        default=["test"],
        # default=test_dirs,
        type=list_str,
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        type=str,
    )
    parser.add_argument(
        "--train_set",
        default="tr_no_dev",
        type=str,
    )
    parser.add_argument(
        "--valid_set",
        default="dev",
        type=str,
    )
    parser.add_argument(
        "--test_sets",
        default="eval1",
        type=str,
    )
    parser.add_argument(
        "--lang",
        default=['zh', 'en'],
        # default=lang,
        type=list_str,
    )
    return parser


def write_files(esd_path, speaker, corpus_dirs, scp_file, utt2spk_file, text_file):
    speaker_dir = esd_path / speaker
    utt2text = {}
    with open(speaker_dir / f'{speaker}.txt') as f:
        for line in f:
            utt, sentence, _ = line.split('\t')
            utt2text[utt] = sentence
    for corpus_dir in corpus_dirs:
        if '/' in corpus_dir:
            wav_dirs = [speaker_dir / corpus_dir]
        else:
            wav_dirs = sorted(speaker_dir.glob(f'*/{corpus_dir}'))
        for wav_dir in wav_dirs:
            wav_names = sorted(os.listdir(wav_dir))
            for wav_name in wav_names:
                utt = os.path.splitext(wav_name)[0]
                sentence = utt2text[utt]
                wav_path = wav_dir / wav_name
                scp_file.write(f'{utt} {wav_path}\n')
                utt2spk_file.write(f'{utt} {speaker}\n')
                text_file.write(f'{utt} {sentence}\n')


def main(args):
    zh_speakers = []
    en_speakers = []
    if 'zh' in args.lang:
        zh_speakers = [f'{i:04d}' for i in range(1, 11)]
    if 'en' in args.lang:
        en_speakers = [f'{i:04d}' for i in range(11, 21)]
    if not zh_speakers and not en_speakers:
        print(f'{script}: Language must be specified: "en", "zh" or "en zh"!')

    esd_path = Path(args.esd_dir)
    data_dir = Path(args.data_dir)

    for corpus_set, corpus_dirs in [
        (args.train_set, args.train_dirs),
        (args.valid_set, args.valid_dirs),
        (args.test_sets, args.test_dirs),
    ]:
        dataset_dir = data_dir / corpus_set
        os.makedirs(dataset_dir, exist_ok=True)
        with open(dataset_dir / 'wav.scp', 'w') as scp_file:
            with open(dataset_dir / 'utt2spk', 'w') as utt2spk_file:
                with open(dataset_dir / 'text', 'w') as text_file:
                    for speaker in zh_speakers:
                        write_files(esd_path, speaker, corpus_dirs, scp_file, utt2spk_file, text_file)
                    print(f'{script}: Finished writing {corpus_set} wav.scp, utt2spk, text for zh')
                    for speaker in en_speakers:
                        write_files(esd_path, speaker, corpus_dirs, scp_file, utt2spk_file, text_file)
                    print(f'{script}: Finished writing {corpus_set} wav.scp, utt2spk, text for en')

        if zh_speakers and en_speakers:
            with open(dataset_dir / 'utt2lang', 'w') as lang_file:
                with open(dataset_dir / 'utt2spk') as scp_file:
                    for line in scp_file:
                        utt = line.split(' ')[0]
                        speaker = utt.split('_')[0]
                        if int(speaker) < 11:
                            lang_file.write(f'{utt} zh\n')
                        else:
                            lang_file.write(f'{utt} en\n')
        print(f'{script}: Finished writing {corpus_set} utt2lang')


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)


