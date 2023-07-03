#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000
n_fft=2048

win_length=1200

fmin=60
fmax=1000

opts=
if [ "${fs}" -eq 24000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_config=conf/tuning/train_jets.yaml
inference_config=conf/decode.yaml

# If using MFA
n_shift=256
train_set=train-clean-100_phn
valid_set=dev-clean_phn
test_sets=test-clean_phn

cleaner=none
g2p=none # or g2p_en
local_data_opts="--trim_all_silence false" # trim all silence in the audio

# If not using MFA
#n_shift=300
#train_set=train-clean-100
#valid_set=dev-clean
#test_sets=test-clean
#
#cleaner=tacotron
#g2p=g2p_en_no_space # or g2p_en
#local_data_opts="--trim_all_silence true" # trim all silence in the audio

echo "$@"

./tts.sh \
    --ngpu 1 \
    --lang en \
    --feats_type raw \
    --use_sid true \
    --fmin ${fmin} \
    --fmax ${fmax} \
    --local_data_opts "${local_data_opts}" \
    --fs "${fs}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type phn \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --tts_task gan_tts \
    ${opts} "$@"
