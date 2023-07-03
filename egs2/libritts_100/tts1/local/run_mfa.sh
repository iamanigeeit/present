#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000
n_shift=256


./scripts/utils/mfa.sh \
    --split_sets "dev-clean test-clean train-clean-100" \
    --nj 12 \
    --acoustic_model english_us_arpa \
    --g2p_model english_us_arpa \
    --dictionary english_us_arpa \
    --train false \
    --cleaner tacotron \
    --samplerate ${fs} \
    --hop-size ${n_shift} \
    --clean_temp true \
    "$@"
