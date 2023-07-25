#!/usr/bin/env bash


download_dir=$1

# check arguments
if [ $# != 1 ]; then
    echo "Usage: $0 <download_dir>"
    exit 1
fi

set -euo pipefail

cwd=$(pwd)
if [ ! -e "${download_dir}/ESD" ]; then
    mkdir -p "${download_dir}"
    cd "${download_dir}"
    # Code from https://stackoverflow.com/questions/48133080/how-to-download-a-google-drive-url-via-curl-or-wget
    # Download flow may change again, report issue if it fails
    fileid="1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v"
    filename="ESD.zip"
    html=$(curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}")
    confirm=$(echo ${html} | grep -Po '(confirm=[a-zA-Z0-9\-_]+)')
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&${confirm}&id=${fileid}" -o ${filename}
    unzip -q ESD.zip -d ESD
    rm ESD.zip
    for f in ESD/Emotional\ Speech\ Dataset\ \(ESD\)/*; do
        mv "$f" "ESD/";
    done
    rm -r ESD/Emotional\ Speech\ Dataset\ \(ESD\)
    cd "${cwd}"
    echo "Successfully prepared ESD."
else
    echo "${download_dir}/ESD already exists. Skipped downloading."
fi
