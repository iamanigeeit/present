This repository is built on top of [ESPNet](https://github.com/espnet/espnet).

[Paper](https://arxiv.org/abs/2408.06827) has been (re)submitted to Signal Processing Letters. Audio samples are [here](https://present2024.web.app/).

Various notebook examples:
- [Aligner](prosody/aligner.ipynb)
- [Text effects](prosody/text_effects.ipynb) (long phonemes, questions)
- [Prosody transfer](prosody/prosody.ipynb)

Zero shot language transfer:
- [English to German](prosody/german.ipynb)
  - [Evaluation on CSS10 with Whisper](prosody/whisper/german_cer.ipynb)


- [English to Hungarian](prosody/hungarian.ipynb)
  - [Evaluation on CSS10 with Whisper](prosody/whisper/hungarian_cer.ipynb)


- [English to Spanish](prosody/mandarin.ipynb)
  - [Evaluation on CSS10 with Whisper](prosody/whisper/spanish_cer.ipynb)


- [English to Mandarin](prosody/mandarin.ipynb)
  - For info on how to run the baseline, refer to my [clone of zm-text-tts](https://github.com/iamanigeeit/zm-text-tts)
  - [Evaluation on AISHELL-3 with FunASR Paraformer](prosody/funasr/mandarin_cer.ipynb) 

