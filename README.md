# Stuttering-Aware ASR for Indonesian Language

[Paper (arXiv)](https://arxiv.org/abs/2601.03727)

## About

A synthetic Indonesian stuttering speech dataset and two fine-tuned Whisper-small models, released with the paper "Stuttering-Aware ASR for Indonesian Language". Every sentence from Indonesian Common Voice comes in a fluent and a stuttered version (repetition, prolongation, interjection). The stuttered text is produced by a rule-based algorithm ([rule_based.py](dataset/rule_based.py)) and an LLM ([prompts.txt](dataset/prompts.txt)), and the audio is generated with OpenAI gpt-4o-mini-tts. The models are fine-tuned from `cahya/whisper-small-id`, one on stuttered speech only and one on stuttered and fluent speech together.

## Download

Access to the dataset and models is currently restricted while the licensing terms are being finalized. Please request access through the Drive link and include your affiliation and intended use.

- Dataset: [indo-stutter-dataset.zip](https://drive.google.com/file/d/1SQWuUU-279_RdfJHvaKR2r8QmqyA81k5) (`clips/stutter`, `clips/nonstutter`, `metadata.csv`)
- Stutter-only model: [whisper-small-id-stutter-only.zip](https://drive.google.com/file/d/1jskGkg6RYH3x3i6xbW5zvl0FP19tlQYv)
- Stutter + clean model: [whisper-small-id-stutter-clean-mix.zip](https://drive.google.com/file/d/1DvJvI0DrXciigxeuCL2EmcnL_08kdKHw)

## Repository

- `dataset/prompts.txt`: the prompt used for LLM-based stutter text augmentation, together with the TTS instruction and settings.
- `dataset/rule_based.py`: the rule-based stutter augmentation. It has the same logic as the function used to generate the data, refactored for readability. It is reference code and is not meant to be run directly.

Training and evaluation follow standard Whisper fine-tuning and are not included.

## Dataset

`indo-stutter-dataset.zip` unzips to:

```
indo-stutter-dataset/
  clips/
    stutter/      <id>.wav    stuttered speech
    nonstutter/   <id>.wav    fluent speech of the same sentence
  metadata.csv
  README.txt
```

Each `id` appears in both clip folders and in `metadata.csv`. There are 12,133 sentences, split into 9,928 train, 1,102 dev and 1,103 test.

| Column | Description |
|---|---|
| `id` | utterance id, also the file name in `clips/stutter` and `clips/nonstutter` |
| `split` | `train`, `dev` or `test` |
| `text_disfluent` | stuttered text, spoken in `clips/stutter` |
| `text_original` | original fluent sentence, spoken in `clips/nonstutter` |
| `disfluent_type` | disfluency types used, separated by `\|` (`repetition`, `prolongation`, `interjection`) |
| `aug_method` | how the stuttered text was made: `Rule-based` ([rule_based.py](dataset/rule_based.py)) or `LLM` ([prompts.txt](dataset/prompts.txt)) |
| `voice` | OpenAI TTS voice used for this utterance (e.g. `sage`, `coral`, `onyx`) |
| `speed` | TTS speaking speed, between 0.75 and 1.25 (1.0 is normal) |

Audio was generated with OpenAI `gpt-4o-mini-tts`. For every sentence the voice is picked at random from the OpenAI voices and the speed is sampled uniformly from 0.75 to 1.25. The stuttered clips are generated with the instruction `Simulate stuttering voice`. The full prompts are in [prompts.txt](dataset/prompts.txt).

## Models

Each model zip is a standard Hugging Face Whisper-small folder:

```
whisper-small-id-stutter-only/          (or whisper-small-id-stutter-clean-mix/)
  config.json
  generation_config.json
  model.safetensors
  processor_config.json
  tokenizer.json
  tokenizer_config.json
```

- `whisper-small-id-stutter-only`: fine-tuned on stuttered speech only.
- `whisper-small-id-stutter-clean-mix`: fine-tuned on stuttered and fluent speech together.

## Usage

Unzip the files, then transcribe one test clip:

```python
import pandas as pd, librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

meta = pd.read_csv("indo-stutter-dataset/metadata.csv")
row = meta[meta.split == "test"].iloc[0]
audio, _ = librosa.load(f"indo-stutter-dataset/clips/stutter/{row.id}.wav", sr=16000)

path = "whisper-small-id-stutter-only"   # or "whisper-small-id-stutter-clean-mix"
processor = WhisperProcessor.from_pretrained(path)
model = WhisperForConditionalGeneration.from_pretrained(path)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
pred = model.generate(inputs.input_features)

print("model:    ", processor.batch_decode(pred, skip_special_tokens=True)[0])
print("stuttered:", row.text_disfluent)
print("fluent:   ", row.text_original)
```

## Citation

Coming soon.