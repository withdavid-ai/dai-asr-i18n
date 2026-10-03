# DAI-ASR-I18N

DAI-ASR-I18N is a multilingual evaluation toolkit for automatic speech recognition. It provides
versioned text normalization, offline error-rate scoring, and speaker-attributed metrics without
requiring model-provider credentials.

Read the [DAI-ASR-I18N blog post](https://research.withdavid.ai/blog/dai-asr-i18n) and explore the
[DAI-ASR-I18N dataset on Hugging Face](https://huggingface.co/datasets/davidai-labs/DAI-ASR-I18N).

## Quick start

Score a reference and hypothesis from the command line. The language policy selects the default
metric:

```bash
dai-asr-i18n score \
  --language zh \
  --reference "學習語言" \
  --hypothesis "学习语言"
```

Use the same normalization and scoring policies from Python:

```python
from dai_asr_i18n import normalize, score_pair

assert normalize("學習語言", "zh") == "学习语言"

result = score_pair("你好世界", "你好世届", language="zh")
print(result.metric, result.value)
print(result.to_dict())
```

## Installation

DAI-ASR-I18N requires Python 3.11 or newer. Install it from a checkout:

```bash
python -m pip install .
```

Install optional dependencies only when needed:

```bash
python -m pip install ".[audio]"       # audio decoding and preparation
python -m pip install ".[speaker]"     # speaker-attributed and diarization metrics
python -m pip install ".[audio,speaker]"
python -m pip install ".[hf,audio,local]"   # Hugging Face + local inference
python -m pip install ".[hf,audio,hosted]"  # Hugging Face + hosted APIs
```

## Features

- Benchmark and stock-Whisper normalization profiles for 21 languages
- WER and CER scoring with substitution, deletion, insertion, and hit counts
- Speaker-attributed cpWER, cpCER, and cpSER with optimal speaker permutation
- Exact-interval DER, JER, and speaker-count metrics
- JSONL batch scoring with grouped micro-averages and machine-readable provenance
- Pinned Hugging Face dataset pulls and resumable local or hosted inference
- Optional deterministic Modal fan-out using the same execution path

## Batch scoring

Provide newline-delimited JSON containing `reference`, `hypothesis`, and `language`. Additional
fields are retained as metadata and can be used for grouping.

```json
{"id":"clip-1","model":"model-a","language":"en","reference":"hello world","hypothesis":"hello"}
{"id":"clip-2","model":"model-a","language":"zh","reference":"學習語言","hypothesis":"学习语言"}
```

```bash
dai-asr-i18n score --input pairs.jsonl --output scores.json
dai-asr-i18n score --input pairs.jsonl --group-by model --output scores-by-model.json
```

Results include normalized text, edit counts, micro-averaged aggregates, policy identifiers, and
runtime provenance. Use `--profile whisper_baseline` to compare against stock Whisper
normalization, or `--metric wer|cer|ser` to override automatic metric selection where supported.

## Dataset inference

First, users should be able to login and authenticate via Hugging Face:

```bash
hf auth login
```

First, open the dataset’s [commit history on Hugging Face](https://huggingface.co/datasets/davidai-labs/DAI-ASR-I18N/commits/main),
select the version you want to evaluate, and copy its **full 40-character commit SHA** from the
commit page. The abbreviated hash shown in the history list is not sufficient.

Replace the placeholder below with that full SHA and export it in your shell:

```bash
export HF_DATASET_COMMIT="<FULL_40_CHARACTER_COMMIT_SHA>"
```

The existing dataset format is supported: `word_alignments` contain speaker IDs and
aligned words with text, start and end times. A separate segment-level `text` field is
not required. Local manifests remain schema version 2. Scoring uses the supplied
reference timestamps as published; it does not rerun alignment or change the dataset.

Keep this value fixed for the run. Then pull a language selection and run a built-in profile:

```bash
dai-asr-i18n dataset pull \
  --repo davidai-labs/DAI-ASR-I18N \
  --revision "$HF_DATASET_COMMIT" \
  --authenticated \
  --language en \
  --output .artifacts/datasets/public-en

dai-asr-i18n run \
  --dataset .artifacts/datasets/public-en \
  --config whisper-large-v3-portable \
  --output .artifacts/runs/whisper-en
```

We provide starter code for running inference on Modal in [`deploy/`](deploy/).
You can also use your own GPUs or infrastructure, as long as your model outputs plug into
the same scoring framework. You’ll need your own API keys for hosted providers and a Modal
account if you use the Modal runners. The [run profiles](src/dai_asr_i18n/run_configs/)
list each model’s required credential environment variables.

OpenAI file uploads use 16 kHz mono MP3 with encoder-delay metadata, retaining the
source clock. Request records include source/upload checksums and decoded durations.
GPT-Transcribe receives `languages[]`, using `zh-cn` only for an explicit `zh-CN`
dataset locale; other language hints use the primary language code. The diarizing
OpenAI model uses its separate `language` parameter. Changed request protocols require
a new run directory rather than resuming an incompatible run.

Scoring runs automatically on the connector outputs, using the dataset's supplied aligned
references. No published hypotheses or precomputed predictions are required.

The runner writes `scores.jsonl` beside its own inference outputs. Physical channels are
scored independently; dual-channel scores sum their edit counts and require both channels.
Modal shard consolidation combines channels even when they ran on different workers.
Mixed-audio hybrid models receive speaker-permuted transcript scores and diarization scores;
diarization-only models receive only diarization scores.

| Metric | Reference / convention |
|---|---|
| WER/CER (Vietnamese SER) | Normalized aligned-reference text; language-specific scoring units |
| cpWER/cpCER/cpSER | Minimum-permutation speaker-attributed mixed-audio text |
| DER, miss, false alarm, confusion | Reference aligned words merged across gaps ≤0.20 s; overlap included |
| JER | Same aligned speech regions; no collar |
| Speaker-count accuracy / MAE | Speaker identities in the aligned references |

Default DER has no collar; `_c25` adds a ±0.25-second boundary collar. `_forced` identifies
aligned-word speech regions. Rates are fractions; DER numerators/denominators use
speaker-milliseconds. Primary units are CER for Mandarin/Japanese/Korean/Thai and WER
elsewhere, with Vietnamese whitespace-delimited syllables reported as SER.

For Scribe, MAI, Gemini and xAI, hypothesis activity comes from their native word
timestamps: merge words from the same speaker only when the gap is ≤0.20 seconds.
Zero-duration words add no activity and do not bridge gaps. Other models retain their
native speaker segments. This affects DER, JER and speaker-count inputs only; transcript
text and lexical scoring stay unchanged. Gemini words with reversed timestamps count as
missing activity, with their text retained for transcript scoring; other invalid timing
makes the diarization score unavailable.
MAI's word annotations must cover each speaker's complete text in order. This validation
allows NFC Unicode equivalence, casefolding, whitespace and punctuation differences;
it preserves marks, symbols and digits and does not use the scoring normalizer.
If MAI's phrase text and native words disagree, its speaker-labelled phrase text is
still scored, but timing-dependent scores are unavailable. Empty phrase text uses only
words actually supplied by MAI. Empty, wordless phrases add no speech activity. Native
responses and word timestamps are retained unchanged.

Failed requests remain in `failures.jsonl`; they are never substituted with empty predictions.
Successful empty outputs are scored as deletions/missed speech. Scoring problems appear in
`unavailable` and do not trigger another paid inference request. Resume uses this run's own
checkpointed outputs. Install the `speaker` extra for speaker-permutation and diarization scoring.


## Deterministic mono reconstruction

Reconstruct the canonical mixed-speaker input from two lossless channels. Pass an expected checksum when one is
available:

```bash
dai-asr-i18n mix-mono \
  --channel-1 speaker_1.flac \
  --channel-2 speaker_2.flac \
  --output mono.wav
```

`mono-mix-v1` independently resamples each channel to 16 kHz, zero-pads the shorter channel, sums the samples,
conditionally peak-normalizes, and encodes mono PCM16 WAV. The command emits the derived checksum and environment
manifest. The same operation is available as `build_mono_mix` in Python.

## Run manifests

Generate a standalone manifest alongside benchmark outputs:

```bash
dai-asr-i18n manifest --output manifest.json
```

Manifests record the package and dependency versions, normalization/scoring policy identifiers,
implementation hash, metric map, Unicode version, and optional resource hashes. Pin the package
release or Git commit and retain this manifest with scored artifacts.

## Development

```bash
uv sync --locked --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python -m build
uv run twine check dist/*
```

## License

DAI-ASR-I18N is licensed under the [MIT License](LICENSE). Datasets and published benchmark
artifacts may use separate licenses documented with those assets.
