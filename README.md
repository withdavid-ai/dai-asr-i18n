<p align="center">
  <a href="https://www.withdavid.ai/">
    <img alt="David AI" src="https://raw.githubusercontent.com/withdavid-ai/dai-asr-i18n/main/docs/assets/david-ai-wordmark.svg" width="208">
  </a>
</p>

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
uv run --locked dai-asr-i18n score \
  --language hi \
  --reference "मेरा कंप्यूटर खराब है" \
  --hypothesis "मेरा computer खराब है"
```

The hypothesis writes the loanword "computer" in Latin script and the reference writes it in Devanagari.
The scorer reads the hypothesis against the reference, so the pair scores a WER of 0.

Use the same normalization and scoring policies from Python:

```python
from dai_asr_i18n import normalize, normalize_pair, score_pair

assert normalize("(laughs) Call me at (02) 8012-3456, OK?", "en") == "call me at 02 8012 3456 ok"
assert normalize("學習語言", "zh") == "学习语言"

# Both texts go through the whole per-text pipeline (here: the event parenthetical, the native digit ३
# and the danda), then the pair rules read the hypothesis against the reference.
pair = normalize_pair("(हँसी) मेरा कंप्यूटर ३ दिन से ख़राब है।", "मेरा computer 3 दिन से ख़राब है", "hi")
print(pair.reference)  # मेरा कंप्यूटर 3 दिन से ख़राब है
print(pair.hypothesis)  # मेरा कंप्यूटर 3 दिन से ख़राब है

result = score_pair("你好世界", "你好世届", language="zh")
print(result.metric, result.value)  # cer 0.25
print(result.to_dict())
```

## Installation

DAI-ASR-I18N requires Python 3.11 or newer. If `uv` is not installed on macOS or Linux, install it
with the [official installer](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the repository root, synchronize the locked environment:

```bash
uv sync --locked
```

Run commands in the synchronized environment with `uv run`, for example:

```bash
uv run --locked dai-asr-i18n --help
```

`--locked` makes `uv` fail if `uv.lock` is out of date instead of silently updating it. It may still
synchronize the environment to the versions already recorded in the lockfile.

Alternatively, install into the current Python environment with pip:

```bash
python -m pip install .
```

Hugging Face dataset downloads, Parquet release tables, and FLAC validation are included in the
base install. Install optional dependencies only when needed for other workflows:

```bash
python -m pip install ".[audio]"          # audio decoding and preparation
python -m pip install ".[speaker]"        # speaker-attributed and diarization metrics
python -m pip install ".[audio,speaker]"
python -m pip install ".[audio,local]"    # local inference
python -m pip install ".[audio,hosted]"   # hosted APIs
```

## Features

- The benchmark normalizer for 21 languages, and stock Whisper normalization for comparison
- Pair normalization: a hypothesis takes the reference's spelling where a pair rule allows it, and never
  gains an error from it
- WER and CER scoring with substitution, deletion, insertion, and hit counts
- Speaker-attributed cpWER, cpCER, and cpSER with optimal speaker permutation
- Exact-interval DER, JER, and speaker-count metrics
- JSONL batch scoring with grouped micro-averages and machine-readable provenance
- Pinned Hugging Face dataset pulls and resumable local or hosted inference
- Optional deterministic Modal fan-out using the same execution path

## Supported languages

The DAI-ASR-I18N benchmark covers 21 languages. Pass the listed code to `--language`; regional language tags are canonicalized to their primary language.

| Language | Code | Default metric |
|---|---:|---:|
| Arabic | `ar` | WER |
| Bengali | `bn` | WER |
| German | `de` | WER |
| English | `en` | WER |
| Spanish | `es` | WER |
| French | `fr` | WER |
| Hindi | `hi` | WER |
| Indonesian | `id` | WER |
| Italian | `it` | WER |
| Japanese | `ja` | CER |
| Korean | `ko` | CER |
| Marathi | `mr` | WER |
| Portuguese | `pt` | WER |
| Russian | `ru` | WER |
| Tamil | `ta` | WER |
| Telugu | `te` | WER |
| Thai | `th` | CER |
| Tagalog | `tl` | WER |
| Turkish | `tr` | WER |
| Vietnamese | `vi` | WER |
| Mandarin Chinese | `zh` | CER |

## Normalization

The `dai_asr_i18n` profile normalizes each text in fixed stages. A language's base transform does the casing,
Unicode normalization and script-specific cleanup (Turkish dotted and dotless i, Arabic harakat and letter
variants, conservative Indic cleanup, traditional-to-simplified Chinese, Whisper's English spelling and
number rules). The steps below then each change one thing.

| Step | Languages | Rule | Example |
|---|---|---|---|
| Event parentheses | all | A parenthetical is removed with its contents only when it names a sound or event from the event list. Any other parenthetical keeps its contents and loses only the parentheses. | "(laughs) call (02) 8012" → "call 02 8012" |
| Apostrophes | all | Every apostrophe look-alike becomes the straight apostrophe, so the punctuation step sees one character. | ’ ‘ ʼ ´ \` ′ → ' |
| Spaced clitics | en, fr, it | A clitic written with a space before its apostrophe is attached to its word: an English contraction ending, or a French or Italian elision particle. | "I 'm" → "i'm", "c 'est" → "c'est" |
| Russian ё | ru | ё is written as е, composed or decomposed, in either case, since transcripts write the letter both ways. | "ещё" → "еще" |
| Thai mark order | th | Vowel and tone marks are put in one order, and a space before a combining mark is removed. No mark and no repeated vowel is deleted. | "เเปลก" → "แปลก" |
| Japanese characters | ja | NFKC is applied, and hyphen and tilde look-alikes and decorative symbols (〜 ～ ‐ ♪ ☆ ★ ○ ● ◎ ♥) are deleted. 〒 is kept. | "コーヒー〜♪" → "コーヒー" |
| Japanese long vowels | ja | A run of long-vowel marks becomes one ー. It runs after punctuation removal, which can bring two ー together. | "コーーヒー" → "コーヒー" |
| Native digits | all | Every Unicode decimal digit becomes its ASCII digit, so a number is the same in any script. | "৩", "٣", "३", "๓" → "3" |
| Punctuation | all | Punctuation and symbols become spaces. An apostrophe stays between two letters, except in Korean, Japanese, Chinese and Thai, where it is a quote mark unless inside a Latin word, and in the five Indic languages, where it marks a suffix boundary. Every hyphen becomes a space. | "l'homme" stays; "benar-benar" → "benar benar"; "만족감'이라고도" → "만족감이라고도" |
| Leading zeros | en | A digit group that starts with 0 keeps its zeros instead of being read as a number. A lone 0, a decimal and a group joined to a letter are not touched. | "0556" stays "0556" |

English keeps contractions and fillers as written ("don't", "gonna", "um"), and reads numbers as
Whisper does ("twenty-one dollars" → "$21"), except that "oh" is never a zero and "pounds" is not
currency.

Two rules need both transcripts. Scoring applies them to the hypothesis only, after both texts are
normalized:

- **Transliteration** (hi, mr, bn, ta, te, ar, ko, ru, ja, th): a loanword the model writes in one script
  takes the reference's spelling of it in the other ("computer" → "कंप्यूटर", "জিবিটি" → "gpt"). The
  classes come from the transliteration maps in
  [`src/dai_asr_i18n/normalization/resources/transliteration/`](src/dai_asr_i18n/normalization/resources/transliteration/).
  Two spellings in the same script are never matched: that is a spelling difference, not a script choice.
- **Apostrophes against the reference** (de, tl, tr): a stretch that differs from the reference only by
  apostrophes ("aus'm" / "ausm", "türkçe'yi" / "türkçeyi") takes the reference's spelling.

The reference is never changed, so the denominator is the per-text output. A change that would raise the
word or character edit distance is not made. `dai-asr-i18n normalize --reference ... --json` shows the changes
from the command line.

`--profile whisper_baseline` scores with stock Whisper normalization instead: `EnglishTextNormalizer` for
English and `BasicTextNormalizer` for every other language, with no pair rules.

## Batch scoring

Provide newline-delimited JSON containing `reference`, `hypothesis`, and `language`. Additional
fields are retained as metadata and can be used for grouping.

```json
{"id":"clip-1","model":"model-a","language":"en","reference":"hello world","hypothesis":"hello"}
{"id":"clip-2","model":"model-a","language":"zh","reference":"學習語言","hypothesis":"学习语言"}
```

```bash
uv run --locked dai-asr-i18n score --input pairs.jsonl --output scores.json
uv run --locked dai-asr-i18n score --input pairs.jsonl --group-by model --output scores-by-model.json
```

Results include normalized text, edit counts, micro-averaged aggregates, policy identifiers, and
runtime provenance. Use `--profile whisper_baseline` to compare against stock Whisper
normalization, or `--metric wer|cer|ser` to override automatic metric selection where supported.

## Dataset inference

Authenticate with Hugging Face before downloading the gated dataset. Your account must already
have access to the [dataset](https://huggingface.co/datasets/davidai-labs/DAI-ASR-I18N):

```bash
uv run --locked hf auth login
```

Dataset pulls default to `davidai-labs/DAI-ASR-I18N` at revision `latest`. The `latest` alias
resolves the current `main` branch to its full commit SHA when the pull starts.

The resolved SHA is recorded in the dataset receipt and used for every file download in that pull.

Run these commands to pull the latest dataset version and run the `whisper-large-v3-portable`
profile locally:

```bash
uv run --locked dai-asr-i18n dataset pull \
  --authenticated \
  --language en \
  --output .artifacts/datasets/public-en

uv run --locked --extra local --extra speaker dai-asr-i18n run \
  --dataset .artifacts/datasets/public-en \
  --config whisper-large-v3-portable \
  --output .artifacts/runs/whisper-en
```

`--authenticated` lets the dataset client use the token saved by `hf auth login`; the gated
dataset cannot be downloaded anonymously. For inference, the `local` extra provides
`faster-whisper`, while `speaker` provides speaker-aware scoring dependencies. On the first run,
`uv` may install these packages before starting the command.

The Whisper model is public, so an unauthenticated Hugging Face warning during model download is
informational. On a CPU without efficient float16 support, CTranslate2 automatically converts the
model to float32; this is expected, but inference will be slower and use more memory.

During inference, an interactive progress bar reports processed channel tasks and the completed,
resumed, and failed counts. Progress is written to stderr so the final JSON report remains clean on
stdout. Add `--no-progress` to disable the bar.

If you want to run against a particular version of the dataset, open the dataset’s
[commit history on Hugging Face](https://huggingface.co/datasets/davidai-labs/DAI-ASR-I18N/commits/main),
select the version you want to evaluate, and export its **full 40-character commit SHA**. The
abbreviated hash shown in the history list is not sufficient:

```bash
export HF_DATASET_REVISION="<FULL_40_CHARACTER_COMMIT_SHA>"
```

Add `--revision "$HF_DATASET_REVISION"` to the `dataset pull` command to use that version.

The existing dataset format is supported: `word_alignments` contain speaker IDs and
aligned words with text, start and end times.

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
diarization-only models receive only diarization scores. The speaker-permuted scores normalize each
reference speaker and hypothesis speaker as a pair, so the pair rules apply as in `score_pair`; with
perfect diarization, a mono transcript split by speaker scores exactly as separate channels do.

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
uv run --locked --extra audio dai-asr-i18n mix-mono \
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
uv run --locked dai-asr-i18n manifest --output manifest.json
```

Manifests record the package and dependency versions, normalization/scoring policy identifiers,
implementation hash, metric map, Unicode version, and optional resource hashes. Pin the package
release or Git commit and retain this manifest with scored artifacts.

## Previous research

You can follow our frontier AI research on our website:

- [DAI-S2S-ST: Evaluating S2S Model Quality with Human Preference Data](https://research.withdavid.ai/blog/st-s2s-eval)
  presents a single-turn speech-to-speech leaderboard built from 153,000 comparative ratings by 283
  people. It evaluates seven models on 819 recorded prompts and twelve dimensions spanning
  humanness and technical quality.

- [Can Large Audio Language Models reliably judge speech-to-speech models?](https://research.withdavid.ai/blog/lalm-as-judge-vs-hitl)
  compares three LALM judges with a calibrated human panel across content, acoustic, and interaction
  quality. The study explores where model judges track human preferences, where they diverge, and
  when human-in-the-loop evaluation remains necessary.

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
