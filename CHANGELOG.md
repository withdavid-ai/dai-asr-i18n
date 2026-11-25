# Changelog

## 2.0.0

Scores are not comparable with 1.x: the normalization and scoring policies changed
(`dai-asr-i18n-normalization-v10`, `dai-asr-i18n-offline-scoring-v4`).

- New benchmark normalizer (the `dai_asr_i18n` profile). Each language's transform runs first, then one
  step per rule: event parentheses, apostrophes, spaced clitics, Russian ё, Thai mark order, Japanese
  characters and long vowels, native digits, punctuation, and English leading zeros. Only event
  parentheticals lose their contents. English keeps contractions and fillers as written.
- Pair normalization (`normalize_pair`, `dai-asr-i18n normalize --reference`). The hypothesis takes the
  reference's spelling of a loanword written in the other script (hi, mr, bn, ta, te, ar, ko, ru, ja, th)
  or of a stretch that differs only by apostrophes (de, tl, tr). The reference never changes, and a
  rewrite that would add an error is not made. `score_pair`, cpWER/cpCER/cpSER and the evaluation runner
  score through it.
- `DaiAsrI18nNormalizer` is now the benchmark pipeline; the 1.x per-language cleanup class is
  `LanguageNormalizer`. Benchmark normalizer keys are prefixed `dai_asr_i18n:`.
- The manifest records the pipeline steps and the sha256 of the event list and transliteration maps.
- New dependency: `pythainlp==5.3.4`.

## 1.2.0 - Unreleased

- Updated Pyannote integration.
- Added security bug fixes.
- Updated the CLI.

## 1.1.0

- Updated DER numbers.

## 1.0.0

- Initial release.
