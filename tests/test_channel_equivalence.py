"""End to end: with perfect diarization, a mono-channel transcript scores the same as separate channels.

A two-speaker conversation can reach the scorer two ways. With separate channels, each speaker's recording is
transcribed on its own, each channel is a reference/hypothesis pair, and the benchmark number is the
micro-average over the channels. With a mono channel, the model returns one stream of timed words with no
speaker labels; a diarization module returns timed speaker segments under its own labels, each word goes to
the segment that holds its midpoint, and the streams are scored by speaker with the best label assignment.

Given the same recognized words, perfect diarization must give exactly the same edit counts and rate on both
paths, pair rules (transliteration) included. Turns do not overlap: a word spoken over another speaker cannot
be attributed from speaker segments alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from dai_asr_i18n import aggregate_scores, der, normalize_pair, score_pair, score_speakers

WORD_S = 0.3  # length of one word (one character for Chinese)
GAP_S = 0.5  # silence between turns


@dataclass(frozen=True)
class Word:
    text: str
    start_s: float
    end_s: float
    speaker: str  # who said it: known to the test, never shown to the scorer on the mono path

    @property
    def midpoint_s(self) -> float:
        return (self.start_s + self.end_s) / 2


@dataclass(frozen=True)
class Conversation:
    reference: dict[str, str]  # one reference transcript per speaker
    activity: list[tuple[str, float, float]]  # who speaks when: (speaker, start, end), one per turn
    words: list[Word]  # the recognized words with their timestamps, in time order


def units(text: str, language: str) -> list[str]:
    """The timestamped units a model returns: characters for Chinese, whitespace words otherwise."""

    return [char for char in text if not char.isspace()] if language == "zh" else text.split()


def conversation(turns: list[tuple[str, str, str]], language: str) -> Conversation:
    """Lay out ``(speaker, reference turn, recognized turn)`` triples one after another in time."""

    reference: dict[str, list[str]] = {}
    activity, words, clock = [], [], 0.0
    for speaker, reference_turn, hypothesis_turn in turns:
        reference.setdefault(speaker, []).append(reference_turn)
        hypothesis_units = units(hypothesis_turn, language)
        end = clock + WORD_S * max(len(units(reference_turn, language)), len(hypothesis_units), 1)
        activity.append((speaker, clock, end))
        step = (end - clock) / max(len(hypothesis_units), 1)
        words.extend(
            Word(unit, clock + index * step, clock + (index + 1) * step, speaker)
            for index, unit in enumerate(hypothesis_units)
        )
        clock = end + GAP_S
    return Conversation({speaker: " ".join(texts) for speaker, texts in reference.items()}, activity, words)


def separate_channel_hypotheses(conv: Conversation) -> dict[str, str]:
    """What the model returns on each speaker's own channel: that speaker's words in time order."""

    channels: dict[str, list[str]] = {}
    for word in conv.words:
        channels.setdefault(word.speaker, []).append(word.text)
    return {speaker: " ".join(texts) for speaker, texts in channels.items()}


def perfect_diarization(conv: Conversation) -> list[tuple[str, float, float]]:
    """The true speaker turns, under the diarizer's own labels (spk0, spk1, ... in order of first speech)."""

    labels: dict[str, str] = {}
    return [(labels.setdefault(speaker, f"spk{len(labels)}"), start, end) for speaker, start, end in conv.activity]


def attribute(words: list[Word], segments: list[tuple[str, float, float]]) -> dict[str, str]:
    """Give each word of the mono transcript to the diarization segment that holds its midpoint."""

    streams: dict[str, list[str]] = {}
    for word in words:
        (label,) = [label for label, start, end in segments if start <= word.midpoint_s < end]
        streams.setdefault(label, []).append(word.text)
    return {label: " ".join(texts) for label, texts in streams.items()}


ENGLISH = [
    ("A", "Hello, how are you doing today?", "Hello, how are you doing today?"),
    ("B", "I'm fine, thanks. And you?", "I'm fine thanks, and you?"),
    ("A", "Pretty good. Did you see the game last night?", "Pretty good. Did you see the game last nite?"),
    ("B", "No, I missed it. Who won?", "No, I missed it. Who one?"),
    ("A", "The home team, by two points.", "The home team, by points."),
    ("B", "That's a close one.", "That's um a close one."),
]

CHINESE = [
    ("A", "你好，今天怎麼樣？", "你好，今天怎么样？"),
    ("B", "還不錯，你呢？", "还不错，你呢？"),
    ("A", "我在學習語言。", "我在学习语言。"),
    ("B", "哪一種語言？", "哪一种语严？"),
]

# Loanwords written in Latin script where the reference writes them in Devanagari, as models often do.
HINDI = [
    ("A", "हाय, कैसे हो? बहुत टाइम से तुम्हारा फ़ोन नहीं आया।", "हाय, कैसे हो? बहुत time से तुम्हारा phone नहीं आया।"),
    ("B", "हाँ यार, ऑफिस में बहुत काम था।", "हाँ यार, office में बहुत काम था।"),
    ("A", "नई जॉब कैसी चल रही है?", "नई job कैसी चल रही है?"),
    ("B", "अच्छी है, पर टीम थोड़ी बड़ी है।", "अच्छी है पर team थोड़ा बड़ी है"),
    ("A", "रोज़ कितने बजे स्टार्ट करते हो?", "रोज़ कितने बजे start करते हो?"),
    ("B", "सुबह ९ बजे, और शाम को जिम जाता हूँ।", "सुबह 9 बजे, और शाम को gym जाता हूँ।"),
    ("A", "वाह, तुम्हारा रूटीन तो बढ़िया है।", "वाह, तुम्हारा routine तो बढ़िया है।"),
    ("B", "डाइट भी फॉलो कर रहा हूँ।", "diet भी follow कर रहा हूँ।"),
    ("A", "मैं तो कॉलेज के बाद से ही स्ट्रेस में हूँ।", "मैं तो college के बाद से stress में हूँ।"),
    ("B", "क्यों, क्या हुआ?", "क्यों, क्या हुआ?"),
    ("A", "बजट कम है और फैमिली की चिंता भी है।", "budget कम है और family की चिंता भी है।"),
    ("B", "कोई ऑनलाइन काम ट्राई करो।", "कोई online काम try करो।"),
    ("A", "हाँ, मैं यूट्यूब पर वीडियो बनाने का प्लान बना रहा हूँ।", "हाँ, मैं youtube पर video बनाने का plan बना रहा हूँ।"),
    ("B", "अच्छा आइडिया है।", "अच्छा आइडिया है।"),
    ("A", "पर मोबाइल से वीडियो अच्छा नहीं आता।", "पर mobile से वीडियो अच्छा नहीं आता।"),
    ("B", "मेरे पास एक पुराना कैमरा है, ले लेना।", "मेरे पास पुराना कैमरा है ले लेना"),
    ("A", "शुक्रिया यार! कल कॉफी पर मिलते हैं?", "शुक्रिया यार! कल coffee पर मिलते हैं?"),
    ("B", "कल नहीं, कल पार्टी है।", "कल नहीं, कल party है।"),
    ("A", "तो परसों शाम को कॉल करना।", "तो परसों शाम को call करना।"),
    ("B", "ठीक है, पाँच मिनट पहले मैसेज कर दूँगा।", "ठीक है, पाँच minute पहले message कर दूँगा।"),
    ("A", "ठीक है, फिर मिलते हैं।", "ठीक है फिर मिलते हैं"),
    ("B", "बिल्कुल, बाय!", "बिलकुल, बाय!"),
]

FRENCH = [
    ("A", "Salut ! C'est toi qui as appelé ce matin ?", "Salut, c 'est toi qui a appelé ce matin ?"),
    ("B", "Oui, j’ai essayé de te joindre vers 9 heures.", "Oui, j'ai essayé de te joindre vers 9 heures."),
    ("A", "Désolé, j'étais au rendez-vous chez le médecin.", "Désolé, j 'étais au rendez vous chez le médecin."),
    ("B", "Rien de grave, j'espère ?", "Rien de grave j’espère ?"),
    ("A", "Non, peut-être une petite grippe. Et toi, ça va ?", "Non, peut être une petite grippe. Et toi ça va ?"),
    ("B", "Ça va. Je voulais savoir si tu viens samedi.", "Ça va. Je voulais savoir si tu vient samedi."),
    ("A", "Bien sûr ! Qu'est-ce que j'apporte ?", "Bien sûr, qu'est ce que j'apporte ?"),
    (
        "B",
        "Un dessert, si tu peux. L'anniversaire commence à 20 heures.",
        "Un dessert si tu peux. L'anniversaire commence à 20 heures.",
    ),
]

CASES = [
    pytest.param(ENGLISH, "en", id="en-wer"),
    pytest.param(CHINESE, "zh", id="zh-cer"),
    pytest.param(HINDI, "hi", id="hi-wer-transliteration-22-turns"),
    pytest.param(FRENCH, "fr", id="fr-wer"),
]


@pytest.mark.parametrize(("turns", "language"), CASES)
def test_perfect_diarization_scores_the_same_as_separate_channels(turns, language):
    conv = conversation(turns, language)

    # Separate channels: one pair per channel, micro-averaged as batch scoring does.
    hypothesis_channels = separate_channel_hypotheses(conv)
    (separate,) = aggregate_scores(
        score_pair(conv.reference[speaker], hypothesis_channels[speaker], language=language)
        for speaker in conv.reference
    )

    # Mono channel: unlabeled timed words, plus a diarization that DER confirms is perfect.
    diarization = perfect_diarization(conv)
    diarization_score = der(conv.activity, diarization)
    assert diarization_score.value == 0
    mono = score_speakers(conv.reference, attribute(conv.words, diarization), language=language)

    assert separate.counts.numerator > 0, "the scenario must contain errors"
    assert mono.counts == separate.counts
    assert mono.value == separate.value
    assert mono.metric == f"cp{separate.metric}"
    assert set(mono.speaker_assignment) == set(diarization_score.speaker_assignment)


def test_transliteration_applies_on_both_paths():
    """The Hindi case depends on transliteration: without it, every Latin loanword would be an error."""

    conv = conversation(HINDI, "hi")
    hypothesis_channels = separate_channel_hypotheses(conv)
    mono = score_speakers(conv.reference, attribute(conv.words, perfect_diarization(conv)), language="hi")
    latin = re.compile("[a-z]")
    for speaker, label in mono.speaker_assignment:
        assert latin.search(hypothesis_channels[speaker])
        assert normalize_pair(conv.reference[speaker], hypothesis_channels[speaker], "hi").rewrites
        assert not latin.search(mono.hypothesis_normalized[label])
    assert mono.counts.numerator == 4  # थोड़ा, the dropped ही and एक, बिलकुल


def test_a_misattributed_turn_is_charged():
    """The equality depends on the diarization: one turn given to the wrong speaker costs errors."""

    conv = conversation(ENGLISH, "en")
    diarization = perfect_diarization(conv)
    perfect = score_speakers(conv.reference, attribute(conv.words, diarization), language="en")

    wrong = list(diarization)
    wrong[1] = (wrong[0][0], *wrong[1][1:])  # B's "I'm fine thanks, and you?" labeled as A
    assert der(conv.activity, wrong).value > 0
    imperfect = score_speakers(conv.reference, attribute(conv.words, wrong), language="en")

    assert imperfect.counts.numerator > perfect.counts.numerator
    assert imperfect.counts.denominator == perfect.counts.denominator
