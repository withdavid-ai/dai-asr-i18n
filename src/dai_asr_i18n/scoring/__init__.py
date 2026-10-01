"""Public offline scoring API."""

from dai_asr_i18n.scoring.aggregate import aggregate_record_scores, aggregate_scores, score_records
from dai_asr_i18n.scoring.alignment import align_tokens, tokens_for_metric
from dai_asr_i18n.scoring.core import SCORING_POLICY_VERSION, score_pair
from dai_asr_i18n.scoring.diarization import (
    DiarizationScore,
    DiarizationSegment,
    JaccardErrorScore,
    SpeakerCountScore,
    der,
    diarization_error_rate,
    jaccard_error_rate,
    jer,
    speaker_count,
)
from dai_asr_i18n.scoring.models import AggregateScore, EditCounts, Metric, RecordAggregate, Score
from dai_asr_i18n.scoring.permutation import PermutationScore, cpcer, cpser, cpwer, score_speakers

__all__ = [
    "AggregateScore",
    "DiarizationScore",
    "DiarizationSegment",
    "EditCounts",
    "JaccardErrorScore",
    "Metric",
    "PermutationScore",
    "RecordAggregate",
    "Score",
    "SCORING_POLICY_VERSION",
    "SpeakerCountScore",
    "aggregate_record_scores",
    "aggregate_scores",
    "align_tokens",
    "cpcer",
    "cpser",
    "cpwer",
    "der",
    "diarization_error_rate",
    "jaccard_error_rate",
    "jer",
    "score_pair",
    "score_records",
    "score_speakers",
    "speaker_count",
    "tokens_for_metric",
]
