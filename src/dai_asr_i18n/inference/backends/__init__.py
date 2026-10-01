"""Built-in local and hosted inference backends."""

from dai_asr_i18n.inference.backends.deepgram import DeepgramBackend
from dai_asr_i18n.inference.backends.elevenlabs import ElevenLabsBackend
from dai_asr_i18n.inference.backends.faster_whisper import FasterWhisperBackend
from dai_asr_i18n.inference.backends.gemini import GeminiTranscribeBackend
from dai_asr_i18n.inference.backends.mai import MaiTranscribeBackend
from dai_asr_i18n.inference.backends.meta import MetaMuseBackend
from dai_asr_i18n.inference.backends.mistral import MistralVoxtralBackend
from dai_asr_i18n.inference.backends.openai import OpenAITranscriptionBackend
from dai_asr_i18n.inference.backends.qwen import Qwen3AsrBackend
from dai_asr_i18n.inference.backends.xai import XaiGrokBackend

__all__ = [
    "DeepgramBackend",
    "ElevenLabsBackend",
    "FasterWhisperBackend",
    "GeminiTranscribeBackend",
    "MaiTranscribeBackend",
    "MetaMuseBackend",
    "MistralVoxtralBackend",
    "OpenAITranscriptionBackend",
    "Qwen3AsrBackend",
    "XaiGrokBackend",
]
