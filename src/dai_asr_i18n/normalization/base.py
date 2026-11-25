"""Core interface shared by every text normalizer."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextNormalizer(ABC):
    """Minimal public interface implemented by every normalization policy."""

    key: str

    @abstractmethod
    def apply(self, text: str) -> str:
        """Return the canonical form used for scoring."""

    def __call__(self, text: str) -> str:
        return self.apply(text)

    @property
    def resolved_key(self) -> str:
        """Concrete language transform recorded in score lineage."""

        return self.key


class NormalizedText(str):
    """String carrying the normalization profile that already normalized it."""

    profile: str

    def __new__(cls, text: str, profile: str) -> NormalizedText:
        instance = super().__new__(cls, text)
        instance.profile = profile
        return instance

    def __reduce__(self):
        return type(self), (str(self), self.profile)
