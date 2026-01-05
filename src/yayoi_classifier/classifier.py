"""Simple rule-based classifier for Yayoi accounting software subjects.

The implementation uses a combination of keyword rules and optional
predefined mappings. It is intentionally small and dependency free so it can
be embedded into scripts or CLI tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ClassificationResult:
    """Represents a classifier decision.

    Attributes
    ----------
    subject_code:
        The Yayoi subject code that should be assigned.
    confidence:
        A float in ``[0.0, 1.0]`` indicating how confident the classifier is
        in the decision.
    matched_keywords:
        Keywords that triggered the classification. Helpful for debugging and
        logging.
    """

    subject_code: str
    confidence: float
    matched_keywords: List[str] = field(default_factory=list)


class AccountClassifier:
    """Classify text descriptions into Yayoi accounting subject codes.

    The classifier uses:

    * A required mapping of subject codes to keywords.
    * An optional lookup table for exact string matches (e.g., vendor names).

    The first matching exact lookup wins. Otherwise, scores are computed based
    on keyword frequency and length-normalized confidence.
    """

    def __init__(
        self,
        keyword_map: Dict[str, Iterable[str]],
        *,
        exact_lookup: Optional[Dict[str, str]] = None,
    ) -> None:
        if not keyword_map:
            raise ValueError("keyword_map must not be empty")
        self.keyword_map: Dict[str, List[str]] = {
            code: [kw.lower() for kw in keywords]
            for code, keywords in keyword_map.items()
        }
        self.exact_lookup: Dict[str, str] = {
            key.lower(): value for key, value in (exact_lookup or {}).items()
        }

    def classify(self, text: str) -> ClassificationResult:
        """Return the most likely subject code for ``text``.

        Parameters
        ----------
        text:
            A transaction description or memo.
        """

        normalized = text.lower().strip()

        if not normalized:
            raise ValueError("text must not be empty")

        # Exact lookup has top priority.
        if normalized in self.exact_lookup:
            subject = self.exact_lookup[normalized]
            return ClassificationResult(subject, 1.0, [normalized])

        token_matches: Dict[str, List[str]] = {code: [] for code in self.keyword_map}

        for code, keywords in self.keyword_map.items():
            for kw in keywords:
                if kw in normalized:
                    token_matches[code].append(kw)

        best_code = None
        best_score = 0.0
        best_keywords: List[str] = []

        for code, matches in token_matches.items():
            if not matches:
                continue
            # Confidence derived from number of matches and density in text.
            raw_score = len(matches)
            density = sum(len(m) for m in matches) / max(len(normalized), 1)
            score = 0.6 * raw_score + 0.4 * density

            if score > best_score:
                best_score = score
                best_code = code
                best_keywords = matches

        if best_code is None:
            # Fallback: unknown
            return ClassificationResult("UNCLASSIFIED", 0.0, [])

        confidence = min(best_score / (best_score + 1.0), 1.0)
        return ClassificationResult(best_code, confidence, best_keywords)


DEFAULT_KEYWORDS: Dict[str, List[str]] = {
    "411": ["売上", "販売", "入金", "売掛"],
    "521": ["仕入", "購入", "買掛", "発注"],
    "611": ["旅費", "交通", "電車", "タクシー", "出張"],
    "622": ["通信", "電話", "インターネット", "wifi"],
    "631": ["会議", "打合せ", "懇親会", "会食"],
    "701": ["給与", "給料", "賃金"],
}


def create_default_classifier() -> AccountClassifier:
    """Create a classifier with sensible defaults for quick start."""

    exact_lookup = {
        "jr東日本": "611",
        "ntt": "622",
    }
    return AccountClassifier(DEFAULT_KEYWORDS, exact_lookup=exact_lookup)
