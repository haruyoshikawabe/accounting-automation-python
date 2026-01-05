import pytest

from yayoi_classifier.classifier import (
    AccountClassifier,
    ClassificationResult,
    create_default_classifier,
)


def test_raises_on_empty_keyword_map():
    with pytest.raises(ValueError):
        AccountClassifier({})


def test_raises_on_empty_text():
    classifier = create_default_classifier()
    with pytest.raises(ValueError):
        classifier.classify("")


def test_exact_match_takes_precedence():
    classifier = create_default_classifier()
    result = classifier.classify("JR東日本")
    assert result.subject_code == "611"
    assert result.confidence == 1.0
    assert result.matched_keywords == ["jr東日本"]


def test_keyword_match_returns_best_code():
    classifier = create_default_classifier()
    result = classifier.classify("オンライン会議用にインターネット料金を支払い")
    assert result.subject_code == "622"
    assert result.matched_keywords == ["インターネット"]
    assert 0.0 < result.confidence <= 1.0


def test_returns_unclassified_when_no_match():
    classifier = create_default_classifier()
    result = classifier.classify("未知の取引")
    assert result.subject_code == "UNCLASSIFIED"
    assert result.confidence == 0.0
    assert result.matched_keywords == []
