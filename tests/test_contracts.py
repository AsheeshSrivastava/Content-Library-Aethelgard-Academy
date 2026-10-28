"""Tests for the data contract (ADR 0003). Written before the code."""
import unicodedata

import pytest

from ragcore.contracts import (
    NORMALIZER_VERSION,
    SCHEMA_VERSION,
    ContractError,
    Passage,
    Source,
    Span,
    content_sha256,
    make_passage_id,
    make_passages,
    normalize_text,
    verify_passage,
)

RAW = "\ufeff# Title\r\n\r\nFirst paragraph.\r\n\r\nSecond paragraph.\r\n"


def make_source(text=None, **overrides):
    canonical = normalize_text(RAW if text is None else text)
    fields = dict(
        source_id="github:org/repo/docs/a.md",
        version="abc12345",
        text=canonical,
        raw_sha256=content_sha256("raw bytes of the file"),
        loader_version="markdown-1",
        licence="MIT",
        licence_basis="detected",
    )
    fields.update(overrides)
    return Source(**fields)


# --- normalization ---------------------------------------------------------


def test_normalize_removes_bom_and_crlf():
    assert normalize_text(RAW) == "# Title\n\nFirst paragraph.\n\nSecond paragraph.\n"


def test_normalize_lone_cr_becomes_newline():
    assert normalize_text("a\rb") == "a\nb"


def test_normalize_applies_nfc():
    decomposed = "é"  # e + combining acute
    assert normalize_text(decomposed) == unicodedata.normalize("NFC", decomposed) == "é"


def test_normalize_is_idempotent():
    once = normalize_text(RAW)
    assert normalize_text(once) == once


def test_normalize_keeps_other_whitespace():
    assert normalize_text("a  b\t c\n\n\nd") == "a  b\t c\n\n\nd"


# --- Source ----------------------------------------------------------------


def test_source_records_versions_and_text_hash():
    s = make_source()
    assert s.text_sha256 == content_sha256(s.text)
    assert s.normalizer_version == NORMALIZER_VERSION
    assert s.schema_version == SCHEMA_VERSION


def test_source_rejects_text_that_is_not_canonical():
    with pytest.raises(ContractError, match="normalized"):
        Source(
            source_id="x:y", version="v1", text=RAW, raw_sha256=content_sha256("r"),
            loader_version="l-1", licence="MIT", licence_basis="detected",
        )


def test_source_rejects_bad_raw_hash():
    with pytest.raises(ContractError, match="raw_sha256"):
        make_source(raw_sha256="nothex")


@pytest.mark.parametrize("field", ["source_id", "version", "loader_version", "licence"])
def test_source_rejects_empty_fields(field):
    with pytest.raises(ContractError, match=field):
        make_source(**{field: "  "})


def test_source_rejects_unknown_licence_basis():
    with pytest.raises(ContractError, match="licence_basis"):
        make_source(licence_basis="guessed")


def test_source_is_frozen():
    s = make_source()
    with pytest.raises(Exception):
        s.text = "changed"  # type: ignore[misc]


# --- IDs -------------------------------------------------------------------


def test_passage_id_is_deterministic():
    a = make_passage_id("s:1", "v1", 0, 5, content_sha256("hello"), "rules-1")
    b = make_passage_id("s:1", "v1", 0, 5, content_sha256("hello"), "rules-1")
    assert a == b


@pytest.mark.parametrize(
    "change",
    [
        dict(source_id="s:2"),
        dict(version="v2"),
        dict(start=1),
        dict(end=6),
        dict(text_hash=content_sha256("other")),
        dict(chunker_version="rules-2"),
    ],
)
def test_passage_id_changes_with_every_input(change):
    base = dict(source_id="s:1", version="v1", start=0, end=5,
                text_hash=content_sha256("hello"), chunker_version="rules-1")
    assert make_passage_id(**base) != make_passage_id(**{**base, **change})


# --- Span and Passage ------------------------------------------------------


def test_span_rejects_bad_offsets():
    with pytest.raises(ContractError):
        Span(start=5, end=5)
    with pytest.raises(ContractError):
        Span(start=-1, end=3)
    with pytest.raises(ContractError):
        Span(start=4, end=2)


def test_make_passages_slices_the_source_text_exactly():
    s = make_source()
    spans = [Span(0, 7, ("Title",)), Span(9, 25)]
    passages = make_passages(s, spans, chunker_version="rules-1")
    assert [p.text for p in passages] == [s.text[0:7], s.text[9:25]]
    assert passages[0].headings == ("Title",)
    assert all(verify_passage(s, p) for p in passages)


def test_passage_ids_follow_chunker_version_but_content_id_does_not():
    s = make_source()
    a = make_passages(s, [Span(0, 7)], chunker_version="rules-1")[0]
    b = make_passages(s, [Span(0, 7)], chunker_version="rules-2")[0]
    assert a.passage_id != b.passage_id
    assert a.content_id == b.content_id == content_sha256(s.text[0:7])


def test_same_text_in_two_sources_shares_content_id():
    s1 = make_source(source_id="github:a/b/one.md", text="same words here\n")
    s2 = make_source(source_id="github:a/b/two.md", text="same words here\n")
    p1 = make_passages(s1, [Span(0, 15)], chunker_version="rules-1")[0]
    p2 = make_passages(s2, [Span(0, 15)], chunker_version="rules-1")[0]
    assert p1.content_id == p2.content_id
    assert p1.passage_id != p2.passage_id


def test_make_passages_rejects_span_past_the_end():
    s = make_source()
    with pytest.raises(ContractError, match="outside"):
        make_passages(s, [Span(0, len(s.text) + 1)], chunker_version="rules-1")


def test_make_passages_rejects_blank_passage():
    s = make_source(text="abc\n\n\n\ndef\n")
    with pytest.raises(ContractError, match="blank"):
        make_passages(s, [Span(3, 7)], chunker_version="rules-1")


def test_make_passages_rejects_empty_chunker_version():
    with pytest.raises(ContractError, match="chunker_version"):
        make_passages(make_source(), [Span(0, 7)], chunker_version="")


def test_passage_with_wrong_id_is_rejected():
    s = make_source()
    good = make_passages(s, [Span(0, 7)], chunker_version="rules-1")[0]
    with pytest.raises(ContractError, match="passage_id"):
        Passage(
            source_id=good.source_id, version=good.version, start=good.start, end=good.end,
            text=good.text, headings=good.headings, chunker_version=good.chunker_version,
            passage_id="forged",
        )


def test_verify_passage_detects_text_that_is_not_the_slice():
    s = make_source()
    p = make_passages(s, [Span(0, 7)], chunker_version="rules-1")[0]
    other = make_source(text="# Other\n\nDifferent text.\n")
    assert verify_passage(s, p) is True
    assert verify_passage(other, p) is False


def test_verify_passage_detects_other_source_or_version():
    s = make_source()
    p = make_passages(s, [Span(0, 7)], chunker_version="rules-1")[0]
    assert verify_passage(make_source(version="zzz99999"), p) is False
    assert verify_passage(make_source(source_id="github:org/repo/other.md"), p) is False


def test_passage_ids_are_unique_within_a_source():
    s = make_source()
    passages = make_passages(s, [Span(0, 7), Span(9, 25), Span(27, 44)], chunker_version="rules-1")
    assert len({p.passage_id for p in passages}) == 3
