"""Spec 04: licence detection and cross-check. Written before the code."""
from dataclasses import replace
from datetime import date

import pytest

from ragcore.fetch import FetchResult, apply_fetch
from ragcore.gate import LicenceRules
from ragcore.licence import Outcome, cross_check, detect_licence, read_licence_file, recheck_licence
from ragcore.register import (
    FetchSpec, Licence, SourceRecord, TransitionError, load_record, save_record, set_status,
)

MIT = """MIT License

Copyright (c) 2024 Someone

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""
ISC = """ISC License

Copyright (c) 2024 Someone

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES.
"""
BSD2 = """BSD 2-Clause License

Copyright (c) 2024, Someone. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS".
"""
BSD3 = BSD2.replace("BSD 2-Clause", "BSD 3-Clause").replace(
    "THIS SOFTWARE IS PROVIDED",
    "3. Neither the name of the copyright holder nor the names of its\n   contributors may be used to endorse or promote products derived from\n"
    "   this software without specific prior written permission.\n\nTHIS SOFTWARE IS PROVIDED")
BSD_ADVERT = BSD2.replace("THIS SOFTWARE", "3. All advertising materials mentioning features of this software must display "
                          "the following acknowledgement.\n\nTHIS SOFTWARE")
APACHE = "                                 Apache License\n                           Version 2.0, January 2004\n                        http://www.apache.org/licenses/\n"
MPL2 = "Mozilla Public License Version 2.0\n==================================\n\n1. Definitions\n"
GPL3 = ("                    GNU GENERAL PUBLIC LICENSE\n                       Version 3, 29 June 2007\n\n"
        "Preamble\n\nThe GNU General Public License is a free, copyleft license. See also the GNU Lesser General "
        "Public License version 3 and the GNU Affero General Public License version 3.\n")
GPL2 = "                    GNU GENERAL PUBLIC LICENSE\n                       Version 2, June 1991\n\nPreamble\n"
GPL1 = "                    GNU GENERAL PUBLIC LICENSE\n                       Version 1, February 1989\n"
LGPL3 = "                   GNU LESSER GENERAL PUBLIC LICENSE\n                       Version 3, 29 June 2007\n"
LGPL21 = "                  GNU LESSER GENERAL PUBLIC LICENSE\n                       Version 2.1, February 1999\n"
AGPL3 = "                    GNU AFFERO GENERAL PUBLIC LICENSE\n                       Version 3, 19 November 2007\n"
MPL11 = "Mozilla Public License Version 1.1\n---------------\n1. Definitions.\n"
CC0 = "CC0 1.0 Universal\n\nStatement of Purpose\n\nThe laws of most jurisdictions throughout the world...\n"
UNLICENSE = "This is free and unencumbered software released into the public domain.\n\nAnyone is free to copy, modify.\n"
CC_PREAMBLE = ("Attribution{variant} 4.0 International\n\n=======================================================================\n\n"
               "Creative Commons Corporation (\"Creative Commons\") is not a law firm and does not provide legal services.\n\n"
               "Using Creative Commons Public Licenses\n\nCreative Commons public licenses provide a standard set of terms.\n\n"
               "=======================================================================\n\n"
               "Creative Commons {name} 4.0 International Public License\n\n"
               "By exercising the Licensed Rights (defined below), You accept and agree to be bound by the terms and "
               "conditions of this Creative Commons {name} 4.0 International Public License (\"Public License\").\n")
CC_BY = CC_PREAMBLE.format(variant="", name="Attribution")
CC_BY_SA = CC_PREAMBLE.format(variant="-ShareAlike", name="Attribution-ShareAlike")
CC_BY_NC = CC_PREAMBLE.format(variant="-NonCommercial", name="Attribution-NonCommercial")

KNOWN = [
    (MIT, "MIT"), (ISC, "ISC"), (BSD2, "BSD-2-Clause"), (BSD3, "BSD-3-Clause"), (APACHE, "Apache-2.0"),
    (MPL2, "MPL-2.0"), (GPL2, "GPL-2.0"), (GPL3, "GPL-3.0"), (LGPL3, "LGPL-3.0"), (AGPL3, "AGPL-3.0"),
    (CC0, "CC0-1.0"), (UNLICENSE, "Unlicense"), (CC_BY, "CC-BY-4.0"), (CC_BY_SA, "CC-BY-SA-4.0"),
]


@pytest.mark.parametrize("text,expected", KNOWN)
def test_detects_each_licence(text, expected):
    assert detect_licence(text) == expected


def test_years_and_copyright_lines_do_not_matter():
    assert detect_licence(MIT.replace("2024 Someone", "1999-2031 Other Holders Ltd.")) == "MIT"
    assert detect_licence(MIT.replace("Copyright (c) 2024 Someone\n", "")) == "MIT"


def test_case_and_spacing_do_not_matter():
    assert detect_licence(MIT.upper()) == "MIT"
    assert detect_licence(MIT.replace(" ", "  ").replace("\n", "\r\n")) == "MIT"


def test_html_licence_page_is_read_as_text():
    html = ("<html><head><title>x</title><style>p{color:red}</style></head><body><nav>menu</nav>"
            "<p>Permission is hereby granted, <b>free of charge</b>, to any person obtaining a copy of this software "
            "and <i>associated documentation files</i> (the &quot;Software&quot;), to deal in the Software.</p>"
            "<p>THE SOFTWARE IS PROVIDED &quot;AS IS&quot;.</p></body></html>")
    assert detect_licence(html) == "MIT"


@pytest.mark.parametrize("text", [
    "", "   \n\t ", "\x00\x01\x02binary\xff" * 50, "All rights reserved. Do not copy.",
    "This project is licensed under a custom licence. Ask first.",
    BSD_ADVERT, GPL1, LGPL21, MPL11, CC_BY_NC,
    "x" * 5000 + MIT,                       # a licence far below the start of the file
    "# My project\n" + "text\n" * 900 + MIT,  # README that quotes MIT later
    BSD2.split("2. Redistributions")[0],      # only half of the BSD text
])
def test_not_detected(text):
    assert detect_licence(text) is None


def test_gpl_that_mentions_lgpl_and_agpl_is_gpl():
    assert detect_licence(GPL3) == "GPL-3.0"


def test_two_licences_in_one_file_are_not_guessed():
    assert detect_licence(MIT + "\n\n" + APACHE) in (None, "MIT")  # never anything but the first, or nothing


@pytest.mark.parametrize("bad", [None, 5, b"bytes"])
def test_never_raises(bad):
    assert detect_licence(bad) is None


# --- reading the file ----------------------------------------------------------

def test_read_licence_file(tmp_path):
    (tmp_path / "LICENSE").write_text(MIT, encoding="utf-8")
    assert read_licence_file(tmp_path, "LICENSE") == MIT


def test_read_licence_file_edge_cases(tmp_path):
    assert read_licence_file(tmp_path, "LICENSE") is None            # missing
    assert read_licence_file(tmp_path, "") is None                   # no name
    assert read_licence_file(tmp_path, "per-document") is None       # licence inside documents
    (tmp_path / "BIG").write_bytes(b"a" * (200 * 1024 + 1))
    assert read_licence_file(tmp_path, "BIG") is None                # too large
    (tmp_path / "BAD").write_bytes(b"\xff\xfe\x00bad")
    assert read_licence_file(tmp_path, "BAD") is None                # not UTF-8
    inner = tmp_path / "inner"
    inner.mkdir()
    (tmp_path / "OUTSIDE").write_text(MIT, encoding="utf-8")
    assert read_licence_file(inner, "../OUTSIDE") is None            # stays inside the folder


# --- cross-check -----------------------------------------------------------------

@pytest.mark.parametrize("declared,detected,kind,basis", [
    ("MIT", "MIT", "match", "detected"),
    ("mit", "MIT", "match", "detected"),
    (" MIT ", "MIT", "match", "detected"),
    ("GPL-3.0-only", "GPL-3.0", "match", "detected"),
    ("GPL-3.0-or-later", "GPL-3.0", "match", "detected"),
    ("MIT OR Apache-2.0", "MIT", "match", "detected"),
    ("MIT OR Apache-2.0", "Apache-2.0", "match", "detected"),
    ("MIT", "", "undetected", "declared"),
    ("MIT", None, "undetected", "declared"),
    ("unknown", "", "undetected", "declared"),
    ("MIT", "Apache-2.0", "mismatch", "declared"),
    ("MIT OR Apache-2.0", "BSD-3-Clause", "mismatch", "declared"),
    ("unknown", "MIT", "mismatch", "declared"),
    ("GPL-3.0", "LGPL-3.0", "mismatch", "declared"),
])
def test_cross_check_table(declared, detected, kind, basis):
    out = cross_check(declared, detected)
    assert isinstance(out, Outcome)
    assert (out.kind, out.basis) == (kind, basis)
    assert bool(out.warning) == (kind == "mismatch")
    if kind == "mismatch":
        assert declared.strip() in out.warning and detected in out.warning


# --- record integration ------------------------------------------------------------

ON = date(2026, 10, 29)
SHA = "a" * 40


@pytest.fixture
def rules(tmp_path):
    p = tmp_path / "rules.yaml"
    p.write_text("allowed: [MIT, Apache-2.0]\n", encoding="utf-8")
    return LicenceRules.load(p)


def rec(status="approved", declared="MIT", **kw):
    return SourceRecord(
        id="demo", title="T", owner="O", url="https://example.org", kind="docs", area="python",
        licence=Licence(id=declared, file_url="https://x/LICENSE", covers="both", file_hash="sha256:l"),
        version_ref=SHA, fetched=None, content_hash="sha256:c", status=status,
        fetch=FetchSpec(git="https://github.com/a/b"), **kw)


def result(detected="", licence_hash="sha256:l"):
    return FetchResult(SHA, "sha256:c", licence_hash, "LICENSE", detected_licence=detected)


def test_apply_fetch_match_sets_basis_detected():
    out = apply_fetch(rec(), result("MIT"), on=ON)
    assert out.licence.basis == "detected" and out.licence_warning == "" and out.status == "approved"


def test_apply_fetch_undetected_keeps_declared():
    out = apply_fetch(rec(), result(""), on=ON)
    assert out.licence.basis == "declared" and out.licence_warning == "" and out.status == "approved"


@pytest.mark.parametrize("status", ["approved", "limited"])
def test_apply_fetch_mismatch_warns_and_quarantines(status):
    out = apply_fetch(rec(status), result("Apache-2.0"), on=ON)
    assert out.licence_warning and "Apache-2.0" in out.licence_warning and "MIT" in out.licence_warning
    assert out.status == "quarantine" and out.licence.basis == "declared"


def test_apply_fetch_mismatch_leaves_other_statuses():
    for status in ("quarantine", "reference-only", "rejected"):
        assert apply_fetch(rec(status), result("Apache-2.0"), on=ON).status == status


def test_later_matching_fetch_clears_the_warning():
    warned = apply_fetch(rec(), result("Apache-2.0"), on=ON)
    fixed = apply_fetch(replace(warned, licence=replace(warned.licence, id="Apache-2.0")), result("Apache-2.0"), on=ON)
    assert fixed.licence_warning == "" and fixed.licence.basis == "detected"


def test_set_status_refuses_with_warning_and_no_reason(rules):
    warned = replace(rec("quarantine"), licence_warning="the licence file says Apache-2.0, the record says MIT")
    for target in ("approved", "limited"):
        with pytest.raises(TransitionError, match="licence file"):
            set_status(warned, target, rules, reviewer="Me", on=ON)


def test_set_status_accepts_warning_with_reason(rules):
    warned = replace(rec("quarantine"), licence_warning="w")
    out = set_status(warned, "approved", rules, reviewer="Me", on=ON, override_reason=" file is dual licensed ")
    assert out.status == "approved" and out.override_reason == "file is dual licensed"


def test_set_status_other_moves_ignore_the_warning(rules):
    warned = replace(rec("quarantine"), licence_warning="w")
    assert set_status(warned, "reference-only", rules, reviewer="Me", on=ON).status == "reference-only"
    assert set_status(warned, "rejected", rules, reviewer="Me", on=ON).status == "rejected"


def test_recheck_licence_clears_after_correction(rules):
    warned = apply_fetch(rec("quarantine"), result("Apache-2.0"), on=ON)
    assert warned.licence_warning
    corrected = replace(warned, licence=replace(warned.licence, id="Apache-2.0"))
    clean = recheck_licence(corrected, "Apache-2.0")
    assert clean.licence_warning == "" and clean.licence.basis == "detected"
    assert set_status(clean, "approved", rules, reviewer="Me", on=ON).status == "approved"


def test_warning_survives_save_and_load(tmp_path):
    r = replace(rec("quarantine"), licence_warning="the licence file says X, the record says MIT")
    p = tmp_path / "r.yaml"
    save_record(r, p)
    assert load_record(p) == r
    assert load_record(p).licence_warning == r.licence_warning
