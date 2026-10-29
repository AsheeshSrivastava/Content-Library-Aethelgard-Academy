"""Every reader in use returns canonical text (spec 05 loader check: not-normalised)."""
import json

import pytest

from ragcore.contracts import normalize_text
from ragcore.loaders.registry import DEFAULT_READERS

DECOMPOSED_E = chr(0x65) + chr(0x301)
BAD = ("caf" + DECOMPOSED_E + " is a place.\r\n" + "A sentence about retrieval. " * 8 + "\r\n").encode("utf-8")


NOTEBOOK = json.dumps({"nbformat": 4, "metadata": {}, "cells": [
    {"cell_type": "markdown", "source": BAD.decode("utf-8")}]}).encode("utf-8")


@pytest.mark.parametrize("ext", sorted(DEFAULT_READERS))
def test_reader_output_is_canonical(ext):
    version, read = DEFAULT_READERS[ext]
    out = read(NOTEBOOK if ext == ".ipynb" else BAD)
    assert out and normalize_text(out) == out
    assert version
