"""A cancelled or empty upload is a refused user input (NOTEBOOK_SPEC 3.1 DAT20).

The Section 4 upload and the Section 10 artifact upload run the generated notebook's own cells, with ``google.colab``'s
upload dialog stood in and returning nothing. Each must raise the notebook's ``Refused`` (so Run all stops at that
cell) and render exactly one ``Refused: (no file): the upload was cancelled or empty ...`` line with no traceback.
On main the same upload raised a plain ``RuntimeError``, shown with a traceback.
"""
# ruff: noqa: E501

from __future__ import annotations

import pytest

import test_refusal_display as rd


def _upload_cell(where: str) -> str:
    if where == "section4":
        cell = rd._set(rd._cell("USE_BYOD = False  # @param"), "USE_BYOD", True)
        if "\nDATA_SOURCE = " in "\n" + cell:
            cell = rd._set(cell, "DATA_SOURCE", "Upload CSV")
        return cell
    return rd._set(rd._cell("ARTIFACT_SOURCE = 'Pinned sample artifact'"), "ARTIFACT_SOURCE", "My own artifact ZIP")


@pytest.mark.parametrize("where", ["section4", "section10"])
def test_cancelled_or_empty_upload_is_one_refused_line(tmp_path, where) -> None:
    ns = rd._kernel(tmp_path)
    with rd._colab({}):
        raised, shown, ran = rd._run_all(ns, [_upload_cell(where), "raise AssertionError('Run all went past the refusing cell')"])
    line = rd._assert_refused(raised, shown, ran, "(no file)")
    assert line.startswith("Refused: (no file): the upload was cancelled or empty: no file was received"), line
    assert "Run this cell again" in line, line
