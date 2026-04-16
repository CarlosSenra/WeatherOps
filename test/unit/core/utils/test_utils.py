"""Unit tests for core/utils."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.utils.models.file_management import MapCsvFilesInput, MapCsvFilesOutput
from core.utils.manege_files import map_csv_files_by_name
from core.utils import MapCsvFilesInput as UtilsInput, MapCsvFilesOutput as UtilsOutput, map_csv_files_by_name as util_fn


# ---------------------------------------------------------------------------
# MapCsvFilesInput
# ---------------------------------------------------------------------------

def test_map_csv_files_input_valid() -> None:
    inp = MapCsvFilesInput(root_path="/tmp", search_names=["salvador"])
    assert inp.root_path == "/tmp"
    assert inp.search_names == ["salvador"]


def test_map_csv_files_input_multiple_names() -> None:
    inp = MapCsvFilesInput(root_path="/data", search_names=["foo", "bar", "baz"])
    assert len(inp.search_names) == 3


def test_map_csv_files_output_structure() -> None:
    out = MapCsvFilesOutput(results={"salvador": ["/tmp/salvador_001.csv"]})
    assert out.results["salvador"] == ["/tmp/salvador_001.csv"]


def test_map_csv_files_output_empty_results() -> None:
    out = MapCsvFilesOutput(results={"term": []})
    assert out.results["term"] == []


# ---------------------------------------------------------------------------
# map_csv_files_by_name
# ---------------------------------------------------------------------------

def test_finds_csv_file_matching_name(tmp_path: Path) -> None:
    (tmp_path / "salvador_001.csv").write_text("col1,col2\n1,2")

    results = map_csv_files_by_name(str(tmp_path), ["salvador"])

    assert len(results["salvador"]) == 1
    assert "salvador_001.csv" in results["salvador"][0]


def test_ignores_non_csv_files(tmp_path: Path) -> None:
    (tmp_path / "data.txt").write_text("not a csv")
    (tmp_path / "report.xlsx").write_bytes(b"fake xlsx")

    results = map_csv_files_by_name(str(tmp_path), ["data"])
    assert results["data"] == []


def test_search_is_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "BRASILIA_2024.CSV").write_text("col\n1")

    results = map_csv_files_by_name(str(tmp_path), ["brasilia"])
    assert len(results["brasilia"]) == 1


def test_searches_subdirectories_recursively(tmp_path: Path) -> None:
    subdir = tmp_path / "2024"
    subdir.mkdir()
    (subdir / "salvador_2024.csv").write_text("x\n1")

    results = map_csv_files_by_name(str(tmp_path), ["salvador"])
    assert len(results["salvador"]) == 1


def test_empty_search_names_returns_empty_dict(tmp_path: Path) -> None:
    (tmp_path / "any.csv").write_text("x\n1")
    results = map_csv_files_by_name(str(tmp_path), [])
    assert results == {}


def test_no_match_returns_empty_list(tmp_path: Path) -> None:
    (tmp_path / "other.csv").write_text("x\n1")
    results = map_csv_files_by_name(str(tmp_path), ["brasilia"])
    assert results["brasilia"] == []


def test_multiple_search_names_independent(tmp_path: Path) -> None:
    (tmp_path / "salvador_01.csv").write_text("x\n1")
    (tmp_path / "brasilia_01.csv").write_text("y\n2")

    results = map_csv_files_by_name(str(tmp_path), ["salvador", "brasilia"])
    assert len(results["salvador"]) == 1
    assert len(results["brasilia"]) == 1


def test_same_file_matches_multiple_terms(tmp_path: Path) -> None:
    (tmp_path / "salvador_brasilia.csv").write_text("x\n1")

    results = map_csv_files_by_name(str(tmp_path), ["salvador", "brasilia"])
    assert len(results["salvador"]) == 1
    assert len(results["brasilia"]) == 1


# ---------------------------------------------------------------------------
# Re-exports from core.utils
# ---------------------------------------------------------------------------

def test_utils_module_exports_all_symbols() -> None:
    assert UtilsInput is MapCsvFilesInput
    assert UtilsOutput is MapCsvFilesOutput
    assert util_fn is map_csv_files_by_name
