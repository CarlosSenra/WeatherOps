from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import pytest
from urllib.error import URLError

from src.data_airflow.scraping.inmet import (
    YearRangeConfig,
    download_bytes,
    extract_csv_zip_overwrite,
    find_year_zip_url,
    load_year_range_from_yaml,
)


def _make_zip(files: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buffer.getvalue()


def test_year_range_is_inclusive() -> None:
    cfg = YearRangeConfig(start_year=2024, end_year=2026)
    assert cfg.years() == [2024, 2025, 2026]


def test_year_range_validation() -> None:
    cfg = YearRangeConfig(start_year=2026, end_year=2024)
    with pytest.raises(ValueError):
        cfg.years()


def test_load_year_range_from_yaml(tmp_path: Path) -> None:
    config = tmp_path / "inmet_scraping.yml"
    config.write_text("start_year: 2023\nend_year: 2025\n", encoding="utf-8")

    loaded = load_year_range_from_yaml(config)
    assert loaded.start_year == 2023
    assert loaded.end_year == 2025
    assert loaded.years() == [2023, 2024, 2025]


def test_find_year_zip_url_prefers_year_match() -> None:
    html = """
    <html><body>
      <a href="/files/outro.zip">arquivo sem ano</a>
      <a href="/files/inmet_2025.zip">ANO 2025 (AUTOMATICA)</a>
      <a href="/files/inmet_2026.zip">ANO 2026 (AUTOMATICA)</a>
    </body></html>
    """
    url = find_year_zip_url("https://portal.inmet.gov.br/dadoshistoricos", 2026, html)
    assert url.endswith("/files/inmet_2026.zip")


def test_find_year_zip_url_raises_when_missing() -> None:
    with pytest.raises(ValueError):
        find_year_zip_url("https://portal.inmet.gov.br/dadoshistoricos", 2027, "<html></html>")


def test_extract_csv_zip_overwrite_replaces_existing_content(tmp_path: Path) -> None:
    out = tmp_path / "2026"
    out.mkdir(parents=True)
    (out / "old_file.csv").write_text("old", encoding="utf-8")

    zip_bytes = _make_zip(
        {
            "folder/salvador_001.csv": b"a,b\n1,2\n",
            "folder/readme.txt": b"ignore",
            "folder/salvador_002.csv": b"a,b\n3,4\n",
        }
    )
    count = extract_csv_zip_overwrite(zip_bytes, out)

    assert count == 2
    assert not (out / "old_file.csv").exists()
    csv_files = sorted(p.name for p in out.glob("*.csv"))
    assert csv_files == ["salvador_001.csv", "salvador_002.csv"]


def test_extract_csv_zip_overwrite_handles_name_collision(tmp_path: Path) -> None:
    out = tmp_path / "2024"
    zip_bytes = _make_zip(
        {
            "a/data.csv": b"1",
            "b/data.csv": b"2",
        }
    )
    count = extract_csv_zip_overwrite(zip_bytes, out)

    assert count == 2
    csv_files = sorted(p.name for p in out.glob("*.csv"))
    assert csv_files == ["data.csv", "data_1.csv"]


def test_download_bytes_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b"ok"

    def fake_urlopen(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["n"] += 1
        if calls["n"] < 3:
            raise URLError("temporary failure")
        return _Response()

    monkeypatch.setattr("src.data_airflow.scraping.inmet.urlopen", fake_urlopen)
    monkeypatch.setattr("src.data_airflow.scraping.inmet.time.sleep", lambda *_: None)

    result = download_bytes("https://example.com", max_retries=4, retry_delays_s=(0, 0, 0))
    assert result == b"ok"
    assert calls["n"] == 3


def test_download_bytes_raises_after_retry_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    def always_fail(*args, **kwargs):  # noqa: ANN002, ANN003
        raise URLError("down")

    monkeypatch.setattr("src.data_airflow.scraping.inmet.urlopen", always_fail)
    monkeypatch.setattr("src.data_airflow.scraping.inmet.time.sleep", lambda *_: None)

    with pytest.raises(RuntimeError, match="Falha ao baixar URL"):
        download_bytes("https://example.com", max_retries=3, retry_delays_s=(0, 0))

