"""Helpers para baixar e extrair dados históricos anuais do INMET.

Este módulo não depende de Airflow para facilitar testes unitários.
"""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from http.client import RemoteDisconnected
from io import BytesIO
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import zipfile

import yaml

INMET_HISTORICAL_URL = "https://portal.inmet.gov.br/dadoshistoricos"
DEFAULT_HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; WeatherOpsBot/1.0; +https://github.com/)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}


@dataclass(frozen=True)
class YearRangeConfig:
    start_year: int
    end_year: int

    def years(self) -> list[int]:
        if self.start_year > self.end_year:
            raise ValueError("start_year deve ser menor ou igual a end_year.")
        return list(range(self.start_year, self.end_year + 1))


class _AnchorParser(HTMLParser):
    """Parser mínimo para coletar href + texto de links."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._current_href: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self._current_href = href
            self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current_href is not None:
            text = " ".join(part.strip() for part in self._text_parts if part.strip())
            self.links.append((self._current_href, text))
            self._current_href = None
            self._text_parts = []


def load_year_range_from_yaml(config_path: str | Path) -> YearRangeConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    start_year = data.get("start_year")
    end_year = data.get("end_year")

    if not isinstance(start_year, int) or not isinstance(end_year, int):
        raise ValueError("YAML inválido: start_year e end_year devem ser inteiros.")

    return YearRangeConfig(start_year=start_year, end_year=end_year)


def find_year_zip_url(page_url: str, year: int, html: str) -> str:
    """Encontra a URL do ZIP do ano informado na página HTML."""
    parser = _AnchorParser()
    parser.feed(html)

    year_token = str(year)
    candidates: list[tuple[int, str]] = []
    for href, text in parser.links:
        href_l = href.lower()
        text_l = text.lower()
        score = 0
        if ".zip" in href_l:
            score += 2
        if year_token in href_l:
            score += 4
        if year_token in text_l:
            score += 3
        if re.search(rf"\b{year}\b", f"{href} {text}"):
            score += 1

        if score > 0 and ".zip" in href_l:
            candidates.append((score, urljoin(page_url, href)))

    if not candidates:
        raise ValueError(f"Nenhum link ZIP encontrado para o ano {year}.")

    # Score alto primeiro; em empate mantém primeira ocorrência.
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def download_bytes(
    url: str,
    timeout_s: int = 60,
    max_retries: int = 4,
    retry_delays_s: tuple[float, ...] = (2.0, 3.0, 5.0),
) -> bytes:
    """Baixa bytes com retries para falhas transitórias de conexão."""
    if max_retries < 1:
        raise ValueError("max_retries deve ser >= 1.")

    attempt = 0
    while True:
        attempt += 1
        try:
            request = Request(url, headers=DEFAULT_HTTP_HEADERS)
            with urlopen(request, timeout=timeout_s) as response:  # noqa: S310
                return response.read()
        except (RemoteDisconnected, TimeoutError, URLError, HTTPError, ConnectionResetError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Falha ao baixar URL após {attempt} tentativas: {url}"
                ) from exc

            delay_idx = min(attempt - 1, max(0, len(retry_delays_s) - 1))
            sleep_s = retry_delays_s[delay_idx] if retry_delays_s else 2.0
            time.sleep(sleep_s)


def extract_csv_zip_overwrite(zip_bytes: bytes, output_dir: str | Path) -> int:
    """Sobrescreve output_dir e extrai apenas CSVs do zip para ele."""
    out = Path(output_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    names_seen: set[str] = set()

    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            filename = Path(member.filename).name
            if not filename.lower().endswith(".csv"):
                continue

            # Evita colisão de nome quando houver subpastas no zip.
            final_name = filename
            if final_name in names_seen:
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                idx = 1
                while f"{stem}_{idx}{suffix}" in names_seen:
                    idx += 1
                final_name = f"{stem}_{idx}{suffix}"
            names_seen.add(final_name)

            target = out / final_name
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_count += 1

    if extracted_count == 0:
        raise ValueError("ZIP não contém arquivos CSV.")
    return extracted_count
