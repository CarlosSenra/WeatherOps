from pathlib import Path
from typing import List, Dict
import logging
import re
import unicodedata

from .models import MapCsvFilesInput, MapCsvFilesOutput

logger = logging.getLogger(__name__)

_INMET_FILENAME_REGEX = re.compile(
    r"^INMET_[^_]+_[^_]+_[A-Z0-9]+_(?P<municipio>.+?)_"
    r"\d{2}-\d{2}-\d{4}_A_\d{2}-\d{2}-\d{4}\.csv$",
    re.IGNORECASE,
)


def slugify_municipio(municipio: str) -> str:
    """Normaliza nome do município para slug ASCII em snake_case."""
    normalized = unicodedata.normalize("NFKD", municipio)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def parse_inmet_filename(filename: str) -> tuple[str, str] | None:
    """
    Extrai município original e slug a partir de filename INMET.

    Exemplo aceito:
      INMET_S_RS_A881_DOM PEDRITO_01-01-2026_A_31-03-2026.CSV
    """
    match = _INMET_FILENAME_REGEX.match(filename)
    if not match:
        return None

    municipio_original = match.group("municipio").strip()
    municipio_slug = slugify_municipio(municipio_original)
    if not municipio_slug:
        return None
    return municipio_original, municipio_slug


def map_inmet_csv_files_by_municipio(root_path: str) -> Dict[str, List[str]]:
    """
    Percorre `root_path` recursivamente e agrupa CSVs INMET por município slug.
    """
    root = Path(root_path)
    result: Dict[str, List[str]] = {}

    for file_path in root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() != ".csv":
            continue

        parsed = parse_inmet_filename(file_path.name)
        if not parsed:
            continue

        _, municipio_slug = parsed
        result.setdefault(municipio_slug, []).append(str(file_path.absolute()))

    # Determinismo para logs/testes
    return {k: sorted(v) for k, v in sorted(result.items(), key=lambda x: x[0])}


def apply_municipio_filters(
    grouped_files: Dict[str, List[str]],
    include: List[str] | None = None,
    exclude: List[str] | None = None,
    slug_overrides: Dict[str, str] | None = None,
) -> Dict[str, List[str]]:
    """
    Aplica include/exclude/overrides sobre dicionário {municipio_slug: [paths]}.
    """
    include_set = set(include or [])
    exclude_set = set(exclude or [])
    overrides = slug_overrides or {}

    remapped: Dict[str, List[str]] = {}
    for municipio_slug, paths in grouped_files.items():
        target_slug = overrides.get(municipio_slug, municipio_slug)
        remapped.setdefault(target_slug, []).extend(paths)

    filtered: Dict[str, List[str]] = {}
    for municipio_slug, paths in remapped.items():
        if include_set and municipio_slug not in include_set:
            continue
        if municipio_slug in exclude_set:
            continue
        filtered[municipio_slug] = sorted(paths)

    return {k: v for k, v in sorted(filtered.items(), key=lambda x: x[0])}


def map_csv_files_by_name(root_path: str, search_names: List[str]) -> Dict[str, List[str]]:
    """
    Percorre pastas recursivamente buscando arquivos .csv que contenham 
    os nomes da lista no título.
    
    Args:
        root_path: Caminho raiz para a busca de arquivos CSV
        search_names: Lista de nomes para filtrar os arquivos CSV
    
    Returns:
        Dict[str, List[str]]: Dicionário onde as chaves são os nomes de busca 
                              e os valores são listas de caminhos absolutos dos arquivos
    """
    # Validar entrada usando Pydantic
    input_data = MapCsvFilesInput(root_path=root_path, search_names=search_names)
    
    root = Path(input_data.root_path)
    logger.info(f"Procurando arquivos CSV em: {input_data.root_path} com termos: {input_data.search_names}")
    
    result = {name: [] for name in input_data.search_names}
    
    def search_in_directory(directory: Path):
        """Função recursiva para buscar CSVs nas subpastas"""
        try:
            for item in directory.iterdir():
                if item.is_dir():
                    logger.debug(f"Entrando na pasta: {item.name}")
                    search_in_directory(item)
                
                elif item.is_file() and item.suffix.lower() == '.csv':
                    file_name_lower = item.name.lower()
                    
                    for search_name in input_data.search_names:
                        if search_name.lower() in file_name_lower:
                            logger.debug(f"  ✓ Match com: {search_name}")
                            result[search_name].append(str(item.absolute()))
        
        except PermissionError as e:
            logger.warning(f"Sem permissão: {directory} - {e}")
        except Exception as e:
            logger.error(f"Erro ao processar {directory}: {e}")
    
    # Começar a busca a partir da raiz
    search_in_directory(root)
    
    # Validar saída usando Pydantic
    output_data = MapCsvFilesOutput(results=result)
    logger.info(f"Resultado final: {output_data.results}")
    
    return output_data.results

#test_dict = map_csv_files_by_name(root_path='../data/raw/', search_names=['salvador_'])