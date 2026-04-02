from pathlib import Path
from typing import List, Dict
import os 
import logging

from .models import MapCsvFilesInput, MapCsvFilesOutput

logger = logging.getLogger(__name__)


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