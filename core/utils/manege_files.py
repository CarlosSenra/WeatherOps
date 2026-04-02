from pathlib import Path
from typing import List, Dict
import os 

def map_csv_files_by_name(root_path: str, search_names: List[str]) -> Dict[str, List[str]]:
    """
    Percorre pastas recursivamente buscando arquivos .csv que contenham 
    os nomes da lista no título.
    """
    root = Path(root_path)
    print(f"Procurando arquivos CSV em: {root_path} com termos: {search_names}")
    print(f"Conteúdo: {os.listdir(root_path)}")
    
    result = {name: [] for name in search_names}
    
    def search_in_directory(directory: Path):
        """Função recursiva para buscar CSVs nas subpastas"""
        try:
            for item in directory.iterdir():
                if item.is_dir():
                    print(f"Entrando na pasta: {item.name}")
                    search_in_directory(item)
                
                elif item.is_file() and item.suffix.lower() == '.csv':
                    file_name_lower = item.name.lower()
                    
                    for search_name in search_names:
                        if search_name.lower() in file_name_lower:
                            print(f"  ✓ Match com: {search_name}")
                            result[search_name].append(str(item.absolute()))
        
        except PermissionError as e:
            print(f"Sem permissão: {directory} - {e}")
        except Exception as e:
            print(f"Erro ao processar {directory}: {e}")
    
    # Começar a busca a partir da raiz
    search_in_directory(root)
    
    print(f"Resultado final: {result}")
    return result

#test_dict = map_csv_files_by_name(root_path='../data/raw/', search_names=['salvador_'])