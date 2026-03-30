from pathlib import Path
from typing import List, Dict

def map_csv_files_by_name(root_path: str, search_names: List[str]) -> Dict[str, List[str]]:
    """
    Percorre pastas recursivamente buscando arquivos .csv que contenham 
    os nomes da lista no título.
    """
    root = Path(root_path)
    
    
    result = {name: [] for name in search_names}
    
    
    for csv_file in root.rglob('*.csv'):
        
        file_name_lower = csv_file.name.lower()
        
        for search_name in search_names:
            
            if search_name.lower() in file_name_lower:
                result[search_name].append(str(csv_file.absolute()))
                
    return result

#test_dict = map_csv_files_by_name(root_path='../data/raw/', search_names=['salvador_'])