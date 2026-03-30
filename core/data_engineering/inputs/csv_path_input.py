from typing import List
from pydantic.dataclasses import dataclass

@dataclass
class DataEngInput:
    csv_paths: List[str]