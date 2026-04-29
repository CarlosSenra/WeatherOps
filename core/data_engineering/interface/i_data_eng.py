from abc import ABC, abstractmethod

class IDataCleaning(ABC):

    @abstractmethod
    def _load_all_data(self):
        pass

    @abstractmethod
    def _default_read_csv(self):
        pass

    @abstractmethod
    def _cleaning_str_data_hours_columns(self):
        pass
    