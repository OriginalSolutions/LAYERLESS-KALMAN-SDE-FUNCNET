# plik: config_loader.py
# WERSJA: 80.0 (Uniwersalna, dynamiczna)

import commentjson as json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Config:
    """
    Uniwersalna klasa wczytująca konfigurację z dowolnego pliku JSON.
    Dynamicznie tworzy atrybuty obiektu na podstawie wszystkich znalezionych kluczy.
    """
    def __init__(self, config_path: str):
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Plik konfiguracyjny nie został znaleziony: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except Exception as e:
            logger.critical(f"Krytyczny błąd podczas parsowania pliku JSON '{config_path}': {e}")
            raise ValueError(f"Nie można wczytać pliku konfiguracyjnego: {config_path}") from e

        for key, value in config_data.items():
            setattr(self, key, value)
        logger.info(f"Pomyślnie wczytano {len(config_data)} kluczy konfiguracyjnych z pliku: '{config_path}'.")

    def get(self, key, default=None):
        return getattr(self, key, default)