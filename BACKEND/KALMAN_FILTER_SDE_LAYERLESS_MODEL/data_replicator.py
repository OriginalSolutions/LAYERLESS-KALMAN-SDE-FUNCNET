# file: data_replicator.py
# WERSJA: 1.0 - Tworzy bezpieczną replikę danych, aby uniknąć blokad.

import time
import logging
import os
import shutil

# Konfiguracja
SOURCE_HDF5_PATH = './pure_sde_data/sde_data.h5'
REPLICA_HDF5_PATH = './pure_sde_data/sde_data_REPLICA.h5'
CHECK_INTERVAL_SECONDS = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s|REPLICATOR|%(levelname)-8s|%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("Replicator_v1.0")

def get_file_mod_time(path):
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return 0

def run_replicator():
    logger.info(f"--- 🚀 Uruchamianie Replikatora Danych (v1.0) 🚀 ---")
    last_mod_time = 0

    while True:
        try:
            source_mod_time = get_file_mod_time(SOURCE_HDF5_PATH)
            if source_mod_time > last_mod_time:
                logger.info(f"Wykryto zmianę w pliku źródłowym. Tworzenie repliki...")
                try:
                    shutil.copy(SOURCE_HDF5_PATH, REPLICA_HDF5_PATH)
                    last_mod_time = source_mod_time
                    logger.info(f"✅ Pomyślnie utworzono replikę: '{REPLICA_HDF5_PATH}'")
                except Exception as e:
                    logger.error(f"Nie udało się skopiować pliku: {e}")
            
            time.sleep(CHECK_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logger.info("🛑 Zamykanie."); break
        except Exception as e:
            logger.error(f"Błąd krytyczny: {e}", exc_info=True); time.sleep(30)

if __name__ == "__main__":
    run_replicator()
