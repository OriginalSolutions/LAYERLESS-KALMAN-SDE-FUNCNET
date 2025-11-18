# plik: analyzer_model/run_analyzer.py
# WERSJA: 63.0 (OSTATECZNA, DZIAŁAJĄCA)

import logging
import time
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / 'KALMAN_FILTER_SDE_LAYERLESS_MODEL'))

from config_loader import Config
from hdf5_manager import UltraAdvancedHDF5Manager
from learning_engine import PureFunctionalOnlineLearningEngine
from ai_feature_manager import AIFeatureHDF5Manager

def setup_logging(level: str):
    logging.basicConfig(level=level.upper(), 
                        format='%(asctime)s|ANALYZER|%(levelname)-8s|%(name)-15s|%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

def main():
    try:
        config = Config(config_path='analyzer_config.json')
        setup_logging(config.system_log_level)
    except Exception as e:
        logging.critical(f"KRYTYCZNY BŁĄD podczas wczytywania konfiguracji: {e}", exc_info=True)
        return
        
    logger = logging.getLogger(__name__)
    logger.info("--- 🤖 URUCHAMIANIE PRODUCENTA CECH AI 🤖 ---")

    try:
        main_storage = UltraAdvancedHDF5Manager(config=config, filename=config.storage_main_hdf5_filename)
        ai_feature_columns = [f'ai_drift_{feat}' for feat in config.model_feature_columns] + \
                             [f'ai_diffusion_{feat}' for feat in config.model_feature_columns]
        ai_storage = AIFeatureHDF5Manager(config=config, ai_feature_columns=ai_feature_columns)
        engine = PureFunctionalOnlineLearningEngine(config)
    except Exception as e:
        logger.critical(f"Krytyczny błąd podczas inicjalizacji komponentów: {e}", exc_info=True)
        return

    last_processed_ts = 0
    is_first_run = True

    logger.info(f"Analityk obserwuje plik: {main_storage.filename}")
    logger.info(f"Analityk produkuje cechy AI do pliku: {ai_storage.filename}")

    while True:
        try:
            logger.info("\n--- Nowy cykl analizy ---")
            
            full_df = main_storage.get_all_data_as_dataframe()
            df = full_df.tail(config.analyzer_lookback_minutes).copy()
            
            min_data_initial = config.prediction_input_history_minutes + 2
            
            if df.empty or len(df) < min_data_initial:
                logger.warning(f"Brak wystarczających danych do analizy (znaleziono: {len(df)}, wymagane: {min_data_initial}). Czekam...")
                time.sleep(config.reanalysis_interval_seconds)
                continue

            current_ts = df['timestamps'].iloc[-1]
            if current_ts <= last_processed_ts:
                logger.info(f"Brak nowych danych. Następna analiza za {config.reanalysis_interval_seconds}s.")
                time.sleep(config.reanalysis_interval_seconds)
                continue

            logger.info(f"Wykryto nowe dane (ts: {current_ts}). Rozpoczynam analizę...")
            
            df['target_price'] = df['prices'].shift(-1)
            df.dropna(subset=['target_price'], inplace=True)
            
            min_data_for_sequence = config.prediction_input_history_minutes + 1
            if len(df) < min_data_for_sequence:
                logger.error(f"Po przygotowaniu danych pozostało za mało wierszy. Pomijam cykl.")
                time.sleep(config.reanalysis_interval_seconds)
                continue

            engine.run_training(df, epochs=config.training_initial_epochs, lr=config.training_initial_lr, is_initial=is_first_run)

            if engine.is_trained:
                ai_features = engine.extract_ai_features()
                
                # --- OSTATNIA POPRAWKA ---
                # Zapisujemy cechy AI tylko, jeśli zostały faktycznie wyekstrahowane
                if ai_features:
                    ai_storage.save_features(current_ts, ai_features)
                    logger.info(f"✅ Pomyślnie wyprodukowano i zapisano {len(ai_features)} cech AI dla ts: {current_ts}.")
                # --- KONIEC POPRAWKI ---
                    
                last_processed_ts = current_ts
                is_first_run = False
            
            logger.info(f"Cykl analityka zakończony. Następna analiza za {config.reanalysis_interval_seconds}s.")
            time.sleep(config.reanalysis_interval_seconds)

        except KeyboardInterrupt:
            logger.info("🛑 Zamykanie producenta cech AI...")
            break
        except Exception as e:
            logger.error(f"Błąd krytyczny w pętli analitycznej: {e}", exc_info=True)
            logger.info("System spróbuje kontynuować pracę za 20 sekund...")
            time.sleep(20)

if __name__ == "__main__":
    main()