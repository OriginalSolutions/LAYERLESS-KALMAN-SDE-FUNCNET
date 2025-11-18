# file: accuracy_monitor.py
# WERSJA: 2.0 (NIEZAWODNA) - Odporny na błędy, nie blokuje pliku źródłowego.

import pandas as pd
import numpy as np
import h5py
import time
import logging
import os

# ==============================================================================
# KONFIGURACJA
# ==============================================================================
INPUT_HDF5_PATH = './pure_sde_data/sde_data.h5'
OUTPUT_HDF5_PATH = './pure_sde_data/accuracy_metrics.h5'
CHECK_INTERVAL_SECONDS = 5
ACCURACY_WINDOW_SIZE = 30
FORECASTS_TO_ANALYZE = {
    'raw_forecast': 'Accuracy_Raw',
    'arimax_corrected_forecast': 'Accuracy_ARIMAX',
    'final_predictions': 'Accuracy_Final'
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s|ACC_MONITOR|%(levelname)-8s|%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("AccuracyMonitor_v2.0")

# ==============================================================================
# GŁÓWNE FUNKCJE
# ==============================================================================

def calculate_rolling_accuracy(df: pd.DataFrame, price_col: str, forecast_col: str) -> pd.Series:
    """Oblicza kroczącą dokładność kierunkową na całym DataFrame."""
    if forecast_col not in df.columns or df[forecast_col].isnull().all():
        return pd.Series(np.nan, index=df.index)

    valid_mask = df[price_col].notna() & df[price_col].shift(1).notna()
    price_diff = df.loc[valid_mask, price_col].diff()
    forecast_diff = df.loc[valid_mask, forecast_col].diff()

    hits = (np.sign(price_diff) == np.sign(forecast_diff)).astype(float)
    hits[price_diff == 0] = 1.0 # Uznaj brak zmiany za trafienie
    
    rolling_accuracy = hits.rolling(window=ACCURACY_WINDOW_SIZE, min_periods=1).mean() * 100
    
    return rolling_accuracy.reindex(df.index)

def run_live_monitor():
    logger.info(f"--- 🚀 Uruchamianie Monitora Dokładności (v2.0 - Niezawodny) 🚀 ---")
    last_processed_count = 0

    while True:
        try:
            # 1. Szybko sprawdź, czy plik źródłowy ma nowe dane
            current_total_count = 0
            try:
                with h5py.File(INPUT_HDF5_PATH, 'r') as f:
                    if 'timestamps' in f:
                        current_total_count = f['timestamps'].shape[0]
            except (IOError, FileNotFoundError, OSError):
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            if current_total_count <= last_processed_count:
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue

            # 2. Skoro są nowe dane, wczytaj cały plik do pamięci i natychmiast go zamknij
            logger.info(f"Wykryto {current_total_count - last_processed_count} nowych rekordów. Przetwarzam cały plik...")
            try:
                with h5py.File(INPUT_HDF5_PATH, 'r') as f:
                    cols_to_read = ['timestamps', 'prices'] + list(FORECASTS_TO_ANALYZE.keys())
                    data = {col: f[col][:] for col in cols_to_read if col in f}
                    source_df = pd.DataFrame(data)
            except Exception as e:
                logger.error(f"Nie udało się wczytać danych z '{INPUT_HDF5_PATH}': {e}")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue
            
            # Plik źródłowy jest zamknięty. Blokada zwolniona.

            # 3. Wykonaj wszystkie obliczenia na danych w pamięci
            output_df = pd.DataFrame({'Timestamp': source_df['timestamps']})
            for forecast_col, output_col in FORECASTS_TO_ANALYZE.items():
                if forecast_col in source_df.columns:
                    output_df[output_col] = calculate_rolling_accuracy(source_df, 'prices', forecast_col)

            # 4. Nadpisz plik wynikowy (to jest OK, bo tylko ten skrypt do niego pisze)
            with h5py.File(OUTPUT_HDF5_PATH, 'w') as f_out:
                for column in output_df.columns:
                    if output_df[column].notna().any():
                        f_out.create_dataset(column, data=output_df[column].to_numpy())
                f_out.attrs['last_update_utc'] = pd.Timestamp.utcnow().isoformat()

            logger.info(f"✅ Plik metryk '{OUTPUT_HDF5_PATH}' pomyślnie nadpisany. Rekordów: {len(output_df)}")
            last_processed_count = current_total_count

        except KeyboardInterrupt:
            logger.info("🛑 Zamykanie."); break
        except Exception as e:
            logger.error(f"Błąd krytyczny: {e}", exc_info=True); time.sleep(30)

if __name__ == "__main__":
    run_live_monitor()
