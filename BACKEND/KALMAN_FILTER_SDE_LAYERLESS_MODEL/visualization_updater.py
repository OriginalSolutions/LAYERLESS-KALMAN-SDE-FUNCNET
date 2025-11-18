# file: visualization_updater.py
# WERSJA: 11.0 (IDIOTOODPORNA) - Ostateczna, pancerna logika dla KAMA i czyszczenia wiersza prognozy.

import pandas as pd
import numpy as np
import h5py
import time
import logging
import pandas_ta as ta
import os

# ==============================================================================
# KONFIGURACJA
# ==============================================================================
SDE_HDF5_PATH = './pure_sde_data/sde_data_REPLICA.h5'
HDF5_OUTPUT = './pure_sde_data/VISUALIZATION_DATA.hdf5'
CHECK_INTERVAL_SECONDS = 5
ACCURACY_WINDOW_SIZE = 480
KAMA_PERIOD = 60    ##  60
KAMA_FAST_PERIOD =  2  ##   2
KAMA_SLOW_PERIOD = 20  ##  15

REQUIRED_OUTPUT_COLUMNS = ['Time', 'Actual_BTC_Price', 'KAMA_Actual_BTC_Price', 'ARIMAX_Forecast', 'Forecast', 'KAMA_Forecast', 'Accuracy_ARIMAX', 'Accuracy_Final']

logging.basicConfig(level=logging.INFO, format='%(asctime)s|VISUALIZER|%(levelname)-8s|%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("Visualizer_v11.0")

# ==============================================================================
# FUNKCJE POMOCNICZE
# ==============================================================================

def get_record_count(filepath: str) -> int:
    try:
        with h5py.File(filepath, 'r') as f:
            if 'timestamps' in f: return f['timestamps'].shape[0]
    except (IOError, FileNotFoundError, OSError): return 0
    return 0

def calculate_rolling_accuracy(df: pd.DataFrame, price_col: str, forecast_col: str) -> pd.Series:
    if len(df) < 2 or forecast_col not in df.columns: return pd.Series(np.nan, index=df.index)
    df_calc = df[[price_col, forecast_col]].copy().dropna()
    if len(df_calc) < 2: return pd.Series(np.nan, index=df.index)
    price_diff, forecast_diff = df_calc[price_col].diff(), df_calc[forecast_col].diff()
    hits = np.where(np.sign(price_diff) == np.sign(forecast_diff), 1.0, 0.0)
    hits[price_diff == 0] = 1.0
    hits_series = pd.Series(hits, index=df_calc.index)
    rolling_accuracy = hits_series.rolling(window=ACCURACY_WINDOW_SIZE, min_periods=1).mean() * 100
    return rolling_accuracy.reindex(df.index)

# ==============================================================================
# GŁÓWNA PĘTLA
# ==============================================================================

def run_continuous_updater():
    logger.info(f"--- 🚀 Uruchamianie Agregatora Wizualizacji (v11.0 - Idiotoodporna) 🚀 ---")
    last_processed_count = 0

    while True:
        try:
            current_total_count = get_record_count(SDE_HDF5_PATH)

            if current_total_count > last_processed_count:
                logger.info(f"Wykryto {current_total_count - last_processed_count} nowych rekordów w replice. Pełne przetwarzanie...")
                
                df = pd.DataFrame()
                try:
                    with h5py.File(SDE_HDF5_PATH, 'r') as f:
                        cols = ['timestamps', 'prices', 'arimax_corrected_forecast', 'final_predictions']
                        data = {c: f[c][:] for c in cols if c in f}
                        df = pd.DataFrame(data)
                except Exception as e:
                    logger.error(f"Nie udało się wczytać danych z repliki '{SDE_HDF5_PATH}': {e}")
                    time.sleep(CHECK_INTERVAL_SECONDS)
                    continue
                
                # Krok 1: Oblicz wskaźniki, które nie wymagają specjalnego traktowania
                df['Accuracy_ARIMAX'] = calculate_rolling_accuracy(df, 'prices', 'arimax_corrected_forecast')
                df['Accuracy_Final'] = calculate_rolling_accuracy(df, 'prices', 'final_predictions')

                # Krok 2: PANCERNA LOGIKA DLA KAMA
                # Tworzymy tymczasowe, "załatatane" kolumny, aby zagwarantować, że KAMA zawsze dostanie kompletne dane
                prices_filled = df['prices'].interpolate(method='linear').bfill().ffill()
                final_preds_filled = df['final_predictions'].interpolate(method='linear').bfill().ffill()

                df['KAMA_Actual_BTC_Price'] = ta.kama(prices_filled, length=KAMA_PERIOD, fast=KAMA_FAST_PERIOD, slow=KAMA_SLOW_PERIOD)
                df['KAMA_Forecast'] = ta.kama(final_preds_filled, length=KAMA_PERIOD, fast=KAMA_FAST_PERIOD, slow=KAMA_SLOW_PERIOD)

                output_df = df.rename(columns={'timestamps': 'Time', 'prices': 'Actual_BTC_Price', 'arimax_corrected_forecast': 'ARIMAX_Forecast', 'final_predictions': 'Forecast'})

                # Krok 3: Ręczne czyszczenie ostatniego wiersza (prognozy t+1)
                if not output_df.empty:
                    last_row_index = output_df.index[-1]
                    cols_to_clear = ['Actual_BTC_Price', 'KAMA_Actual_BTC_Price', 'Accuracy_ARIMAX', 'Accuracy_Final']
                    for col in cols_to_clear:
                        if col in output_df.columns:
                            output_df.loc[last_row_index, col] = np.nan
                
                # Krok 4: Zapisz plik wynikowy
                with h5py.File(HDF5_OUTPUT, 'w') as f_out:
                    for col in REQUIRED_OUTPUT_COLUMNS:
                        if col not in output_df.columns:
                             output_df[col] = np.nan
                        data_to_save = output_df[col].to_numpy()
                        f_out.create_dataset(col, data=data_to_save)
                
                logger.info(f"✅ Plik '{HDF5_OUTPUT}' pomyślnie nadpisany. Zawiera {len(output_df)} rekordów.")
                last_processed_count = current_total_count
            
            time.sleep(CHECK_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logger.info("🛑 Zamykanie."); break
        except Exception as e:
            logger.error(f"Błąd krytyczny w agregatorze: {e}", exc_info=True); time.sleep(30)

if __name__ == "__main__":
    run_continuous_updater()
