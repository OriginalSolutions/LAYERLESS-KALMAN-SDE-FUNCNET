# file: read_h5_analyzer.py
# VERSION: FINAL - Correctly displays datetime index.

import h5py
import pandas as pd
import sys
import logging

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Configure which file to read by uncommenting the desired line



#  HDF5_FILE_PATH = 'sde_data.h5'
#  HDF5_FILE_PATH = 'base_accuracy_metrics.h5'

#  HDF5_FILE_PATH = 'final_forecast.h5'

#  HDF5_FILE_PATH = 'raw_market_data.h5'
HDF5_FILE_PATH = 'VISUALIZATION_DATA.hdf5'

#  HDF5_FILE_PATH = 'accuracy_metrics.h5'

# HDF5_FILE_PATH = 'ai_features.h5'



try:
    ROWS_TO_DISPLAY = int(sys.argv[1]) if len(sys.argv) > 1 else 30
except (ValueError, IndexError):
    ROWS_TO_DISPLAY = 30

pd.set_option('display.precision', 6)
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', None)

logging.basicConfig(level=logging.INFO, format='%(levelname)-8s| %(message)s')

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
def main():
    try:
        with h5py.File(HDF5_FILE_PATH, 'r') as f:
            print("="*120)
            print(f"Analyzing HDF5 file: '{HDF5_FILE_PATH}'")
            print("\nFile Structure:")
            for key, item in f.items():
                if isinstance(item, h5py.Dataset):
                    print(f"- Dataset: {key}, Shape: {item.shape}, Dtype: {item.dtype}")
            print("="*120)

            all_data = {key: f[key][:] for key in f.keys()}
            if not all_data:
                print("No datasets found in the file.")
                return
            
            lengths = {key: len(value) for key, value in all_data.items()}
            if len(set(lengths.values())) > 1:
                print("\n!!! WARNING: Inconsistent column lengths detected !!!")
                min_len = min(lengths.values())
                all_data = {key: value[:min_len] for key, value in all_data.items()}

            df = pd.DataFrame(all_data)

            # --- OSTATECZNA, POPRAWIONA LOGIKA CZASU ---
            time_col_name = None
            for name in ['Timestamp', 'timestamps', 'Time', 'Czas']:
                if name in df.columns:
                    time_col_name = name
                    break
            
            if time_col_name:
                print(f"\nDetected time column '{time_col_name}'. Converting to readable UTC datetime...")
                df['UTC_Time'] = pd.to_datetime(df[time_col_name], unit='ms', utc=True)
                df.set_index('UTC_Time', inplace=True)
                # KROK KLUCZOWY: Usuń starą, numeryczną kolumnę czasu
                df.drop(columns=[time_col_name], inplace=True)
            # --- KONIEC POPRAWKI ---

            print(f"\nLast {ROWS_TO_DISPLAY} records (time in UTC):")
            print("-" * 120)
            print(df.tail(ROWS_TO_DISPLAY).to_string())
            print("-" * 120)

    except FileNotFoundError:
        print(f"\nERROR: File '{HDF5_FILE_PATH}' not found.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
