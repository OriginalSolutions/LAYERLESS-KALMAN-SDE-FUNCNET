# plik: hdf5_manager.py
# WERSJA: 7.2 (FINALNA) - Rozbudowany o brakujące, kluczowe metody.

import h5py
import numpy as np
import logging
import datetime
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

class UltraAdvancedHDF5Manager:
    def __init__(self, config: Any, filename: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.filename = f"{config.system_data_directory}/{filename}"
        self.max_retries = getattr(config, 'hdf5_max_retries', 100)
        self.retry_delay = getattr(config, 'hdf5_retry_delay', 0.1)
        
        db_path = Path(self.filename)
        if not db_path.exists() or (getattr(config, 'clean_hdf5_on_start', False) and "features" in filename):
            if db_path.exists() and getattr(config, 'clean_hdf5_on_start', False):
                self.logger.warning(f"Opcja 'clean_hdf5_on_start' jest włączona. Usuwanie i tworzenie nowego pliku: {self.filename}")
                db_path.unlink()
            else:
                self.logger.info(f"🛠️ Tworzenie nowego, pustego pliku bazy danych: {self.filename}")
            
            with h5py.File(self.filename, 'w') as f:
                f.attrs['version'] = getattr(self.config, 'system_version', 'N/A')
                f.attrs['creation_date'] = str(datetime.datetime.now())
        else:
            self.logger.info(f"✅ Otwieram istniejącą bazę danych: {self.filename}")
    
    def _execute_with_retry(self, operation, operation_name: str, mode: str = 'a'):
        for attempt in range(self.max_retries):
            try:
                with h5py.File(self.filename, mode) as f:
                    return operation(f)
            except (OSError, BlockingIOError) as e:
                if 'errno' in dir(e) and e.errno == 11 and attempt < self.max_retries - 1:
                    self.logger.warning(f"Plik HDF5 zablokowany ({operation_name}). Próba {attempt+2}/{self.max_retries}...")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    self.logger.error(f"❌ OSTATECZNA PORAŻKA: '{operation_name}' po {self.max_retries} próbach. Błąd: {e}", exc_info=True)
                    return None
            except Exception as e:
                self.logger.error(f"❌ Nieoczekiwany błąd podczas '{operation_name}': {e}", exc_info=True)
                return None
        return None

    def get_record_count(self) -> int:
        def op(f):
            return f['timestamps'].shape[0] if 'timestamps' in f else 0
        return self._execute_with_retry(op, "pobranie liczby rekordów") or 0

    def _ensure_and_get_index(self, f: h5py.File, timestamp: int) -> int:
        if 'timestamps' not in f:
            f.create_dataset('timestamps', (1,), maxshape=(None,), dtype='int64', data=[timestamp])
            return 0

        timestamps_arr = f['timestamps'][:]
        indices = np.where(timestamps_arr == timestamp)[0]

        if len(indices) > 0:
            return indices[0]
        else:
            # Zakładamy, że nowe dane są zawsze na końcu
            n = f['timestamps'].shape[0]
            f['timestamps'].resize((n + 1,))
            f['timestamps'][n] = timestamp

            for dset_name in f.keys():
                if dset_name != 'timestamps':
                    dset = f[dset_name]
                    if dset.shape[0] < n + 1:
                        dset.resize((n + 1,))
                        if 'float' in str(dset.dtype):
                            dset[n] = np.nan
            return n

    def get_all_data_as_dataframe(self) -> pd.DataFrame:
        def op(f):
            if 'timestamps' not in f: return pd.DataFrame()
            n_timestamps = len(f['timestamps'])
            all_data_dict = {}
            for key in f.keys():
                if len(f[key]) == n_timestamps:
                    all_data_dict[key] = f[key][:]
                else:
                    self.logger.warning(f"Niespójna długość kolumny '{key}' w {self.filename}. Ignoruję ją.")
            return pd.DataFrame(all_data_dict)
        
        result = self._execute_with_retry(op, "wczytanie całego DataFrame'u")
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame()

    # ##########################################################################
    # ### NOWA METODA 1: Wczytywanie pojedynczej kolumny ###
    # ##########################################################################
    def get_column_data(self, column_name: str) -> Optional[np.ndarray]:
        def op(f):
            return f[column_name][:] if column_name in f else None
        return self._execute_with_retry(op, f"pobranie kolumny '{column_name}'")

    def save_full_dataframe(self, df: pd.DataFrame):
        def op(f):
            for column in df.columns:
                if column in f: del f[column]
                data = df[column].to_numpy()
                dtype = 'int64' if column == 'timestamps' else 'float64'
                f.create_dataset(column, data=data, maxshape=(None,), dtype=dtype, compression='gzip')
            self.logger.info(f"✅ Zapis blokowy {df.shape[0]} wierszy do {self.filename} zakończony.")
        
        self._execute_with_retry(op, "zapis blokowy DataFrame", mode='w')
    
    def append_candle_data(self, candle: dict):
        def op(f):
            ts = candle.get('timestamps')
            if ts is None: return
            idx = self._ensure_and_get_index(f, ts)

            for key, value in candle.items():
                if key == 'timestamps': continue
                if key not in f:
                    dtype = 'float64'
                    f.create_dataset(key, (len(f['timestamps']),), maxshape=(None,), dtype=dtype, fillvalue=np.nan)

                dset = f[key]
                if dset.shape[0] < len(f['timestamps']):
                     dset.resize((len(f['timestamps']),))
                dset[idx] = value
        
        self._execute_with_retry(op, "dopisywanie świecy")

    def save_single_feature(self, timestamp: int, feature_name: str, value: Any):
        def op(f):
            idx = self._ensure_and_get_index(f, timestamp)
            if feature_name not in f:
                dtype = 'int8' if isinstance(value, int) else 'float64'
                fill_value = 0 if dtype == 'int8' else np.nan
                f.create_dataset(feature_name, (len(f['timestamps']),), maxshape=(None,), dtype=dtype, fillvalue=fill_value)
            
            dset = f[feature_name]
            if dset.shape[0] <= idx:
                dset.resize((idx + 1,))
            
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                dset[idx] = value
        
        self._execute_with_retry(op, f"zapis pojedynczej cechy '{feature_name}'")

    # ##########################################################################
    # ### NOWA METODA 2: Bezpieczne dopisywanie wiersza z cechami ###
    # ##########################################################################
    def update_or_append_row(self, timestamp: int, features_dict: dict):
        def op(f):
            idx = self._ensure_and_get_index(f, timestamp)
            for column_name, value in features_dict.items():
                if column_name == 'timestamps': continue
                if column_name not in f:
                    dtype = np.float64 # Załóż, że wszystkie cechy są float
                    f.create_dataset(column_name, (len(f['timestamps']),), maxshape=(None,), dtype=dtype, fillvalue=np.nan)
                
                dset = f[column_name]
                if dset.shape[0] < len(f['timestamps']):
                    dset.resize((len(f['timestamps']),))
                dset[idx] = value
        
        self._execute_with_retry(op, "aktualizacja/dopisanie wiersza cech")

    # ##########################################################################
    # ### NOWA METODA 3: Zapis wielu prognoz dla jednego, przyszłego czasu ###
    # ##########################################################################
    def save_multiple_features(self, timestamp: int, features_dict: dict):
        def op(f):
            idx = self._ensure_and_get_index(f, timestamp)
            for column_name, value in features_dict.items():
                if column_name not in f:
                    f.create_dataset(column_name, (len(f['timestamps']),), maxshape=(None,), dtype=np.float64, fillvalue=np.nan)
                
                dset = f[column_name]
                if dset.shape[0] < len(f['timestamps']):
                    dset.resize((len(f['timestamps']),))
                dset[idx] = value
        
        self._execute_with_retry(op, "zapis wielu prognoz")
