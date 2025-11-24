# plik: NEURAL__sde_pure_gaussian_process.py
# WERSJA: 189.8 (FINAL STATEFUL CHECKPOINT FIX) - Poprawiono logikę identyfikacji klucza `prev_y` przy wczytywaniu checkpointu.

import logging, time, datetime, os, gc, warnings, json, random
import numpy as np
import pandas as pd
import torch
from collections import deque
from queue import Queue, Empty
from pathlib import Path
from typing import Tuple, Deque, Dict, List, Optional, Any
from sklearn.impute import SimpleImputer

# Importy z projektu
from config_loader import Config
from hdf5_manager import UltraAdvancedHDF5Manager
from binance_websocket_provider import BinanceWebsocketProvider
from learning_engine import PureFunctionalOnlineLearningEngine
from feature_engineering import calculate_price_features
from ai_feature_manager import AIFeatureHDF5Manager
from particle_smoother import ParticleSmoother
from rest_api_fetcher import fetch_historical_data

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s|PREDICTOR|%(levelname)-8s|%(name)-15s|%(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

MODEL_CHECKPOINT_DIR = "./model_checkpoints"
MODEL_STATE_PATH = "./model_checkpoints/sde_predictor_state.pth"
ANTI_INERTIA_STATE = {'consecutive_lazy_count': 0}

# ═══════════════════════════════════════════════════════════════════════════════
# Stan cyklu predykcyjnego
# ═══════════════════════════════════════════════════════════════════════════════
PREDICTION_CYCLE_STATE = {
    'last_arimax_forecast': None,
    'last_final_prediction': None,
    'last_verified_price': None,
    'last_verified_prediction': None,
    'last_raw_sde_forecast': None
}

# ═══════════════════════════════════════════════════════════════════════════════
# FUNKCJA USTAWIAJĄCA ZIARNO LOSOWOŚCI
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed_value: int):
    """
    Ustawia ziarno losowości dla wszystkich kluczowych bibliotek, aby zapewnić
    reprodukowalność eksperymentów i treningu.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    logger.info(f"--- Wczytano wartość ziarna losowego = seed({seed_value}) ---")

# ═══════════════════════════════════════════════════════════════════════════════
# FUNKCJE ZARZĄDZANIA STANEM MODELU
# ═══════════════════════════════════════════════════════════════════════════════

def save_model_state(engine: PureFunctionalOnlineLearningEngine, config: Config):
    """Zapisuje kompletny stan modelu do checkpointu."""
    try:
        os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)
        state_to_save = {
            'model_state_dict': engine.model.state_dict(),
            'scaler': engine.scaler, 
            'main_imputer': engine.main_imputer, 
            'ai_imputer': engine.ai_feature_imputer, 
            'architecture_version': getattr(config, 'architecture_version', '1.0-legacy')
        }
        torch.save(state_to_save, MODEL_STATE_PATH)
        logger.info(f"✅ Pomyślnie zapisano stan predyktora (Architektura: {state_to_save['architecture_version']}).")
    except Exception as e: 
        logger.error(f"❌ Błąd podczas zapisywania stanu predyktora: {e}", exc_info=True)

def load_model_state(engine: PureFunctionalOnlineLearningEngine, config: Config) -> bool:
    """Inteligentnie wczytuje stan modelu z checkpointu."""
    if not Path(MODEL_STATE_PATH).is_file():
        logger.info("📁 Checkpoint nie istnieje. Model będzie wytrenowany od zera.")
        return False
    
    try:
        logger.info("=" * 80)
        logger.info("📥 WCZYTYWANIE CHECKPOINTU MODELU")
        logger.info("=" * 80)
        
        loaded_state = torch.load(MODEL_STATE_PATH, map_location=engine.device, weights_only=False)
        
        saved_version = loaded_state.get('architecture_version', '1.0-legacy')
        current_version = getattr(config, 'architecture_version', '1.0-legacy')
        
        logger.info(f"   ├─ Wersja w checkpoincie: {saved_version}")
        logger.info(f"   ├─ Wersja w konfiguracji: {current_version}")
        
        if saved_version != current_version:
            logger.warning("=" * 80)
            logger.warning(f"❌ NIEZGODNOŚĆ WERSJI ARCHITEKTURY!")
            logger.warning(f"   Checkpoint: '{saved_version}' ≠ Config: '{current_version}'")
            logger.warning(f"   Checkpoint zostanie ODRZUCONY - wymagany pełny retrening.")
            logger.warning("=" * 80)
            return False
        
        logger.info(f"   ├─ ✅ Wersje zgodne: {current_version}")
        
        model_state = loaded_state['model_state_dict']
        
        # <<< KLUCZOWA POPRAWKA: Poprawiono warunek dla `prev_y` >>>
        # Teraz poprawnie identyfikujemy wszystkie stany przejściowe.
        transient_state_keys = [
            k for k in model_state 
            if k.endswith('.x_kalman') or k.endswith('.P_kalman') or k == 'prev_y' or k.endswith('.prev_y')
        ]
        
        if transient_state_keys:
            logger.info("   ├─ ℹ️  Znaleziono przejściowy stan (Kalman/Velocity) w checkpoincie.")
            for key in transient_state_keys:
                # Sprawdźmy na wszelki wypadek, czy klucz na pewno istnieje przed usunięciem
                if key in model_state:
                    del model_state[key]
            logger.info(f"   ├─ ✓ Usunięto {len(transient_state_keys)} kluczy stanu, aby uniknąć konfliktu rozmiaru paczki.")

        # Wczytaj stan modelu, ignorując brakujące klucze (które właśnie usunęliśmy)
        missing_keys, unexpected_keys = engine.model.load_state_dict(model_state, strict=False)
        
        # Logowanie, jeśli jakieś INNE klucze nie pasują (dla celów diagnostycznych)
        critical_missing_keys = [k for k in missing_keys if k not in transient_state_keys]
        if critical_missing_keys:
             logger.warning(f"   ├─ ⚠️  W modelu brakuje niektórych kluczy z checkpointu: {critical_missing_keys}")
        if unexpected_keys:
             logger.warning(f"   ├─ ⚠️  Checkpoint zawiera dodatkowe klucze, których nie ma w modelu: {unexpected_keys}")

        engine.scaler = loaded_state.get('scaler')
        engine.main_imputer = loaded_state.get('main_imputer')
        engine.ai_feature_imputer = loaded_state.get('ai_imputer', SimpleImputer(strategy='constant', fill_value=0.0))
        engine.is_trained = True
        
        logger.info("=" * 80)
        logger.info("✅ CHECKPOINT ZAŁADOWANY POMYŚLNIE")
        logger.info(f"   └─ Architektura: {current_version}")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ KRYTYCZNY BŁĄD podczas wczytywania checkpointu: {e}")
        logger.error("=" * 80)
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# FUNKCJE POMOCNICZE
# ═══════════════════════════════════════════════════════════════════════════════

def get_feature_dataframe(feature_storage: UltraAdvancedHDF5Manager, ai_storage: AIFeatureHDF5Manager, lookback: int, config: Config) -> pd.DataFrame:
    df = feature_storage.get_all_data_as_dataframe().tail(lookback).copy()
    if df.empty: 
        return pd.DataFrame()
    
    ai_data = ai_storage.get_all_data_for_features(lookback_minutes=lookback)
    if ai_data and 'timestamps' in ai_data: 
        df = pd.merge(df, pd.DataFrame(ai_data), on='timestamps', how='left')
    
    for col in config.model_feature_columns:
        if col not in df.columns: 
            df[col] = 0.0
    
    return df[list(dict.fromkeys([c for c in ['timestamps', 'prices', 'volumes', 'errors'] if c in df.columns] + config.model_feature_columns))].fillna(0.0)

def apply_anti_inertia(prediction: float, current_price: float, config: Config, state: dict) -> Tuple[float, bool]:
    cfg = getattr(config, 'anti_inertia_module', {})
    if not cfg.get('enabled', False): 
        return prediction, False
    
    threshold = cfg.get('trigger_factor_threshold', 1.0)
    consecutive_req = cfg.get('trigger_consecutive_moves', 2)
    gain = cfg.get('anti_inertia_gain', 0.15)
    
    change_percent = (abs(prediction - current_price) / current_price * 100) if current_price != 0 else 0.0
    
    if change_percent < threshold: 
        state['consecutive_lazy_count'] += 1
    else: 
        state['consecutive_lazy_count'] = 0
    
    if state['consecutive_lazy_count'] >= consecutive_req:
        return prediction + ((prediction - current_price) * gain), True
    
    return prediction, False

# ═══════════════════════════════════════════════════════════════════════════════
# 🎓 ONLINE LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

def perform_online_learning(engine, feature_storage, ai_storage, config, reason: str) -> bool:
    """Wykonuje dotrenowanie online modelu."""
    logger.info("=" * 80)
    logger.info("🎓 ROZPOCZĘCIE DOTRENOWANIA ONLINE")
    logger.info("=" * 80)
    logger.info(f"   ├─ Powód: {reason}")
    logger.info(f"   ├─ Learning rate: {config.online_learning_fine_tune_lr}")
    logger.info(f"   ├─ Liczba epok: {config.online_learning_fine_tune_epochs}")
    
    try:
        history_minutes = config.online_learning_history_minutes
        max_samples = config.online_learning_max_samples
        
        logger.info(f"   ├─ Pobieranie danych: ostatnie {history_minutes} minut")
        
        train_df = get_feature_dataframe(feature_storage, ai_storage, history_minutes, config)
        
        if train_df.empty:
            logger.error("   └─ ❌ Brak danych do dotrenowania!")
            return False
        
        original_count = len(train_df)
        if max_samples and len(train_df) > max_samples:
            train_df = train_df.tail(max_samples)
            logger.warning(f"   ├─ ⚠️  Ograniczono dane: {original_count} → {max_samples} próbek (ochrona RAM)")
        else:
            logger.info(f"   ├─ ✅ Użyto {len(train_df)} próbek (limit {max_samples} nie aktywny)")
        
        logger.info(f"   ├─ Rozpoczynam fine-tuning na {len(train_df)} próbkach...")
        
        start_time = time.time()
        
        final_loss = engine.run_training(
            train_df, 
            epochs=config.online_learning_fine_tune_epochs, 
            lr=config.online_learning_fine_tune_lr, 
            is_initial=False
        )
        
        elapsed_time = time.time() - start_time
        
        logger.info(f"   ├─ ✅ Trening zakończony w {elapsed_time:.2f}s")
        if final_loss is not None:
            logger.info(f"   ├─ Końcowa strata: {final_loss:.6f}")
        
        if config.online_learning_update_checkpoint:
            logger.info(f"   ├─ Zapisywanie zaktualizowanego checkpointu...")
            save_model_state(engine, config)
            logger.info(f"   └─ ✅ Checkpoint zaktualizowany")
        else:
            logger.info(f"   └─ ⏭️  Checkpoint NIE został zapisany (disabled w config)")
        
        logger.info("=" * 80)
        logger.info("✅ DOTRENOWANIE ONLINE ZAKOŃCZONE POMYŚLNIE")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ BŁĄD podczas dotrenowania online: {e}")
        logger.error("=" * 80)
        logger.debug("Szczegóły błędu:", exc_info=True)
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# FUNKCJE CYKLU PREDYKCYJNEGO
# ═══════════════════════════════════════════════════════════════════════════════

def run_prediction_cycle(engine, feature_storage, ai_storage, config, prediction_buffer, particle_smoother, smoothed_predicted_error, cycle_state):
    history_len = max(config.prediction_input_history_minutes, config.online_training_history_minutes, 2)
    X_inference_df = get_feature_dataframe(feature_storage, ai_storage, history_len, config)
    
    if X_inference_df.empty or len(X_inference_df) < 2: 
        return particle_smoother, smoothed_predicted_error
    
    current_price = X_inference_df['prices'].iloc[-1]

    # ═══════════════════════════════════════════════════════════════════════════
    # 🔧 PROGNOZA SDE - OBSŁUGA LOG_RETURNS
    # ═══════════════════════════════════════════════════════════════════════════
    predicted_target = engine.predict(X_inference_df.tail(config.prediction_input_history_minutes))
    
    if config.model_target_feature == 'log_returns':
        raw_sde_price_forecast = current_price * np.exp(predicted_target)
        predicted_delta = raw_sde_price_forecast - current_price
    elif config.model_target_feature == 'price_delta':
        predicted_delta = predicted_target
        raw_sde_price_forecast = current_price + predicted_delta
    else:
        raw_sde_price_forecast = predicted_target
        predicted_delta = raw_sde_price_forecast - current_price
    # ═══════════════════════════════════════════════════════════════════════════

    smoothed_pred = raw_sde_price_forecast
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🧬 FILTR CZĄSTECZKOWY
    # ═══════════════════════════════════════════════════════════════════════════
    if particle_smoother:
        particle_smoother.rebase(current_price)
        particle_smoother.predict(predicted_delta)
        particle_smoother.update(raw_sde_price_forecast)
        
        ess_before = particle_smoother.get_effective_sample_size()
        ess_ratio = (ess_before / config.particle_smoother_settings['num_particles']) * 100
        
        logger.info("=" * 80)
        logger.info("🧬 FILTR CZĄSTECZKOWY - DIAGNOSTYKA")
        logger.info("=" * 80)
        logger.info(f"   ├─ Całkowita liczba cząsteczek: {config.particle_smoother_settings['num_particles']:,}")
        logger.info(f"   ├─ Efektywne cząsteczki (ESS): {ess_before:,.0f}")
        logger.info(f"   ├─ ESS Ratio: {ess_ratio:.1f}% {'✅ DOBRY STAN' if ess_ratio > 50 else '⚠️  NISKA EFEKTYWNOŚĆ'}")
        logger.info(f"   ├─ Próg resampingu: {config.particle_smoother_settings['num_particles'] / 2.0:,.0f}")
        
        resampled = particle_smoother.resample_if_needed()
        
        if resampled:
            ess_after = particle_smoother.get_effective_sample_size()
            logger.info(f"   ├─ 🔄 RESAMPLING WYKONANY! ESS: {ess_before:,.0f} → {ess_after:,.0f}")
        else:
            logger.info(f"   ├─ ✓ Resampling NIE był potrzebny")
        
        smoothed_pred = particle_smoother.get_smoothed_prediction()
        smoothing_delta = smoothed_pred - raw_sde_price_forecast
        smoothing_impact_percent = (smoothing_delta / current_price * 100) if current_price != 0 else 0
        
        stats = particle_smoother.get_particle_stats()
        
        logger.info(f"   │")
        logger.info(f"   ├─ 📊 WYNIKI WYGŁADZANIA:")
        logger.info(f"   ├─── Surowa prognoza SDE: {raw_sde_price_forecast:.2f} USDT")
        logger.info(f"   ├─── Wygładzona prognoza: {smoothed_pred:.2f} USDT")
        logger.info(f"   ├─── Delta wygładzenia: {smoothing_delta:+.2f} USDT ({smoothing_impact_percent:+.3f}%)")
        logger.info(f"   │")
        logger.info(f"   ├─ 📈 STATYSTYKI CHMURY:")
        logger.info(f"   ├─── Rozrzut (σ): {stats['std']:.2f} USDT")
        logger.info(f"   ├─── Zakres: [{stats['min']:.2f}, {stats['max']:.2f}] USDT")
        logger.info("=" * 80)
        
    raw_predicted_error = 0.0
    if STATSMODELS_AVAILABLE and 'errors' in X_inference_df.columns:
        errors = X_inference_df['errors'].dropna()
        if len(errors) >= 2:
            try:
                cfg_err = config.error_correction_module
                raw_predicted_error = ARIMA(errors, order=(cfg_err['arimax_order_p'], cfg_err['arimax_order_d'], cfg_err['arimax_order_q'])).fit().forecast(steps=1).iloc[-1]
            except Exception: 
                pass
        
    smoothing = config.error_correction_module.get('error_smoothing_factor', 0.1)
    smoothed_predicted_error = (smoothing * raw_predicted_error) + ((1 - smoothing) * smoothed_predicted_error)
    arimax_corrected_forecast = smoothed_pred - smoothed_predicted_error
    
    # ═══════════════════════════════════════════════════════════════════════════
    # AUTOREFLEKSJA
    # ═══════════════════════════════════════════════════════════════════════════
    signal_state = 'TRUST_ARIMAX'
    rebellion_strength = 1.0
    
    try:
        with open('./pure_sde_data/reflection_state.json', 'r') as f:
            reflection_data = json.load(f)
            signal_state = reflection_data.get('signal_state', 'TRUST_ARIMAX')
            rebellion_strength = reflection_data.get('rebellion_strength', 1.0)
    except (FileNotFoundError, json.JSONDecodeError): 
        pass

    arimax_now = arimax_corrected_forecast
    pre_anti_inertia_prediction = 0.0
    
    if signal_state == 'TRUST_ARIMAX':
        pre_anti_inertia_prediction = arimax_now
        cycle_state['last_final_prediction'] = arimax_now
        logger.info(f"   ✅ Tryb Autorefleksji: ZAUFANIE (używam prognozy ARIMAX)")
    else:
        arimax_prev = cycle_state['last_arimax_forecast']
        final_prev = cycle_state['last_final_prediction']
        
        if arimax_prev is not None and final_prev is not None:
            arimax_move = arimax_now - arimax_prev
            contrarian_move = -arimax_move * rebellion_strength
            pre_anti_inertia_prediction = final_prev + contrarian_move
            
            logger.info(f"   🛡️  BUNT AUTOREFLEKSYJNY AKTYWNY:")
            logger.info(f"       ├─ Ruch ARIMAX:        {arimax_move:+.2f} USDT")
            logger.info(f"       ├─ Siła odwracania:    {rebellion_strength:.1%}")
            logger.info(f"       └─ Cel:                {pre_anti_inertia_prediction:.2f} USDT")
        else:
            pre_anti_inertia_prediction = arimax_now
    
    final_prediction, anti_inertia_activated = apply_anti_inertia(pre_anti_inertia_prediction, current_price, config, ANTI_INERTIA_STATE)
    
    cycle_state['last_arimax_forecast'] = arimax_now
    cycle_state['last_final_prediction'] = final_prediction

    target_time_dt = datetime.datetime.fromtimestamp(X_inference_df['timestamps'].iloc[-1] / 1000.0) + datetime.timedelta(minutes=config.prediction_horizon_minutes)
    
    particle_correction = smoothed_pred - raw_sde_price_forecast
    arimax_correction = -smoothed_predicted_error
    total_correction = arimax_corrected_forecast - raw_sde_price_forecast
    
    logger.info("─" * 80)
    logger.info("🔮 ŁAŃCUCH TRANSFORMACJI PROGNOZY:")
    logger.info(f"   ┌─ [1] Surowa SDE:                 {raw_sde_price_forecast:>10.2f} USDT")
    logger.info(f"   ├─ [2] Wygładzona (Particle):     {smoothed_pred:>10.2f} USDT (korekta: {particle_correction:+.2f})")
    logger.info(f"   ├─ [3] ARIMAX (korekta błędu):    {arimax_corrected_forecast:>10.2f} USDT (korekta: {arimax_correction:+.2f})")
    logger.info(f"   └─ [4] Finalna (Autorefleksja):   {final_prediction:>10.2f} USDT")
    logger.info("─" * 80)
    logger.info("📊 SZCZEGÓŁY KOREKT:")
    logger.info(f"   ├─ Korekta Particle Filter:    {particle_correction:+10.2f} USDT ({(particle_correction/current_price*100):+.4f}%)")
    logger.info(f"   ├─ Korekta ARIMAX (błąd):      {arimax_correction:+10.2f} USDT (surowy: {-raw_predicted_error:+.2f})")
    logger.info(f"   ├─ Łączna korekta:             {total_correction:+10.2f} USDT ({(total_correction/current_price*100):+.4f}%)")
    logger.info(f"   └─ Delta SDE:                  {predicted_delta:+10.2f} USDT ({(predicted_delta/current_price*100):+.3f}%)")
    logger.info("─" * 80)
    
    if anti_inertia_activated:
        logger.info(f"   🔥 Anti-Inertia aktywna: {pre_anti_inertia_prediction:.2f} → {final_prediction:.2f}")
    
    logger.info("─" * 80)
    logger.info(f"🏁 OSTATECZNA PROGNOZA NA {target_time_dt:%H:%M:%S}: {final_prediction:.2f} USDT")
    logger.info("=" * 80)

    feature_storage.save_multiple_features(int(target_time_dt.timestamp() * 1000), {
        'raw_forecast': raw_sde_price_forecast, 
        'smoothed_forecast': smoothed_pred, 
        'error_forecast': smoothed_predicted_error,
        'arimax_corrected_forecast': arimax_corrected_forecast, 
        'final_predictions': final_prediction
    })
    
    prediction_buffer.append({
        'target_time_ts': int(target_time_dt.timestamp() * 1000), 
        'final_prediction': final_prediction,
        'raw_sde_forecast': raw_sde_price_forecast,
        'arimax_forecast': arimax_corrected_forecast,
        'smoothed_forecast': smoothed_pred
    })
    
    return particle_smoother, smoothed_predicted_error

def rebuild_feature_file(raw_storage, feature_storage, config):
    if feature_storage.get_record_count() > 0 and not config.clean_hdf5_on_start: 
        return
    
    logger.info("Uruchamianie procedury przebudowy pliku cech...")
    df_raw = raw_storage.get_all_data_as_dataframe()
    
    if df_raw.empty: 
        logger.error("Brak surowych danych do przebudowy.")
        return
    
    features_df = calculate_price_features(df_raw['prices'], df_raw['volumes'], config)
    feature_storage.save_full_dataframe(pd.concat([df_raw, features_df], axis=1).dropna(subset=['price_delta']))
    logger.info("✅ Przebudowa pliku cech zakończona.")

def startup_procedure(engine, raw_storage, feature_storage, ai_storage, config):
    logger.info("=" * 50 + "\n--- PROCEDURA STARTOWA ---")
    
    if not load_model_state(engine, config):
        logger.info("Model niewytrenowany/niekompatybilny. Rozpoczynam trening.")
        required_data = config.max_batches_per_epoch * config.training_batch_size + config.prediction_input_history_minutes + 200
        
        if raw_storage.get_record_count() < required_data:
            logger.warning(f"Brak danych surowych. Pobieranie {required_data} świec...")
            raw_data = fetch_historical_data(config.data_management_symbol, config.data_management_interval, required_data)
            if raw_data: 
                raw_storage.save_full_dataframe(pd.DataFrame(raw_data).rename(columns={'price': 'prices', 'volume': 'volumes', 'timestamp': 'timestamps'}))
        
        rebuild_feature_file(raw_storage, feature_storage, config)
        
        total_available = feature_storage.get_record_count()
        max_samples = getattr(config, 'training_max_samples', None)
        
        if max_samples is not None and max_samples > 0:
            samples_to_use = min(max_samples, total_available)
        else:
            samples_to_use = total_available
        
        train_df = get_feature_dataframe(feature_storage, ai_storage, samples_to_use, config)
        
        if train_df.empty: 
            raise SystemExit("Brak danych do treningu.")
        
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            logger.info(f"💾 Pamięć RAM przed treningiem: {mem_before:.1f} MB")
        
        logger.info(f"🎓 Rozpoczynam trening początkowy na {len(train_df)} próbkach...")
        engine.run_training(train_df, epochs=config.training_initial_epochs, lr=config.training_initial_lr, is_initial=True)
        
        if PSUTIL_AVAILABLE:
            mem_after = process.memory_info().rss / 1024 / 1024
            logger.info(f"💾 Pamięć RAM po treningu: {mem_after:.1f} MB (wzrost: {mem_after - mem_before:.1f} MB)")
        
        save_model_state(engine, config)

def main():
    # ═════════════════════════════════════════════════════════════════════════
    # === POPRAWIONA, UPROSZCZONA LOGIKA USTAWIANIA ZIARNA LOSOWOŚCI ===
    # ═════════════════════════════════════════════════════════════════════════
    try:
######  config = Config()
        config = Config(config_path='config.json')

        # Sprawdź, czy w konfiguracji zdefiniowano ziarno
        seed_value = getattr(config, 'random_seed', None)
        if seed_value is not None:
            set_seed(int(seed_value))
        else:
            logger.info("🌱 Ziarno losowości (seed) nie zostało zdefiniowane w config.json. Używam losowego ziarna.")

        logging.getLogger().setLevel(config.system_log_level.upper())

    except Exception as e:
        logger.critical(f"Błąd wczytywania konfiguracji lub ustawiania ziarna: {e}", exc_info=True)
        return
    # ═════════════════════════════════════════════════════════════════════════
    
    logger.info(f"--- 🚀 URUCHAMIANIE PREDYKTORA v{config.architecture_version} ---")
    
    raw_storage = UltraAdvancedHDF5Manager(config, "raw_market_data.h5")
    feature_storage = UltraAdvancedHDF5Manager(config, "sde_data.h5")
    ai_storage = AIFeatureHDF5Manager(config, [c for c in config.model_feature_columns if c.startswith('ai_')])
    engine = PureFunctionalOnlineLearningEngine(config)
    
    logger.info(f"🧠 Wybrano model do trenowania: {engine.model.__class__.__name__}")
    
    provider = None
    
    try:
        startup_procedure(engine, raw_storage, feature_storage, ai_storage, config)
        
        df_feat = get_feature_dataframe(feature_storage, ai_storage, 1, config)
        last_processed_ts = df_feat['timestamps'].iloc[-1] if not df_feat.empty else 0
        prediction_buffer = deque(maxlen=config.prediction_input_history_minutes)
        smoothed_predicted_error = 0.0
        
        particle_smoother: Optional[ParticleSmoother] = None
        
        if config.particle_smoother_settings.get('enabled', True):
            current_price = df_feat['prices'].iloc[-1] if not df_feat.empty else 0.0
            particle_smoother = ParticleSmoother(
                config.particle_smoother_settings,
                engine.device,
                initial_value=current_price
            )
            logger.info(f"✅ Particle Smoother zainicjalizowany z ceną bazową: {current_price:.2f} USDT")
        else:
            logger.warning("⚠️  Particle Smoother WYŁĄCZONY w konfiguracji")
        
        candle_queue = Queue()
        provider = BinanceWebsocketProvider({'symbol': config.data_management_symbol, 'interval': config.data_management_interval}, candle_queue.put)
        provider.start()
        
        logger.info("--- ROZPOCZYNANIE GŁÓWNEJ PĘTLI ---")
        
        while True:
            try:
                candle = candle_queue.get(timeout=65.0)
                if candle['timestamp'] <= last_processed_ts: 
                    continue
                
                current_time_ts = candle['timestamp']
                
                logger.info("\n\n\n" + "=" * 80)
                logger.info("--- Nowa świeca: %s ---", datetime.datetime.fromtimestamp(current_time_ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S'))
                logger.info("=" * 80)
                
                raw_storage.append_candle_data({'timestamps': current_time_ts, 'prices': candle['price'], 'volumes': candle['volume']})
                all_features_df = calculate_price_features(pd.Series(raw_storage.get_column_data('prices')), pd.Series(raw_storage.get_column_data('volumes')), config)
                latest_features = all_features_df.iloc[-1:].to_dict('records')[0]
                feature_storage.update_or_append_row(current_time_ts, {'prices': candle['price'], 'volumes': candle['volume'], **latest_features})
                
                # ═══════════════════════════════════════════════════════════════
                # WERYFIKACJA PROGNOZY
                # ═══════════════════════════════════════════════════════════════
                matched_pred = next((p for p in list(prediction_buffer) if p['target_time_ts'] == current_time_ts), None)
                
                if matched_pred:
                    prediction_buffer.remove(matched_pred)
                    
                    base_error = matched_pred['raw_sde_forecast'] - candle['price']
                    base_error_percent = abs(base_error / candle['price'] * 100)
                    
                    arimax_error = matched_pred['arimax_forecast'] - candle['price']
                    arimax_error_percent = abs(arimax_error / candle['price'] * 100)
                    
                    final_error = matched_pred['final_prediction'] - candle['price']
                    final_error_percent = abs(final_error / candle['price'] * 100)
                    
                    feature_storage.save_single_feature(current_time_ts, 'errors', arimax_error)
                    
                    # WERYFIKACJA KIERUNKU
                    direction_status = ""
                    direction_correct = None
                    
                    if PREDICTION_CYCLE_STATE['last_verified_price'] is not None and PREDICTION_CYCLE_STATE['last_verified_prediction'] is not None:
                        prev_price = PREDICTION_CYCLE_STATE['last_verified_price']
                        prev_base_prediction = PREDICTION_CYCLE_STATE['last_verified_prediction']
                        
                        predicted_up = matched_pred['raw_sde_forecast'] > prev_base_prediction
                        actual_up = candle['price'] > prev_price
                        
                        direction_correct = bool(predicted_up == actual_up)
                        direction_status = "✅ TRAFNY" if direction_correct else "❌ BŁĘDNY"
                    else:
                        direction_status = "⏸️  PIERWSZA WERYFIKACJA"
                    
                    logger.info("─" * 80)
                    logger.info(f"📈 WERYFIKACJA PROGNOZY:")
                    logger.info(f"   ├─ Rzeczywista cena:       {candle['price']:>10.2f} USDT")
                    logger.info(f"   ├─ Prognoza SDE:           {matched_pred['raw_sde_forecast']:>10.2f} USDT (błąd: {base_error:+.2f} / {base_error_percent:.3f}%)")
                    logger.info(f"   ├─ Kierunek:               {direction_status}")
                    logger.info(f"   ├─ Prognoza ARIMAX:        {matched_pred['arimax_forecast']:>10.2f} USDT (błąd: {arimax_error:+.2f} / {arimax_error_percent:.3f}%)")
                    logger.info(f"   └─ Prognoza FINALNA:       {matched_pred['final_prediction']:>10.2f} USDT (błąd: {final_error:+.2f} / {final_error_percent:.3f}%)")
                    logger.info("─" * 80)
                    
                    # ═══════════════════════════════════════════════════════════════
                    # ONLINE LEARNING (UŻYWA PŁASKICH ATRYBUTÓW)
                    # ═══════════════════════════════════════════════════════════════
                    should_retrain = False
                    retrain_reasons = []
                    
                    if config.online_learning_enabled:
                        if config.online_learning_retrain_on_error and base_error_percent > config.online_learning_error_threshold_percent:
                            should_retrain = True
                            retrain_reasons.append(f"Błąd bazowy {base_error_percent:.3f}% > próg {config.online_learning_error_threshold_percent}%")
                        
                        if config.online_learning_retrain_on_wrong_direction and (direction_correct is not None and not direction_correct):
                            should_retrain = True
                            retrain_reasons.append("Błędny kierunek")
                        
                        if should_retrain:
                            logger.warning("=" * 80)
                            logger.warning("⚠️  WARUNKI DOTRENOWANIA SPEŁNIONE (MODEL BAZOWY):")
                            for idx, reason in enumerate(retrain_reasons, 1):
                                logger.warning(f"   {idx}. {reason}")
                            logger.warning("=" * 80)
                            
                            perform_online_learning(
                                engine, 
                                feature_storage, 
                                ai_storage, 
                                config, 
                                reason=" + ".join(retrain_reasons)
                            )
                        else:
                            logger.info("ℹ️  Online learning: warunki NIE spełnione (brak dotrenowania)")
                    
                    PREDICTION_CYCLE_STATE['last_verified_price'] = candle['price']
                    PREDICTION_CYCLE_STATE['last_verified_prediction'] = matched_pred['raw_sde_forecast']

                particle_smoother, smoothed_predicted_error = run_prediction_cycle(
                    engine, feature_storage, ai_storage, config, prediction_buffer, 
                    particle_smoother, smoothed_predicted_error, PREDICTION_CYCLE_STATE
                )
                
                last_processed_ts = current_time_ts
                
            except Empty: 
                logger.warning("Timeout WebSocket. Sprawdzanie połączenia...")
                continue
                
    except (SystemExit, KeyboardInterrupt) as e:
        logger.info(f"Proces zatrzymany: {e}")
    except Exception as e:
        logger.critical("Błąd krytyczny w pętli głównej.", exc_info=True)
    finally:
        if provider and provider.is_running():
            provider.stop()
        logger.info("--- PREDYKTOR ZAKOŃCZYŁ PRACĘ ---")

if __name__ == "__main__":
    main()
