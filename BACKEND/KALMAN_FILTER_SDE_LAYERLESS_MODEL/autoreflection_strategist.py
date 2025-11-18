# plik: autoreflection_strategist.py
# WERSJA: 3.0 (FINALNA - Obsługa wszystkich trybów + ochrona przed accuracy=0)

import pandas as pd
import numpy as np
import h5py
import time
import commentjson as json
import logging
import os
from datetime import datetime, timedelta

# ==============================================================================
# KONFIGURACJA
# ==============================================================================
ACCURACY_HDF5_PATH = './pure_sde_data/accuracy_metrics.h5'
REFLECTION_JSON_PATH = './pure_sde_data/reflection_state.json'
AUTOREFLECTION_CONFIG_PATH = 'autoreflection_config.json'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|STRATEGIST|%(levelname)-8s|%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AutoreflectionStrategist")

# ==============================================================================
# GLOBALNE ZMIENNE STANU
# ==============================================================================
EMERGENCY_STATE = {
    'consecutive_zero_count': 0,
    'is_shutdown': False,
    'last_valid_signal': 'TRUST_ARIMAX'
}

# ==============================================================================
# FUNKCJE POMOCNICZE
# ==============================================================================

def load_accuracy_data(config, analysis_mode):
    """Wczytuje dane accuracy z HDF5 z obsługą filtrowania zer."""
    try:
        accuracy_source = config.get(f"{analysis_mode.lower()}_settings", {}).get("accuracy_source", "Accuracy_ARIMAX")
        
        with h5py.File(ACCURACY_HDF5_PATH, 'r') as f_acc:
            if accuracy_source not in f_acc:
                logger.warning(f"Źródło '{accuracy_source}' nie istnieje w pliku HDF5")
                return None
            
            acc_hist = pd.Series(f_acc[accuracy_source][:])
        
        # ═══════════════════════════════════════════════════════════════
        # 🔧 OCHRONA PRZED ACCURACY = 0
        # ═══════════════════════════════════════════════════════════════
        ignore_zero = config.get(f"{analysis_mode.lower()}_settings", {}).get("ignore_zero_accuracy", True)
        
        original_count = len(acc_hist)
        
        if ignore_zero:
            acc_hist = acc_hist[acc_hist != 0.0]
            filtered_count = len(acc_hist)
            
            if filtered_count < original_count:
                zero_count = original_count - filtered_count
                logger.info(f"🔧 Filtr accuracy=0: Usunięto {zero_count}/{original_count} próbek ({zero_count/original_count*100:.1f}%)")
                
                # Aktualizuj licznik zerowych próbek
                EMERGENCY_STATE['consecutive_zero_count'] = zero_count if zero_count == original_count else 0
            else:
                EMERGENCY_STATE['consecutive_zero_count'] = 0
        
        acc_hist = acc_hist.dropna()
        # ═══════════════════════════════════════════════════════════════
        
        return acc_hist
        
    except (IOError, FileNotFoundError, KeyError) as e:
        logger.warning(f"Nie można załadować danych accuracy: {e}")
        return None

def check_emergency_shutdown(config):
    """Sprawdza czy należy aktywować awaryjne wyłączenie."""
    emergency_cfg = config.get('strategy_behavior_settings', {}).get('emergency_shutdown', {})
    
    if not emergency_cfg.get('enabled', False):
        return False
    
    limit = emergency_cfg.get('consecutive_zero_accuracy_limit', 20)
    
    if EMERGENCY_STATE['consecutive_zero_count'] >= limit:
        if not EMERGENCY_STATE['is_shutdown']:
            logger.critical("=" * 80)
            logger.critical("🚨 AWARYJNE WYŁĄCZENIE AUTOREFLEKSJI!")
            logger.critical("=" * 80)
            logger.critical(f"   Wykryto {EMERGENCY_STATE['consecutive_zero_count']} kolejnych próbek z accuracy=0")
            logger.critical(f"   Limit: {limit}")
            logger.critical(f"   Przełączam na TRUST_ARIMAX do czasu normalizacji danych")
            logger.critical("=" * 80)
            
            EMERGENCY_STATE['is_shutdown'] = True
        
        return True
    else:
        if EMERGENCY_STATE['is_shutdown']:
            logger.info("=" * 80)
            logger.info("✅ DANE ZNORMALIZOWANE - WZNOWIENIE AUTOREFLEKSJI")
            logger.info("=" * 80)
            EMERGENCY_STATE['is_shutdown'] = False
        
        return False

def calculate_adaptive_rebellion_strength(config, acc_hist):
    """Dynamicznie dostosowuje siłę odwracania w zależności od accuracy."""
    adaptive_cfg = config.get('strategy_behavior_settings', {}).get('adaptive_rebellion', {})
    
    if not adaptive_cfg.get('enabled', False):
        return config.get('strategy_behavior_settings', {}).get('rebellion_strength', 1.0)
    
    if len(acc_hist) < 10:
        return adaptive_cfg.get('min_strength', 0.3)
    
    recent_accuracy = acc_hist.tail(10).mean()
    
    min_strength = adaptive_cfg.get('min_strength', 0.3)
    max_strength = adaptive_cfg.get('max_strength', 1.0)
    
    # Mapa: 0% accuracy → max_strength, 100% accuracy → min_strength
    if recent_accuracy >= 70:
        strength = min_strength
    elif recent_accuracy <= 30:
        strength = max_strength
    else:
        # Liniowa interpolacja
        strength = max_strength - (recent_accuracy - 30) / 40 * (max_strength - min_strength)
    
    logger.debug(f"Adaptive rebellion: accuracy={recent_accuracy:.1f}% → strength={strength:.2f}")
    
    return strength

# ==============================================================================
# TRYBY ANALIZY
# ==============================================================================

def analyze_trend_mode(config, acc_hist):
    """Tryb TREND - analiza trendu krótko/długoterminowego."""
    params = config.get("trend_mode_settings", {})
    
    short_window = params.get("trend_lookback_short", 30)
    long_window = params.get("trend_lookback_long", 180)
    min_samples = params.get("min_samples_required", 10)
    
    if len(acc_hist) < min_samples:
        logger.warning(f"Za mało danych ({len(acc_hist)}) - wymagane minimum: {min_samples}. Używam TRUST_ARIMAX.")
        return 'TRUST_ARIMAX'
    
    if len(acc_hist) < long_window:
        logger.warning(f"Za mało danych do pełnej analizy ({len(acc_hist)} < {long_window}). Uproszczona analiza.")
        short_avg = acc_hist.tail(min(short_window, len(acc_hist))).mean()
        long_avg = acc_hist.mean()
    else:
        short_avg = acc_hist.tail(short_window).mean()
        long_avg = acc_hist.tail(long_window).mean()
    
    trend = short_avg - long_avg
    trust_threshold = params.get("trust_threshold_percent", 0.0)
    
    logger.info("=" * 80)
    logger.info("📊 ANALIZA TRENDU")
    logger.info("=" * 80)
    logger.info(f"   ├─ Próbek dostępnych: {len(acc_hist)}")
    logger.info(f"   ├─ Krótka średnia ({short_window}m): {short_avg:.2f}%")
    logger.info(f"   ├─ Długa średnia ({long_window}m): {long_avg:.2f}%")
    logger.info(f"   ├─ Trend: {trend:+.2f}%")
    logger.info(f"   └─ Próg zaufania: {trust_threshold}%")
    
    if trend >= trust_threshold:
        signal = 'TRUST_ARIMAX'
        logger.info(f"   ✅ Werdykt: TRUST (trend {trend:+.2f}% >= {trust_threshold}%)")
    else:
        signal = 'DISTRUST_ARIMAX'
        logger.info(f"   ⚠️  Werdykt: DISTRUST (trend {trend:+.2f}% < {trust_threshold}%)")
    
    logger.info("=" * 80)
    
    return signal

def analyze_bollinger_mode(config, acc_hist):
    """Tryb BOLLINGER_STATE - analiza pasm Bollingera."""
    params = config.get("bollinger_state_mode_settings", {})
    
    period = params.get("bollinger_period", 20)
    std_dev = params.get("bollinger_std_dev", 2.0)
    
    if len(acc_hist) < period:
        logger.warning(f"Za mało danych ({len(acc_hist)} < {period}). Używam TRUST_ARIMAX.")
        return 'TRUST_ARIMAX'
    
    recent_data = acc_hist.tail(period)
    ma = recent_data.mean()
    std = recent_data.std()
    
    upper_band = ma + (std_dev * std)
    lower_band = ma - (std_dev * std)
    current = acc_hist.iloc[-1]
    
    logger.info("=" * 80)
    logger.info("📊 ANALIZA BOLLINGER BANDS")
    logger.info("=" * 80)
    logger.info(f"   ├─ Aktualna wartość: {current:.2f}%")
    logger.info(f"   ├─ Średnia ({period}m): {ma:.2f}%")
    logger.info(f"   ├─ Górne pasmo: {upper_band:.2f}%")
    logger.info(f"   └─ Dolne pasmo: {lower_band:.2f}%")
    
    if current >= upper_band:
        signal = 'TRUST_ARIMAX'
        logger.info(f"   ✅ Werdykt: TRUST (powyżej górnego pasma)")
    elif current <= lower_band:
        signal = 'DISTRUST_ARIMAX'
        logger.info(f"   ⚠️  Werdykt: DISTRUST (poniżej dolnego pasma)")
    else:
        signal = 'TRUST_ARIMAX'
        logger.info(f"   ℹ️  Werdykt: TRUST (w środku pasm - neutralny)")
    
    logger.info("=" * 80)
    
    return signal

def analyze_simple_oscillator_mode(config, acc_hist):
    """Tryb SIMPLE_OSCILLATOR - prosta analiza zmian."""
    params = config.get("simple_oscillator_settings", {})
    
    min_change = params.get("min_accuracy_change_percent", 0.5)
    
    if len(acc_hist) < 2:
        logger.warning("Za mało danych (< 2). Używam TRUST_ARIMAX.")
        return 'TRUST_ARIMAX'
    
    current = acc_hist.iloc[-1]
    previous = acc_hist.iloc[-2]
    change = current - previous
    
    logger.info("=" * 80)
    logger.info("📊 ANALIZA OSCYLATORA")
    logger.info("=" * 80)
    logger.info(f"   ├─ Poprzednia wartość: {previous:.2f}%")
    logger.info(f"   ├─ Aktualna wartość: {current:.2f}%")
    logger.info(f"   ├─ Zmiana: {change:+.2f}%")
    logger.info(f"   └─ Próg: ±{min_change}%")
    
    if abs(change) < min_change:
        signal = 'TRUST_ARIMAX'
        logger.info(f"   ℹ️  Werdykt: TRUST (zmiana poniżej progu)")
    elif change > 0:
        signal = 'TRUST_ARIMAX'
        logger.info(f"   ✅ Werdykt: TRUST (poprawa accuracy)")
    else:
        signal = 'DISTRUST_ARIMAX'
        logger.info(f"   ⚠️  Werdykt: DISTRUST (spadek accuracy)")
    
    logger.info("=" * 80)
    
    return signal

def analyze_cyclical_mode(config):
    """Tryb CYCLICAL_STRATEGY - cykliczne przełączanie."""
    params = config.get("cyclical_strategy_settings", {})
    
    half_cycle_minutes = params.get("half_cycle_duration_minutes", 60)
    
    # Oblicz fazę cyklu
    minutes_since_epoch = int(datetime.now().timestamp() / 60)
    phase = (minutes_since_epoch % (half_cycle_minutes * 2)) / half_cycle_minutes
    
    logger.info("=" * 80)
    logger.info("📊 ANALIZA CYKLICZNA")
    logger.info("=" * 80)
    logger.info(f"   ├─ Długość pół-cyklu: {half_cycle_minutes} minut")
    logger.info(f"   ├─ Faza cyklu: {phase:.2f}")
    
    if phase < 1.0:
        signal = 'TRUST_ARIMAX'
        logger.info(f"   ✅ Werdykt: TRUST (faza {phase:.2f} < 1.0)")
    else:
        signal = 'DISTRUST_ARIMAX'
        logger.info(f"   ⚠️  Werdykt: DISTRUST (faza {phase:.2f} >= 1.0)")
    
    logger.info("=" * 80)
    
    return signal

# ==============================================================================
# GŁÓWNA FUNKCJA STRATEGA
# ==============================================================================

def run_strategist():
    try:
        with open(AUTOREFLECTION_CONFIG_PATH, 'r') as f: 
            config = json.load(f)
    except FileNotFoundError:
        logger.critical(f"KRYTYCZNY BŁĄD: Brak pliku '{AUTOREFLECTION_CONFIG_PATH}'!")
        return
    except json.JSONDecodeError as e:
        logger.critical(f"KRYTYCZNY BŁĄD: Nieprawidłowy format JSON: {e}")
        return

    if not config.get("enabled", False):
        logger.warning("Autorefleksja jest wyłączona w konfiguracji. Zamykanie.")
        return

    analysis_mode = config.get("analysis_mode", "TREND")
    check_interval = config.get("update_interval_seconds", 10)
    
    logger.info("=" * 80)
    logger.info(f"🚀 URUCHAMIANIE STRATEGA AUTOREFLEKSJI v3.0")
    logger.info("=" * 80)
    logger.info(f"   ├─ Tryb analizy: {analysis_mode}")
    logger.info(f"   ├─ Interwał sprawdzania: {check_interval}s")
    logger.info(f"   ├─ Plik wyjściowy: {config.get('output_file_path', REFLECTION_JSON_PATH)}")
    logger.info(f"   └─ Rebellion strength: {config.get('strategy_behavior_settings', {}).get('rebellion_strength', 1.0)}")
    logger.info("=" * 80)
    
    last_processed_count = 0

    while True:
        try:
            # ═══════════════════════════════════════════════════════════════
            # 1. SPRAWDŹ STAN AWARYJNY
            # ═══════════════════════════════════════════════════════════════
            if check_emergency_shutdown(config):
                emergency_cfg = config.get('strategy_behavior_settings', {}).get('emergency_shutdown', {})
                if emergency_cfg.get('force_trust_on_shutdown', True):
                    signal_state = 'TRUST_ARIMAX'
                else:
                    signal_state = EMERGENCY_STATE['last_valid_signal']
                
                # Zapisz stan awaryjny
                state_to_save = {
                    'correction_factor': 1.0, 
                    'signal_state': signal_state, 
                    'last_update_utc': datetime.now().isoformat(),
                    'rebellion_strength': 0.0,  # Wyłącz bunt w trybie awaryjnym
                    'emergency_mode': True,
                    'emergency_reason': f'consecutive_zero_accuracy >= {emergency_cfg.get("consecutive_zero_accuracy_limit", 20)}'
                }
                
                with open(REFLECTION_JSON_PATH, 'w') as f_json:
                    json.dump(state_to_save, f_json, indent=2)
                
                time.sleep(check_interval)
                continue
            
            # ═══════════════════════════════════════════════════════════════
            # 2. WCZYTAJ DANE ACCURACY
            # ═══════════════════════════════════════════════════════════════
            acc_hist = load_accuracy_data(config, analysis_mode)
            
            if acc_hist is None or len(acc_hist) == 0:
                logger.warning("Brak danych accuracy. Czekam...")
                time.sleep(check_interval)
                continue
            
            current_count = len(acc_hist)
            
            # Sprawdź czy są nowe dane
            if current_count <= last_processed_count:
                time.sleep(check_interval)
                continue
            
            # ═══════════════════════════════════════════════════════════════
            # 3. ANALIZA WEDŁUG WYBRANEGO TRYBU
            # ═══════════════════════════════════════════════════════════════
            if analysis_mode == "TREND":
                signal_state = analyze_trend_mode(config, acc_hist)
            elif analysis_mode == "BOLLINGER_STATE":
                signal_state = analyze_bollinger_mode(config, acc_hist)
            elif analysis_mode == "SIMPLE_OSCILLATOR":
                signal_state = analyze_simple_oscillator_mode(config, acc_hist)
            elif analysis_mode == "CYCLICAL_STRATEGY":
                signal_state = analyze_cyclical_mode(config)
            else:
                logger.error(f"Nieznany tryb analizy: {analysis_mode}. Używam TRUST_ARIMAX.")
                signal_state = 'TRUST_ARIMAX'
            
            # ═══════════════════════════════════════════════════════════════
            # 4. OBLICZ SIŁĘ ODWRACANIA (rebellion_strength)
            # ═══════════════════════════════════════════════════════════════
            rebellion_strength = calculate_adaptive_rebellion_strength(config, acc_hist)
            
            # ═══════════════════════════════════════════════════════════════
            # 5. ZAPISZ STAN
            # ═══════════════════════════════════════════════════════════════
            state_to_save = {
                'correction_factor': 1.0,  # Legacy parameter (już nie używany)
                'signal_state': signal_state, 
                'last_update_utc': datetime.now().isoformat(),
                'rebellion_strength': rebellion_strength,
                'emergency_mode': False,
                'analysis_mode': analysis_mode,
                'samples_analyzed': len(acc_hist)
            }
            
            output_path = config.get('output_file_path', REFLECTION_JSON_PATH)
            with open(output_path, 'w') as f_json:
                json.dump(state_to_save, f_json, indent=2)
            
            # ═══════════════════════════════════════════════════════════════
            # 6. LOGOWANIE
            # ═══════════════════════════════════════════════════════════════
            logger.critical("=" * 80)
            logger.critical(f"🎯 ROZKAZ STRATEGA: {signal_state}")
            logger.critical("=" * 80)
            logger.critical(f"   ├─ Siła odwracania: {rebellion_strength:.1%}")
            logger.critical(f"   ├─ Tryb analizy: {analysis_mode}")
            logger.critical(f"   ├─ Próbek przeanalizowanych: {len(acc_hist)}")
            logger.critical(f"   └─ Zapisano do: {output_path}")
            logger.critical("=" * 80)
            
            # Zapisz ostatni poprawny sygnał
            EMERGENCY_STATE['last_valid_signal'] = signal_state
            last_processed_count = current_count

            time.sleep(check_interval)

        except KeyboardInterrupt:
            logger.info("🛑 Otrzymano sygnał zamknięcia. Zamykanie stratega.")
            break
        except Exception as e:
            logger.error(f"Wystąpił błąd krytyczny w strategu: {e}", exc_info=True)
            time.sleep(30)

if __name__ == "__main__":
    run_strategist()
