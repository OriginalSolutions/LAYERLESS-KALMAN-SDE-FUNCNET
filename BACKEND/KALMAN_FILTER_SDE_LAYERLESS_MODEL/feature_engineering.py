# plik: feature_engineering.py
# WERSJA: 10.0 (KRYTYCZNA POPRAWKA: Skalowanie price_delta × 1000 dla BTC)

import numpy as np
import pandas as pd
from typing import Dict, Any

def _calculate_tema(data: pd.Series, window: int) -> pd.Series:
    """Oblicza Potrójną Średnią Kroczącą Wykładniczą (TEMA)."""
    if len(data.dropna()) < window: 
        return pd.Series(np.nan, index=data.index)
    ema1 = data.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    return (3 * ema1) - (3 * ema2) + ema3

def _robust_fillna(series: pd.Series) -> pd.Series:
    """Bezpieczne wypełnianie brakujących danych."""
    return series.ffill().fillna(0)

def _calculate_kama(series: pd.Series, period: int, fast_ema: int, slow_ema: int) -> pd.Series:
    """Oblicza Adaptacyjną Średnią Kroczącą Kaufmana (KAMA)."""
    if series.isnull().all() or len(series) < period: 
        return pd.Series(np.nan, index=series.index)
    
    change = series.diff(period).abs()
    volatility = series.diff().abs().rolling(window=period).sum()
    epsilon = 1e-9
    efficiency_ratio = change / (volatility + epsilon)
    
    fast_sc = 2 / (fast_ema + 1)
    slow_sc = 2 / (slow_ema + 1)
    smoothing_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = pd.Series(np.nan, index=series.index, dtype=np.float64)
    first_valid_index_pos = series.first_valid_index()
    if first_valid_index_pos is None: 
        return kama
    
    start_index_pos = series.index.get_loc(first_valid_index_pos) + period
    if start_index_pos >= len(series): 
        return kama
    
    start_index = series.index[start_index_pos]
    initial_sma = series.iloc[start_index_pos - period : start_index_pos + 1].mean()
    kama.loc[start_index] = initial_sma

    for i in range(start_index_pos + 1, len(series)):
        current_idx, prev_idx = series.index[i], series.index[i-1]
        sc = smoothing_constant.loc[current_idx]
        if pd.notna(sc) and pd.notna(kama.loc[prev_idx]):
            kama.loc[current_idx] = kama.loc[prev_idx] + sc * (series.loc[current_idx] - kama.loc[prev_idx])
        else:
            kama.loc[current_idx] = kama.loc[prev_idx]
    
    return kama

def calculate_price_features(prices: pd.Series, volumes: pd.Series, config: Any) -> pd.DataFrame:
    """
    Oblicza "czyste" cechy dynamiczne, w tym wskaźniki wyprzedzające.
    
    WERSJA 10.0: KRYTYCZNA POPRAWKA - Skalowanie price_delta × 1000
    
    POWÓD ZMIANY:
    Dla BTC (cena ~110,000 USDT):
    - Typowa zmiana 1 minuta: ±10-50 USDT
    - Po normalizacji RobustScaler: ±0.00001-0.0005
    - Model LSTM/SDE widzi to jako SZUM!
    
    ROZWIĄZANIE:
    - Przemnażamy price_delta × 1000
    - Teraz: ±10-50 USDT → ±10,000-50,000 (w skali × 1000)
    - Po normalizacji: ±0.01-0.5 (WIDOCZNE dla modelu!)
    
    ⚠️  UWAGA: W learning_engine.predict() MUSI być odwrotne skalowanie (/ 1000)!
    """
    df = pd.DataFrame({'prices': prices.copy(), 'volumes': volumes.copy()})
    
    if df.shape[0] < 3:
        all_feature_cols = [
            'price_delta', 'log_returns', 'momentum_tema', 'volatility', 
            'kama_log_returns', 'kama_price', 'price_kama_context',
            'volume_delta', 'trade_intensity', 'price_acceleration', 'momentum_oscillator'
        ]
        return pd.DataFrame(columns=all_feature_cols)

    epsilon = 1e-9

    # ═══════════════════════════════════════════════════════════════════════════
    # 🔧 KRYTYCZNA ZMIANA: SKALOWANIE price_delta × 1000
    # ═══════════════════════════════════════════════════════════════════════════
    # BYŁO:
    # df['price_delta'] = df['prices'].diff()
    
    # NOWE (× 1000 dla lepszej widoczności w modelu):
    df['price_delta'] = df['prices'].diff() * 1000.0
    
    # PRZYKŁAD:
    # Cena wzrasta z 110,000 → 110,015 USDT
    # STARA wersja: price_delta = +15
    # NOWA wersja:  price_delta = +15,000
    # Po normalizacji (stara): ~0.00014 (niewidoczne)
    # Po normalizacji (nowa):  ~0.14 (widoczne!)
    # ═══════════════════════════════════════════════════════════════════════════
    
    df['log_returns'] = np.log(df['prices'] / df['prices'].shift(1).replace(0, epsilon))
    df['volume_delta'] = df['volumes'].diff()
    df['trade_intensity'] = np.log(df['volumes'] + 1)
    df['momentum_tema'] = _calculate_tema(df['log_returns'], window=config.feature_tema_period)
    df['volatility'] = df['log_returns'].rolling(window=config.feature_volatility_window, min_periods=1).std()

    # ═══════════════════════════════════════════════════════════════════════════
    # PRICE ACCELERATION (używa przeskalowanego price_delta)
    # ═══════════════════════════════════════════════════════════════════════════
    raw_acceleration = df['price_delta'].diff()
    
    # Normalizacja przez zmienność price_delta
    price_delta_std = df['price_delta'].rolling(window=20, min_periods=1).std()
    df['price_acceleration'] = raw_acceleration / (price_delta_std + epsilon)
    
    # ═══════════════════════════════════════════════════════════════════════════

    # MOMENTUM OSCILLATOR (MACD-like na TEMA)
    fast_tema_period = getattr(config, 'feature_fast_tema_period', 3) 
    slow_tema_period = getattr(config, 'feature_slow_tema_period', 9)
    
    fast_tema = _calculate_tema(df['log_returns'], window=fast_tema_period)
    slow_tema = _calculate_tema(df['log_returns'], window=slow_tema_period)
    df['momentum_oscillator'] = fast_tema - slow_tema

    # ═══════════════════════════════════════════════════════════════════════════
    # KAMA (używa przeskalowanego price_delta)
    # ═══════════════════════════════════════════════════════════════════════════
    kama_settings = getattr(config, 'kama_settings', {})
    if kama_settings.get('enabled', False):
        df['kama_log_returns'] = _calculate_kama(
            series=df['log_returns'], 
            period=kama_settings.get('period', 10), 
            fast_ema=kama_settings.get('fast_ema_period', 2), 
            slow_ema=kama_settings.get('slow_ema_period', 30)
        )
        
        # KAMA na przeskalowanym price_delta
        df['kama_price'] = _calculate_kama(
            series=df['price_delta'], 
            period=kama_settings.get('period', 10), 
            fast_ema=kama_settings.get('fast_ema_period', 2), 
            slow_ema=kama_settings.get('slow_ema_period', 30)
        )
        
        df['price_kama_context'] = df['price_delta'] - df['kama_price']
    else:
        df['kama_log_returns'], df['kama_price'], df['price_kama_context'] = 0.0, 0.0, 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # FINALNA LISTA CECH
    # ═══════════════════════════════════════════════════════════════════════════
    final_features = [
        'price_delta',           # ← Teraz przeskalowane × 1000!
        'log_returns', 
        'momentum_tema', 
        'volatility',
        'kama_log_returns', 
        'kama_price',           # ← Bazuje na przeskalowanym price_delta
        'price_kama_context',   # ← Bazuje na przeskalowanym price_delta
        'volume_delta', 
        'trade_intensity',
        'price_acceleration',   # ← Bazuje na przeskalowanym price_delta
        'momentum_oscillator'
    ]
    
    df.drop(columns=['prices', 'volumes'], inplace=True, errors='ignore')
    return_cols = [col for col in final_features if col in df.columns]
    return df[return_cols]


def calculate_error_features(errors: pd.Series, config: Any) -> pd.DataFrame:
    """
    Oblicza cechy związane z błędem predykcji.
    
    ⚠️  UWAGA: Jeśli errors bazują na price_delta, mogą być również przeskalowane!
    """
    df = pd.DataFrame({'errors': errors.copy()})
    
    if df.shape[0] < config.error_volatility_std_window:
        return pd.DataFrame(columns=['error_volatility'])
    
    std_window = config.error_volatility_std_window
    tema_period = config.error_volatility_tema_period
    error_std = df['errors'].rolling(window=std_window).std()
    epsilon_vol = 1e-9
    error_std_filled = error_std.fillna(0).replace(0, epsilon_vol)
    smoothed_error_std = _calculate_tema(error_std_filled, window=tema_period)
    df['error_volatility'] = smoothed_error_std
    
    return df[['error_volatility']]
