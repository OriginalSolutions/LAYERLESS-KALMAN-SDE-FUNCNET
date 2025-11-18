# plik: rest_api_fetcher.py
import requests
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"

def fetch_historical_data(symbol: str, interval: str, limit: int) -> List[Dict[str, Any]]:
    """
    Pobiera historyczne dane świecowe (k-lines) z Binance REST API.

    Args:
        symbol (str): Symbol pary, np. "BTCUSDT".
        interval (str): Interwał świec, np. "1m", "5m", "1h".
        limit (int): Liczba świec do pobrania (max 1000 na jedno zapytanie).

    Returns:
        Lista słowników, gdzie każdy słownik reprezentuje świecę
        w formacie oczekiwanym przez główny skrypt.
    """
    params = {
        'symbol': symbol.upper(),
        'interval': interval,
        'limit': min(limit, 1000) # Binance API ma limit 1000
    }
    
    logger.info(f"Pobieranie {params['limit']} świec historycznych dla {params['symbol']} z interwałem {params['interval']}...")
    
    try:
        response = requests.get(BINANCE_API_URL, params=params, timeout=10)
        response.raise_for_status()  # Rzuci wyjątkiem dla kodów błędów 4xx/5xx
        
        raw_data = response.json()
        
        # --- Transformacja danych do wymaganego formatu ---
        # API Binance zwraca: [timestamp, open, high, low, close, volume, ...]
        # My potrzebujemy: [{'timestamp': ..., 'price': ..., 'volume': ...}]
        
        formatted_data = []
        for kline in raw_data:
            formatted_data.append({
                'timestamp': int(kline[0]),
                'price': float(kline[4]),  # Używamy ceny zamknięcia (close price)
                'volume': float(kline[5])
            })
            
        logger.info(f"✅ Pomyślnie pobrano i sformatowano {len(formatted_data)} świec.")
        return formatted_data

    except requests.exceptions.RequestException as e:
        logger.error(f"Błąd sieciowy podczas pobierania danych z Binance API: {e}")
        raise  # Rzucamy dalej wyjątek, aby główny skrypt wiedział o problemie
    except Exception as e:
        logger.error(f"Nieoczekiwany błąd podczas przetwarzania danych z API: {e}")
        raise
