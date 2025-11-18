"""
binance_websocket_provider.py (WERSJA ZREFRAKTORYZOWANA I ULEPSZONA)

Moduł dostarczający dane z Binance poprzez WebSocket.
Obsługuje połączenie, subskrypcję i przetwarzanie
danych o świecach (k-lines) w czasie rzeczywistym.

Kluczowe cechy tej wersji:
- Poprawnie odczytuje konfigurację przekazaną jako słownik (dict).
- Zawiera metody do sprawdzania stanu połączenia (`is_connected`, `is_running`).
- Posiada ulepszoną, bardziej niezawodną logikę automatycznego ponownego łączenia.
- Zwiększona czytelność i odporność na błędy.
"""

import websocket
import json
import logging
import threading
import time
from typing import Callable, Dict

class BinanceWebsocketProvider:
    """
    Dostarcza dane o świecach (k-lines) z Binance za pomocą WebSocket.
    """
    BASE_URL = "wss://stream.binance.com:9443/ws"

    def __init__(self, config: Dict[str, str], on_candle_callback: Callable[[Dict], None]):
        """
        Inicjalizuje dostawcę danych.
        :param config: Słownik zawierający 'symbol' i 'interval'.
        :param on_candle_callback: Funkcja zwrotna wywoływana dla każdej zamkniętej świecy.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # === POPRAWKA: Odczyt z słownika, a nie z obiektu config ===
        try:
            self.symbol = config['symbol']
            self.interval = config['interval']
        except KeyError as e:
            self.logger.critical(f"Krytyczny błąd: Brak wymaganego klucza w konfiguracji providera: {e}")
            raise ValueError(f"Konfiguracja dla BinanceWebsocketProvider musi zawierać klucz {e}")

        self.stream_name = f"{self.symbol.lower()}@kline_{self.interval}"
        self.on_candle_callback = on_candle_callback

        self._ws_app = None
        self._thread = None
        self._is_running = False
        self._stop_event = threading.Event()

    def _on_message(self, ws, message: str):
        """Callback wywoływany przy każdej nowej wiadomości z WebSocket."""
        try:
            data = json.loads(message)
            if data.get('e') == 'kline':
                candle = data['k']
                if candle['x']:  # Świeca jest zamknięta
                    formatted_candle = {
                        'timestamp': int(candle['t']),
                        'price': float(candle['c']),  # Cena zamknięcia
                        'volume': float(candle['v']) # Wolumen
                    }
                    self.on_candle_callback(formatted_candle)
        except json.JSONDecodeError:
            self.logger.warning("Nie udało się zdekodować wiadomości JSON z WebSocket.")
        except Exception as e:
            self.logger.error(f"Nieoczekiwany błąd w _on_message: {e}", exc_info=True)

    def _on_error(self, ws, error: Exception):
        """Callback dla błędów WebSocket."""
        self.logger.error(f"Błąd WebSocket: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """Callback przy zamknięciu połączenia."""
        self.logger.info("### Połączenie WebSocket zamknięte ###")
        # Logika ponownego łączenia jest teraz w pętli `_run`

    def _on_open(self, ws):
        """Callback po otwarciu połączenia."""
        self.logger.info(f"✅ Połączenie WebSocket otwarte. Subskrybuję strumień: {self.stream_name}")
        params = {
            "method": "SUBSCRIBE",
            "params": [self.stream_name],
            "id": 1
        }
        try:
            ws.send(json.dumps(params))
        except Exception as e:
            self.logger.error(f"Nie udało się wysłać subskrypcji: {e}")

    def _run(self):
        """Główna pętla wątku, która zarządza połączeniem."""
        while not self._stop_event.is_set():
            self.logger.info(f"Próba nawiązania połączenia z {self.BASE_URL}...")
            self._ws_app = websocket.WebSocketApp(
                self.BASE_URL,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            self._ws_app.run_forever()
            
            if not self._stop_event.is_set():
                self.logger.warning("Połączenie utracone. Próba ponownego połączenia za 5 sekund...")
                time.sleep(5)
        self.logger.info("Wątek dostawcy danych został zatrzymany.")


    def start(self):
        """Uruchamia połączenie WebSocket w osobnym wątku."""
        if self._is_running:
            self.logger.warning("Próba uruchomienia już działającego dostawcy danych.")
            return

        self.logger.info("Uruchamianie dostawcy danych WebSocket...")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._is_running = True

    def stop(self):
        """Zatrzymuje połączenie WebSocket i kończy działanie wątku."""
        if not self._is_running:
            return
            
        self.logger.info("Zatrzymywanie dostawcy danych WebSocket...")
        self._stop_event.set()
        
        if self._ws_app:
            self._ws_app.close()
            
        if self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                self.logger.warning("Wątek WebSocket nie zakończył się w wyznaczonym czasie.")
                
        self._is_running = False
        self.logger.info("✅ Dostawca danych zatrzymany.")

    def is_connected(self) -> bool:
        """Sprawdza, czy WebSocket jest aktualnie połączony."""
        return self._ws_app and self._ws_app.sock and self._ws_app.sock.connected

    def is_running(self) -> bool:
        """Sprawdza, czy główna pętla dostawcy jest uruchomiona."""
        return self._is_running
# END
