#!/bin/bash

# --- Konfiguracja ---
# Nazwa procesu, której będziemy szukać. Ważne, aby była unikalna.
PROCESS_NAME="NEURAL__sde_pure_gaussian_process.py"

# Pełna komenda do uruchomienia procesu W TLE, jeśli nie jest aktywny
COMMAND_TO_RUN="nohup /root/AI_NEURAL_SDE/venv/bin/python3 -u /root/AI_NEURAL_SDE/T_E_S_T/NEURAL__sde_pure_gaussian_process.py > /root/AI_NEURAL_SDE/T_E_S_T/NEURAL__sde_pure_gaussian_process.log 2>&1 &"

# Co ile sekund skrypt ma sprawdzać, czy proces działa
CHECK_INTERVAL=120 # Sprawdzaj co 15 sekund
# --- Koniec Konfiguracji ---

echo "===================================================="
echo " Uruchamiam Watchdog dla procesu: $PROCESS_NAME"
echo " Sprawdzanie co $CHECK_INTERVAL sekund..."
echo "===================================================="

while true; do
    # Użyj pgrep, aby sprawdzić, czy proces zawierający naszą nazwę skryptu jest uruchomiony.
    # -f sprawdza całą linię komendy, a nie tylko nazwę programu.
    # -c zlicza liczbę znalezionych procesów.
    # Używamy [N]EURAL... aby sam grep nie znalazł swojego własnego procesu na liście.
    PROCESS_COUNT=$(pgrep -fc "[N]EURAL__sde_pure_gaussian_process.py")

    # Sprawdź, czy liczba znalezionych procesów jest równa 0
    if [ "$PROCESS_COUNT" -eq 0 ]; then
        # Jeśli tak, proces nie działa. Uruchom go ponownie.
        echo "$(date): UWAGA! Proces '$PROCESS_NAME' nie został znaleziony."
        echo "$(date): Uruchamiam ponownie..."
        
        # Uruchom komendę zdefiniowaną na górze.
        # eval jest używane, aby poprawnie zinterpretować całą złożoną komendę jako jedną całość.
        eval $COMMAND_TO_RUN
        
        echo "$(date): Proces został uruchomiony."
    else
        # Jeśli proces działa, nic nie rób, tylko wyświetl informację (opcjonalnie).
        # Możesz zakomentować tę linię, jeśli nie chcesz widzieć logów "wszystko OK".
        echo "$(date): Proces '$PROCESS_NAME' działa (znaleziono $PROCESS_COUNT instancji). Wszystko w porządku."
    fi

    # Poczekaj zdefiniowaną liczbę sekund przed następnym sprawdzeniem.
    sleep $CHECK_INTERVAL
done
