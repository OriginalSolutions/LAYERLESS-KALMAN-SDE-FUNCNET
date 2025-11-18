# plik: particle_smoother.py
# WERSJA: 3.3 (MONITORING - Dodano metody get_effective_sample_size i get_particle_stats)

import torch
import torch.nn as nn
import logging

class ParticleSmoother(nn.Module):
    """
    Implementuje Filtr Cząsteczkowy z mechanizmem "kotwiczenia" (rebasing),
    aby uniknąć kumulacji błędów i dryfowania prognozy.
    
    Wersja 3.3 dodaje rozszerzone metody monitoringu:
    - get_effective_sample_size() - oblicza ESS (Effective Sample Size)
    - get_particle_stats() - zwraca pełne statystyki chmury cząsteczek
    - resample_if_needed() - zwraca informację czy wykonano resampling
    """
    def __init__(self, settings: dict, device: torch.device, initial_value: float = 0.0):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        
        self.num_particles = settings.get('num_particles', 100)
        self.process_noise = nn.Parameter(torch.tensor(settings.get('process_noise_std', 0.05)), requires_grad=False)
        self.measurement_noise = nn.Parameter(torch.tensor(settings.get('measurement_noise_std', 0.5)), requires_grad=False)
        self.control_gain = settings.get('control_gain', 1.0)
        
        self.register_buffer('particles', torch.full((self.num_particles,), initial_value, dtype=torch.float32, device=self.device))
        self.register_buffer('weights', torch.full((self.num_particles,), 1.0 / self.num_particles, dtype=torch.float32, device=self.device))

        self.logger.info(f"✅ Filtr Cząsteczkowy [v3.3 Monitor] zainicjalizowany. Cząsteczki={self.num_particles}, Szum Procesu={self.process_noise.item():.4f}, Szum Pomiaru={self.measurement_noise.item():.4f}, Wzmocnienie={self.control_gain:.2f}")

    def rebase(self, new_base_price: float):
        """
        "Kotwiczy" całą chmurę cząsteczek wokół nowej, znanej ceny bazowej.
        To resetuje stan filtra i zapobiega dryfowaniu.
        """
        self.particles.fill_(new_base_price)
        self.logger.debug(f"Filtr 'przekotwiczony' do nowej ceny bazowej: {new_base_price:.2f}")

    def predict(self, predicted_change: float):
        """
        Krok predykcji: przesuwa zakotwiczone cząsteczki o prognozowaną ZMIANĘ ceny.
        """
        noise = torch.randn(self.num_particles, device=self.device) * self.process_noise
        self.particles += (predicted_change * self.control_gain) + noise

    def update(self, predicted_target_price: float):
        """
        Krok aktualizacji: waży cząsteczki na podstawie prognozowanej ceny docelowej.
        """
        distance = self.particles - predicted_target_price
        variance = torch.clamp(self.measurement_noise ** 2, min=1e-9)
        likelihood = torch.exp(-0.5 * (distance ** 2) / variance)
        
        self.weights *= likelihood
        self.weights += 1e-12 
        sum_weights = torch.sum(self.weights)
        if sum_weights > 1e-9:
            self.weights /= sum_weights
        else:
            self.weights.fill_(1.0 / self.num_particles)
            self.logger.warning("Wagi filtru cząsteczkowego uległy degeneracji. Zostały zresetowane.")

    def get_effective_sample_size(self) -> float:
        """
        🆕 Oblicza Effective Sample Size (ESS) - liczbę "efektywnych" cząsteczek.
        ESS = 1 / Σ(w_i²)
        
        Interpretacja:
        - ESS ≈ num_particles → wszystkie cząstki mają podobne wagi (dobry stan)
        - ESS << num_particles → wagi zdegenerowane (większość wag ~0)
        
        Returns:
            float: Liczba efektywnych cząsteczek (zakres: 1 do num_particles)
        """
        return (1.0 / torch.sum(self.weights ** 2)).item()

    def resample_if_needed(self) -> bool:
        """
        Próbkowanie wtórne, aby uniknąć degeneracji wag.
        Wykonuje resampling gdy ESS spadnie poniżej 50% całkowitej liczby cząsteczek.
        
        🆕 Returns:
            bool: True jeśli wykonano resampling, False w przeciwnym razie
        """
        effective_sample_size = self.get_effective_sample_size()
        threshold = self.num_particles / 2.0
        
        if effective_sample_size < threshold:
            indices = torch.multinomial(self.weights, self.num_particles, replacement=True)
            self.particles = self.particles[indices]
            self.weights.fill_(1.0 / self.num_particles)
            self.logger.debug(f"Wykonano resampling cząsteczek (ESS={effective_sample_size:.0f} < {threshold:.0f}).")
            return True
        return False

    def get_smoothed_prediction(self, *args, **kwargs) -> float:
        """
        Zwraca ostateczną, wygładzoną prognozę jako średnią ważoną chmury cząsteczek.
        
        Returns:
            float: Wygładzona prognoza (średnia ważona wszystkich cząsteczek)
        """
        return torch.dot(self.particles, self.weights).item()
    
    def get_particle_stats(self) -> dict:
        """
        🆕 Zwraca szczegółowe statystyki chmury cząsteczek dla zaawansowanego monitoringu.
        
        Returns:
            dict: Słownik zawierający:
                - mean: Średnia ważona (= wygładzona prognoza)
                - std: Odchylenie standardowe chmury cząsteczek
                - min: Minimalna wartość cząsteczki
                - max: Maksymalna wartość cząsteczki
                - ess: Effective Sample Size
                - ess_ratio: ESS jako procent całkowitej liczby cząsteczek
                - max_weight: Największa waga pojedynczej cząsteczki
                - min_weight: Najmniejsza waga pojedynczej cząsteczki
        """
        return {
            'mean': self.get_smoothed_prediction(),
            'std': torch.std(self.particles).item(),
            'min': torch.min(self.particles).item(),
            'max': torch.max(self.particles).item(),
            'ess': self.get_effective_sample_size(),
            'ess_ratio': self.get_effective_sample_size() / self.num_particles,
            'max_weight': torch.max(self.weights).item(),
            'min_weight': torch.min(self.weights).item()
        }
