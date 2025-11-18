# plik: sde_models.py
# WERSJA: 12.0 (LEARNED DYNAMICS KF) - Macierz transformacji stanu F w filtrze Kalmana jest teraz uczonym parametrem.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ############################################################################
# ##### KOMPONENTY WSPÓŁDZIELONE                                         #####
# ############################################################################

def ScaledSigmoid(x, min_val, max_val):
    """Przeskalowany sigmoid, który mapuje wyjście do zakresu [min_val, max_val]."""
    return min_val + (max_val - min_val) * torch.sigmoid(x)

class KernelAttention(nn.Module):
    """Standardowy, bezwarstwowy mechanizm atencji oparty na jądrze RBF."""
    def __init__(self, state_size: int, num_prototypes: int = 32):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, state_size))
        self.log_gamma = nn.Parameter(torch.zeros(num_prototypes))
    
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        distances_sq = torch.cdist(y, self.prototypes, p=2).pow(2)
        similarity_scores = torch.exp(-torch.exp(self.log_gamma) * distances_sq)
        attention_weights = F.softmax(similarity_scores, dim=-1)
        return torch.matmul(attention_weights, self.prototypes)

class _EKF_Component(nn.Module):
    """
    Rozszerzony Filtr Kalmana z UCZONĄ MACIERZĄ TRANSFORMACJI STANU (F).
    """
    def __init__(self, state_size, kalman_config):
        super().__init__()
        self.state_size = state_size
        initial_q = kalman_config.get('initial_process_noise', 1e-4)
        initial_r = kalman_config.get('initial_measurement_noise', 1e-1)
        self.log_q = nn.Parameter(torch.tensor(np.log(initial_q)))
        self.log_r = nn.Parameter(torch.tensor(np.log(initial_r)))
        
        # ### KLUCZOWA ZMIANA: F jest teraz uczonym parametrem! ###
        # Inicjalizujemy ją jako macierz jednostkową, aby na początku treningu
        # zachowywała się jak standardowy filtr, ale pozwalamy jej się uczyć.
        self.F = nn.Parameter(torch.eye(state_size)) 
        
        self.register_buffer('I', torch.eye(state_size))
        self.register_buffer('x_kalman', torch.zeros(1, state_size))
        self.register_buffer('P_kalman', torch.eye(state_size).unsqueeze(0) * 10.0)

    def forward(self, y, measurement_source):
        batch_size = y.shape[0]
        device = y.device
        
        if self.x_kalman.shape[0] != batch_size:
            self.x_kalman = torch.zeros(batch_size, self.state_size, device=device)
            self.P_kalman = torch.eye(self.state_size, device=device).repeat(batch_size, 1, 1) * 10.0

        Q = torch.eye(self.state_size, device=device) * torch.exp(self.log_q)
        R = torch.eye(self.state_size, device=device) * torch.exp(self.log_r)

        # Teraz F jest parametrem, więc nie trzeba go przenosić na device
        F_b = self.F.unsqueeze(0).repeat(batch_size, 1, 1)
        H_b = self.I.unsqueeze(0).repeat(batch_size, 1, 1)

        # Krok predykcji teraz używa UCZONEJ dynamiki
        x_pred_b = torch.bmm(F_b, self.x_kalman.unsqueeze(-1))
        P_pred_b = torch.bmm(torch.bmm(F_b, self.P_kalman), F_b.transpose(1, 2)) + Q

        z_b = measurement_source.unsqueeze(-1)
        y_innovation_b = z_b - torch.bmm(H_b, x_pred_b)
        S_b = torch.bmm(torch.bmm(H_b, P_pred_b), H_b.transpose(1, 2)) + R

        try:
            S_inv_b = torch.inverse(S_b)
        except torch.linalg.LinAlgError:
            jitter = torch.eye(self.state_size, device=device).unsqueeze(0) * 1e-6
            S_inv_b = torch.inverse(S_b + jitter)

        K_b = torch.bmm(torch.bmm(P_pred_b, H_b.transpose(1, 2)), S_inv_b)

        x_new_b = x_pred_b + torch.bmm(K_b, y_innovation_b)
        P_new_b = torch.bmm((self.I.unsqueeze(0) - torch.bmm(K_b, H_b)), P_pred_b)

        self.x_kalman = x_new_b.squeeze(-1).detach()
        self.P_kalman = P_new_b.detach()
        
        return self.x_kalman

# ############################################################################
# ##### MODELE BAZOWE (v7.0) - BEZ ZMIAN, DLA ZACHOWANIA KOMPATYBILNOŚCI #####
# ############################################################################

class FeatureAwareSDE(nn.Module):
    """Standardowy Model SDE (bez atencji), WERSJA: 7.0 - Gwarancja mocy dla Velocity."""
    noise_type = "diagonal"; sde_type = "ito"
    def __init__(self, state_size, stability_settings, feature_weights_clamp_range, aug_config):
        super().__init__()
        self.state_size = state_size
        self.min_velocity_gain = stability_settings.get('velocity_min_gain', 0.1)
        self.feature_weights = nn.Parameter(torch.randn(state_size, state_size) * 0.1)
        self.theta = nn.Parameter(torch.tensor(0.5))
        self.non_linear_modulator = F.gelu
        self.sigma_base = nn.Parameter(torch.tensor(-3.0))
        self.sigma_adapt_weight = nn.Parameter(torch.randn(state_size, state_size) * 0.01)
        self.velocity_weights = nn.Parameter(torch.randn(state_size, state_size) * 0.1)
        self.velocity_gain = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('prev_y', torch.zeros(1, state_size))
    def f(self, t, y):
        mean_reversion = torch.tanh(self.theta) * (self.non_linear_modulator(y @ self.feature_weights) - y)
        if t > 0:
            velocity = y - self.prev_y
            velocity_transformed = velocity @ self.velocity_weights
            gain_factor = ScaledSigmoid(self.velocity_gain, self.min_velocity_gain, 1.0)
            velocity_term = gain_factor * velocity_transformed
        else:
            velocity_term = torch.zeros_like(y)
        self.prev_y = y.detach().clone()
        return mean_reversion + velocity_term
    def g(self, t, y):
        return self.sigma_base.exp() * torch.tanh(y @ self.sigma_adapt_weight)

class HybridAttentionSDE(nn.Module):
    """Model SDE z atencją, WERSJA: 7.0 - Gwarancja mocy dla Velocity."""
    noise_type = "diagonal"; sde_type = "ito"
    def __init__(self, state_size, stability_settings, feature_weights_clamp_range, aug_config, attention_config):
        super().__init__()
        self.state_size = state_size
        self.min_velocity_gain = stability_settings.get('velocity_min_gain', 0.1)
        num_prototypes = attention_config.get('attention_dim', 32)
        self.attention_mechanism = KernelAttention(state_size, num_prototypes=num_prototypes)
        self.theta = nn.Parameter(torch.tensor(0.5))
        self.non_linear_modulator = F.gelu
        self.sigma_base = nn.Parameter(torch.tensor(-3.0))
        self.sigma_adapt_weight = nn.Parameter(torch.randn(state_size, state_size) * 0.01)
        self.velocity_weights = nn.Parameter(torch.randn(state_size, state_size) * 0.1)
        self.velocity_gain = nn.Parameter(torch.tensor(0.0))
        self.register_buffer('prev_y', torch.zeros(1, state_size))
    def f(self, t, y):
        mean_reversion = torch.tanh(self.theta) * (self.non_linear_modulator(y + self.attention_mechanism(y)) - y)
        if t > 0:
            velocity = y - self.prev_y
            velocity_transformed = velocity @ self.velocity_weights
            gain_factor = ScaledSigmoid(self.velocity_gain, self.min_velocity_gain, 1.0)
            velocity_term = gain_factor * velocity_transformed
        else:
            velocity_term = torch.zeros_like(y)
        self.prev_y = y.detach().clone()
        return mean_reversion + velocity_term
    def g(self, t, y):
        return self.sigma_base.exp() * torch.tanh(y @ self.sigma_adapt_weight)

# ############################################################################
# ##### NOWE MODELE Z WBUDOWANYM, UCZONYM FILTREM KALMANA (EKF)           #####
# ############################################################################

class KalmanFeatureAwareSDE(nn.Module):
    noise_type = "diagonal"; sde_type = "ito"
    def __init__(self, state_size, stability_settings, feature_weights_clamp_range, aug_config, attention_config, kalman_config):
        super().__init__(); self.internal_sde = FeatureAwareSDE(state_size, stability_settings, feature_weights_clamp_range, aug_config); self.kalman_filter = _EKF_Component(state_size, kalman_config)
    def f(self, t, y): raw_sde_drift = self.internal_sde.f(t, y); kalman_filtered_state = self.kalman_filter(y, y + raw_sde_drift); return kalman_filtered_state - y
    def g(self, t, y): return self.internal_sde.g(t, y)
class KalmanHybridAttentionSDE(nn.Module):
    noise_type = "diagonal"; sde_type = "ito"
    def __init__(self, state_size, stability_settings, feature_weights_clamp_range, aug_config, attention_config, kalman_config):
        super().__init__(); self.internal_sde = HybridAttentionSDE(state_size, stability_settings, feature_weights_clamp_range, aug_config, attention_config); self.kalman_filter = _EKF_Component(state_size, kalman_config)
    def f(self, t, y): raw_sde_drift = self.internal_sde.f(t, y); kalman_filtered_state = self.kalman_filter(y, y + raw_sde_drift); return kalman_filtered_state - y
    def g(self, t, y): return self.internal_sde.g(t, y)
class KalmanKernelAttention(KernelAttention):
    def __init__(self, state_size: int, num_prototypes: int = 32, kalman_config=None):
        super().__init__(state_size, num_prototypes); self.use_kalman = bool(kalman_config); self.kalman_filter = _EKF_Component(state_size, kalman_config) if self.use_kalman else None
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        attention_output = super().forward(y); return self.kalman_filter(attention_output, attention_output) if self.use_kalman else attention_output
class HybridAttentionSDE_WithKalmanAttention(HybridAttentionSDE):
    def __init__(self, state_size, stability_settings, feature_weights_clamp_range, aug_config, attention_config, kalman_config):
        super().__init__(state_size, stability_settings, feature_weights_clamp_range, aug_config, attention_config); num_prototypes = attention_config.get('attention_dim', 32); self.attention_mechanism = KalmanKernelAttention(state_size, num_prototypes, kalman_config)

# ############################################################################
# ##### NOWY MODEL EKSPERYMENTALNY: SDE PROWADZONE PRZEZ KALMANA         #####
# ############################################################################

class KALMANGuidedAttentionSDE(nn.Module):
    noise_type = "diagonal"; sde_type = "ito"
    def __init__(self, state_size, stability_settings, feature_weights_clamp_range, aug_config, attention_config, kalman_config):
        super().__init__(); internal_state_size = state_size + state_size; self.kalman_guide = _EKF_Component(state_size, kalman_config); num_prototypes = attention_config.get('attention_dim', 32); self.attention_mechanism = KernelAttention(internal_state_size, num_prototypes=num_prototypes); self.theta = nn.Parameter(torch.tensor(0.5)); self.sigma_base = nn.Parameter(torch.tensor(-3.0)); self.sigma_adapt_weight = nn.Parameter(torch.randn(state_size, state_size) * 0.01); self.final_drift_scaler = nn.Parameter(torch.randn(internal_state_size, state_size) * 0.01)
    def f(self, t, y):
        kalman_trend_prediction = self.kalman_guide(y, y); combined_input = torch.cat([y, kalman_trend_prediction], dim=1); path_a = torch.tanh(combined_input); path_b = torch.sigmoid(combined_input); processed_drift = path_a * path_b; attention_context = self.attention_mechanism(combined_input); final_drift_input = processed_drift + attention_context; final_drift = final_drift_input @ self.final_drift_scaler; mean_reversion_drift = torch.tanh(self.theta) * (F.mish(final_drift) - y); return mean_reversion_drift
    def g(self, t, y): return self.sigma_base.exp() * torch.tanh(y @ self.sigma_adapt_weight)
