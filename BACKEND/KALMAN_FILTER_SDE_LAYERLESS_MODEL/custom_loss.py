# plik: custom_loss.py
# WERSJA: 55.0 (OPTYMALIZACJA: lepsze balansowanie kar)

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class PriceOnlyMSELoss(nn.Module):
    """Oblicza błąd MSE wyłącznie dla kolumny z ceną."""
    def __init__(self, price_index: int):
        super().__init__()
        self.mse = nn.MSELoss()
        self.price_index = price_index
        logger.info(f"Inicjalizacja PriceOnlyMSELoss dla indeksu cechy: {self.price_index}")

    def forward(self, y_pred_scaled: torch.Tensor, y_true_target_scaled: torch.Tensor):
        pred_price = y_pred_scaled[:, self.price_index]
        return self.mse(pred_price, y_true_target_scaled)


class DirectionalAccuracyLoss(nn.Module):
    """
    Hybrydowa funkcja straty, która łączy:
    1. Błąd wartości (MSE)
    2. Błąd kierunku
    3. Karę za kopiowanie poprzedniej wartości (ANTY-MAŁPOWANIE)
    
    WERSJA: 55.0 - ZOPTYMALIZOWANE balansowanie kar
    """
    def __init__(self, initial_weight_mse: float = 0.6, initial_weight_dir: float = 0.1, 
                 use_log_mse: bool = False, anti_copy_weight: float = 0.3):
        super().__init__()
        self.mse_loss_fn = nn.MSELoss()
        self.weight_mse = initial_weight_mse
        self.weight_dir = initial_weight_dir
        self.use_log_mse = use_log_mse
        self.anti_copy_weight = anti_copy_weight
        
        logger.info(
            f"Inicjalizacja DirectionalAccuracyLoss. "
            f"Wagi: MSE={self.weight_mse:.2f}, Kierunek={self.weight_dir:.2f}, "
            f"Anty-Kopiowanie={self.anti_copy_weight:.2f}"
        )

    def forward(self, y_pred_scaled: torch.Tensor, y_true_scaled: torch.Tensor, 
                y_pred_unscaled_price: torch.Tensor, 
                y_true_unscaled_price: torch.Tensor, 
                y_prev_unscaled_price: torch.Tensor):
        
        # ========== 1. BŁĄD WARTOŚCI (MSE) ==========
        raw_mse_loss = self.mse_loss_fn(y_pred_scaled, y_true_scaled)
        loss_mse = torch.log(1.0 + raw_mse_loss) if self.use_log_mse else raw_mse_loss

        # ========== 2. BŁĄD KIERUNKU ==========
        pred_dir = torch.sign(y_pred_unscaled_price - y_prev_unscaled_price)
        actual_dir = torch.sign(y_true_unscaled_price - y_prev_unscaled_price)
        
        mask = (actual_dir != 0)
        directional_agreement = (pred_dir[mask] * actual_dir[mask])
        
        if directional_agreement.numel() > 0:
            loss_directional = (1.0 - directional_agreement).mean() / 2.0
        else:
            loss_directional = torch.tensor(0.0, device=y_pred_scaled.device)

        # ========== 3. KARA ZA KOPIOWANIE ==========
        if self.anti_copy_weight > 0:
            pred_delta = y_pred_unscaled_price - y_prev_unscaled_price
            relative_change = torch.abs(pred_delta) / (torch.abs(y_prev_unscaled_price) + 1e-8)
            
            # Wykładnik -800 dla BTC (sprawdzony dla cen ~110k USDT)
            # exp(-800 * 0.0001) = 0.92 ✅
            # exp(-800 * 0.001) = 0.45 ✅
            # exp(-800 * 0.01) = 0.0003 ✅
            copy_penalty = torch.exp(-800.0 * relative_change).mean()
        else:
            copy_penalty = torch.tensor(0.0, device=y_pred_scaled.device)

        # ========== ŁĄCZENIE WSZYSTKICH KOMPONENTÓW ==========
        total_loss = (
            self.weight_mse * loss_mse + 
            self.weight_dir * loss_directional + 
            self.anti_copy_weight * copy_penalty
        )
        
        return total_loss, loss_mse, loss_directional


class IndicatorDirectionalLoss(nn.Module):
    """
    Uniwersalna funkcja straty dla prognozowania WSKAŹNIKÓW (np. KAMA, TEMA).
    """
    def __init__(self, initial_weight_mse: float = 0.6, initial_weight_dir: float = 0.1, 
                 use_log_mse: bool = False, anti_copy_weight: float = 0.3):
        super().__init__()
        self.mse_loss_fn = nn.MSELoss()
        self.weight_mse = initial_weight_mse
        self.weight_dir = initial_weight_dir
        self.use_log_mse = use_log_mse
        self.anti_copy_weight = anti_copy_weight
        
        logger.info(
            f"Inicjalizacja IndicatorDirectionalLoss. "
            f"Wagi: MSE={self.weight_mse:.2f}, Kierunek={self.weight_dir:.2f}, "
            f"Anty-Kopiowanie={self.anti_copy_weight:.2f}"
        )

    def forward(self, 
                y_pred_indicator_scaled: torch.Tensor, 
                y_true_indicator_scaled: torch.Tensor, 
                y_pred_indicator_unscaled: torch.Tensor,
                y_true_indicator_unscaled: torch.Tensor,
                y_prev_indicator_unscaled: torch.Tensor):
        
        # ========== 1. MSE ==========
        raw_mse_loss = self.mse_loss_fn(y_pred_indicator_scaled, y_true_indicator_scaled)
        loss_mse = torch.log(1.0 + raw_mse_loss) if self.use_log_mse else raw_mse_loss

        # ========== 2. KIERUNEK ==========
        pred_dir = torch.sign(y_pred_indicator_unscaled - y_prev_indicator_unscaled)
        actual_dir = torch.sign(y_true_indicator_unscaled - y_prev_indicator_unscaled)
        
        mask = (actual_dir != 0)
        directional_agreement = (pred_dir[mask] == actual_dir[mask]).float()
        
        if directional_agreement.numel() > 0:
            loss_directional = 1.0 - directional_agreement.mean()
        else:
            loss_directional = torch.tensor(0.0, device=y_pred_indicator_scaled.device)

        # ========== 3. KARA ZA KOPIOWANIE ==========
        if self.anti_copy_weight > 0:
            pred_delta = y_pred_indicator_unscaled - y_prev_indicator_unscaled
            relative_change = torch.abs(pred_delta) / (torch.abs(y_prev_indicator_unscaled) + 1e-8)
            copy_penalty = torch.exp(-800.0 * relative_change).mean()
        else:
            copy_penalty = torch.tensor(0.0, device=y_pred_indicator_scaled.device)

        # ========== ŁĄCZENIE ==========
        total_loss = (
            self.weight_mse * loss_mse + 
            self.weight_dir * loss_directional +
            self.anti_copy_weight * copy_penalty
        )
        
        return total_loss, loss_mse, loss_directional
