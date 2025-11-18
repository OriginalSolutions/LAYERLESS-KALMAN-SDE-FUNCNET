# plik: learning_engine.py
# WERSJA: 102.10 (FINAL, COMPLETE & WORKING)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from torchsde import sdeint
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from typing import Dict, List
import gc
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CyclicLR

from custom_loss import DirectionalAccuracyLoss
from sde_models import (
    FeatureAwareSDE, HybridAttentionSDE, 
    KalmanFeatureAwareSDE, KalmanHybridAttentionSDE, HybridAttentionSDE_WithKalmanAttention,
    KALMANGuidedAttentionSDE
)

try:
    from adan_pytorch import Adan
    ADAN_AVAILABLE = True
except ImportError:
    ADAN_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class SdeSequenceDataset(Dataset):
    def __init__(self, X_full_state, y_target, y_price, prev_price, seq_length):
        self.X_full_state = X_full_state
        self.y_target = y_target
        self.y_price = y_price
        self.prev_price = prev_price
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.X_full_state) - self.seq_length
    
    def __getitem__(self, idx):
        x_end = idx + self.seq_length
        return (
            torch.from_numpy(self.X_full_state[idx:x_end]), 
            torch.tensor(self.y_target[x_end]), 
            torch.tensor(self.y_price[x_end]), 
            torch.tensor(self.prev_price[x_end])
        )

class PureFunctionalOnlineLearningEngine:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.system_device)
        self.feature_order = config.model_feature_columns
        self.target_feature_name = getattr(config, 'model_target_feature', 'log_returns')
        
        if self.target_feature_name not in self.feature_order or 'prices' not in self.feature_order:
            raise ValueError("`prices` i cel modelu (`model_target_feature`) muszą być w `model_feature_columns`!")
        
        self.target_index = self.feature_order.index(self.target_feature_name)
        self.price_index = self.feature_order.index('prices')
        
        self.main_imputer = SimpleImputer(strategy='mean')
        self.ai_feature_imputer = SimpleImputer(strategy='constant', fill_value=0.0)
        
        # Inicjalizujemy listy cech, ale `feature_order` może się zmienić po pierwszym dopasowaniu
        self.ai_feature_names = [col for col in self.feature_order if col.startswith('ai_')]
        self.main_feature_names = [col for col in self.feature_order if not col.startswith('ai_')]
        
        model_type = config.model_type
        attention_settings = getattr(config, 'attention_settings', {})
        kalman_settings = getattr(config, 'kalman_filter_settings', {})
        aug_config_dict = {**config.state_augmentation_settings, 'feature_names': self.feature_order}
        
        logger.info("=" * 60)
        logger.info("====== Inicjalizacja Głównego Modelu Predykcyjnego ======")

        model_args = (
            len(self.feature_order), 
            config.sde_stability_settings, 
            config.stability_controls['feature_weights_clamp_range'], 
            aug_config_dict
        )
        
        if model_type == 'kalman_feature_aware_sde':
            logger.info("-> Tryb Modelu: Kalman SDE (bez atencji)")
            self.model = KalmanFeatureAwareSDE(*model_args, attention_settings, kalman_settings).to(self.device)
        elif model_type == 'kalman_hybrid_attention_sde':
            logger.info("-> Tryb Modelu: Kalman SDE z Atencją (filtrowanie dryfu)")
            self.model = KalmanHybridAttentionSDE(*model_args, attention_settings, kalman_settings).to(self.device)
        elif model_type == 'hybrid_attention_sde_with_kalman_attention':
            logger.info("-> Tryb Modelu: SDE z Atencją Kalmana (filtrowanie atencji)")
            self.model = HybridAttentionSDE_WithKalmanAttention(*model_args, attention_settings, kalman_settings).to(self.device)
        elif model_type == 'kalman_guided_attention_sde':
            logger.info("-> Tryb Modelu: NOWY (KALMANGuidedAttentionSDE - Kalman jako przewodnik)")
            self.model = KALMANGuidedAttentionSDE(*model_args, attention_settings, kalman_settings).to(self.device)
        elif model_type == 'hybrid_attention_sde' and attention_settings.get('enabled', False):
            logger.info("-> Tryb Modelu: Zaawansowany (HybridAttentionSDE)")
            self.model = HybridAttentionSDE(*model_args, attention_settings).to(self.device)
        else:
            if model_type not in ['feature_aware_sde', 'hybrid_attention_sde']:
                logger.warning(f"-> Nieznany lub niekompatybilny typ modelu: '{model_type}'. Używam standardowego 'FeatureAwareSDE'.")
            logger.info("-> Tryb Modelu: Standardowy (FeatureAwareSDE)")
            self.model = FeatureAwareSDE(*model_args).to(self.device)
        
        num_features = len(self.feature_order)
        params_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"-> Liczba cech (kolumn): {num_features}")
        logger.info(f"-> Nazwy cech: {self.feature_order}")
        logger.info(f"-> Liczba parametrów do trenowania: {params_count}")
        logger.info(f"-> Target: {self.target_feature_name} (indeks {self.target_index})")
        logger.info("=" * 60)

        self.optimizer = self._create_optimizer(config.training_initial_lr)
        self.scaler = RobustScaler()
        self.is_trained = False
        
        loss_settings = getattr(config, 'loss_settings', {})
        if loss_settings.get('name') == 'DirectionalAccuracyLoss':
            self.criterion = DirectionalAccuracyLoss(
                loss_settings.get('initial_weight_mse', 0.6),
                loss_settings.get('initial_weight_dir', 0.1),
                loss_settings.get('use_log_mse', True),
                loss_settings.get('anti_copy_weight', 0.3)
            )
            logger.info(f"✅ Funkcja straty: DirectionalAccuracyLoss (MSE={loss_settings.get('initial_weight_mse')}, DIR={loss_settings.get('initial_weight_dir')}, ANTI-COPY={loss_settings.get('anti_copy_weight')})")
        else:
            self.criterion = nn.MSELoss()
            logger.info(f"✅ Funkcja straty: MSELoss (standard)")
            
        logger.info(f"✅ Silnik gotowy. Cel: '{self.target_feature_name}', Strata: {type(self.criterion).__name__}")
        # print(self.model) # Usunięto lub zakomentowano linię debugującą

    def _create_optimizer(self, lr: float) -> torch.optim.Optimizer:
        opt_settings = getattr(self.config, 'optimizer_settings', {})
        if ADAN_AVAILABLE and opt_settings.get('name') == 'Adan':
            return Adan(
                self.model.parameters(), 
                lr=lr, 
                betas=tuple(opt_settings.get('betas', (0.9,0.99,0.999))), 
                weight_decay=opt_settings.get('weight_decay', 0.1)
            )
        return torch.optim.AdamW(self.model.parameters(), lr=lr)

    def _prepare_full_state(self, X_df: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        # Bezpiecznie wybieramy tylko te kolumny z self.feature_order, które faktycznie istnieją w X_df
        available_cols = [col for col in self.feature_order if col in X_df.columns]
        X_reordered = X_df[available_cols].astype(np.float32)

        # Dynamicznie określamy, które cechy są główne, a które AI, na podstawie dostępnych kolumn
        current_main_features = [col for col in available_cols if not col.startswith('ai_')]
        current_ai_features = [col for col in available_cols if col.startswith('ai_')]

        X_main = X_reordered[current_main_features]
        X_ai = X_reordered[current_ai_features]
        
        imputed_main = self.main_imputer.fit_transform(X_main) if fit_scaler else self.main_imputer.transform(X_main)
        
        if not X_ai.empty:
            imputed_ai = self.ai_feature_imputer.fit_transform(X_ai) if fit_scaler else self.ai_feature_imputer.transform(X_ai)
            imputed_full = np.concatenate([imputed_main, imputed_ai], axis=1)
            final_columns = current_main_features + current_ai_features
            imputed_df = pd.DataFrame(imputed_full, columns=final_columns, index=X_df.index)
        else:
            final_columns = current_main_features
            imputed_df = pd.DataFrame(imputed_main, columns=final_columns, index=X_df.index)
        
        # Aktualizujemy feature_order po pierwszym dopasowaniu, aby odzwierciedlał rzeczywisty porządek
        if fit_scaler:
            self.feature_order = final_columns
            return self.scaler.fit_transform(imputed_df)
            
        return self.scaler.transform(imputed_df)


    def _run_training_loop(self, dataset: SdeSequenceDataset, epochs: int, lr: float):
        self.optimizer = self._create_optimizer(lr)
        dataloader = DataLoader(dataset, batch_size=self.config.training_batch_size, shuffle=True, num_workers=0)
        self.model.train()
        sde_options = {'rtol': self.config.sde_solver_rtol, 'atol': self.config.sde_solver_atol}
        
        initial_w_mse = self.config.loss_settings.get('initial_weight_mse', 0.6)
        initial_w_dir = self.config.loss_settings.get('initial_weight_dir', 0.1)
        final_w_mse = self.config.loss_settings.get('final_weight_mse', initial_w_dir)
        final_w_dir = self.config.loss_settings.get('final_weight_dir', initial_w_mse)
        
        scheduler, scheduler_name = None, None
        opt_settings = self.config.optimizer_settings
        if opt_settings.get('use_lr_scheduler', False):
            scheduler_name = opt_settings.get('lr_scheduler_name', 'StepLR')
            params = opt_settings.get('lr_scheduler_params', {})
            logger.info(f"Aktywowano scheduler kroku uczenia: {scheduler_name}")
            
            if scheduler_name == 'StepLR': 
                scheduler = StepLR(self.optimizer, step_size=params.get('step_size', 1), gamma=params.get('gamma', 0.9))
            elif scheduler_name == 'CosineAnnealingLR': 
                scheduler = CosineAnnealingLR(self.optimizer, T_max=params.get('T_max', epochs), eta_min=params.get('eta_min', 0))
            elif scheduler_name == 'CyclicalLR': 
                scheduler = CyclicLR(self.optimizer, base_lr=params.get('base_lr', 1e-7), max_lr=params.get('max_lr', 1e-4), step_size_up=params.get('step_size_up', 2000), mode='triangular')

        for epoch in range(epochs):
            total_loss, total_mse, total_dir_loss = 0.0, 0.0, 0.0
            max_batches = self.config.max_batches_per_epoch
            if max_batches is None or max_batches == float('inf'):
                num_batches = len(dataloader)
            else:
                num_batches = min(len(dataloader), int(max_batches))
            
            midpoint_batch = num_batches // 2
            
            for i, (x_batch, y_target_batch, y_price_batch, prev_price_batch) in enumerate(dataloader):
                if i >= num_batches: 
                    break
                
                if i < midpoint_batch: 
                    w_mse, w_dir = initial_w_mse, initial_w_dir
                else:
                    progress = (i - midpoint_batch) / (num_batches - midpoint_batch - 1) if num_batches - midpoint_batch > 1 else 1.0
                    w_mse = initial_w_mse + progress * (final_w_mse - initial_w_mse)
                    w_dir = initial_w_dir + progress * (final_w_dir - initial_w_dir)
                
                if isinstance(self.criterion, DirectionalAccuracyLoss): 
                    self.criterion.weight_mse, self.criterion.weight_dir = w_mse, w_dir
                
                x_batch, y_target_batch, y_price_batch, prev_price_batch = [
                    t.to(self.device).float() for t in [x_batch, y_target_batch, y_price_batch, prev_price_batch]
                ]
                
                self.optimizer.zero_grad()
                y0 = x_batch[:, -1, :]
                t = torch.linspace(0, 1, 2, device=self.device)
                predicted_y_scaled = sdeint(self.model, y0, t, method=self.config.sde_solver_method, options=sde_options)[1]
                
                y_batch_for_scaler = np.zeros((len(y_target_batch), len(self.feature_order)), dtype=np.float32)
                y_batch_for_scaler[:, self.target_index] = y_target_batch.cpu().numpy()
                y_batch_target_scaled = torch.from_numpy(self.scaler.transform(y_batch_for_scaler)).float().to(self.device)

                loss, mse_loss, dir_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
                if isinstance(self.criterion, DirectionalAccuracyLoss):
                    pred_unscaled_full = self.scaler.inverse_transform(predicted_y_scaled.cpu().detach().numpy())
                    pred_price_t = torch.from_numpy(pred_unscaled_full[:, self.price_index]).float().to(self.device)
                    loss, mse_loss, dir_loss = self.criterion(predicted_y_scaled, y_batch_target_scaled, pred_price_t, y_price_batch, prev_price_batch)
                else:
                    loss = self.criterion(predicted_y_scaled, y_batch_target_scaled)
                    mse_loss = loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if scheduler and scheduler_name == 'CyclicalLR': 
                    scheduler.step()
                
                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_dir_loss += dir_loss.item()
                
                if (i + 1) % self.config.training_log_frequency == 0:
                     current_lr = self.optimizer.param_groups[0]['lr']
                     dir_accuracy_percent = (1.0 - dir_loss.item()) * 100 if dir_loss.item() >= 0 else 0.0
                     log_msg = (
                         f"  Epoka [{epoch+1}/{epochs}], Batch [{i+1}/{num_batches}] | "
                         f"LR: {current_lr:.2e} | Wagi(M/D): {w_mse:.2f}/{w_dir:.2f} | "
                         f"Strata: {loss.item():.4f} | MSE: {mse_loss.item():.4f} | "
                         f"Błąd Kier.: {dir_loss.item():.3f} | Trafność Kier.: {dir_accuracy_percent:.1f}%"
                     )
                     logger.info(log_msg)

            if scheduler and scheduler_name in ['StepLR', 'CosineAnnealingLR']: 
                scheduler.step()
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            avg_mse = total_mse / num_batches if num_batches > 0 else 0
            avg_dir_loss = total_dir_loss / num_batches if num_batches > 0 else 0
            avg_dir_accuracy = (1.0 - avg_dir_loss) * 100 if avg_dir_loss >= 0 else 0.0
            final_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info("-" * 60)
            logger.info(f"PODSUMOWANIE Epoki [{epoch+1}/{epochs}]")
            logger.info(f"  -> Średnia Strata Całkowita: {avg_loss:.4f}")
            logger.info(f"  -> Średni Błąd Wartości (MSE): {avg_mse:.4f}")
            logger.info(f"  -> Średni Błąd Kierunkowy: {avg_dir_loss:.3f}")
            logger.info(f"  -> Średnia Trafność Kierunkowa: {avg_dir_accuracy:.1f}%")
            logger.info(f"  -> Krok Uczenia (LR) na końcu epoki: {final_lr:.2e}")
            logger.info("-" * 60)
            gc.collect()

    def run_training(self, X_train: pd.DataFrame, epochs: int, lr: float, is_initial: bool):
        log_prefix = 'trening początkowy' if is_initial else 'dotrenowanie'
        logger.info(f"Rozpoczynam {log_prefix} na {len(X_train)} próbkach...")
        
        if X_train.empty: 
            logger.error("Brak danych. Pomijam trening.")
            return
        
        # Tworzymy targety. Używamy .get() dla bezpieczeństwa, na wypadek braku kolumny
        if self.target_feature_name == 'log_returns':
            y_target_series = X_train.get('log_returns', pd.Series()).shift(-1)
        elif self.target_feature_name == 'price_delta':
            y_target_series = X_train.get('prices', pd.Series()).shift(-1) - X_train.get('prices', pd.Series())
        else:
            y_target_series = X_train.get(self.target_feature_name, pd.Series()).shift(-1)
        
        y_price_series = X_train.get('prices', pd.Series()).shift(-1)
        prev_price_series = X_train.get('prices', pd.Series()).copy()
        
        df_for_training = X_train.assign(
            y_target=y_target_series, 
            y_price=y_price_series, 
            prev_price=prev_price_series
        ).dropna()
        
        X_final = df_for_training.drop(columns=['y_target', 'y_price', 'prev_price'])
        
        dataset = SdeSequenceDataset(
            self._prepare_full_state(X_final, fit_scaler=is_initial), 
            df_for_training['y_target'].values, 
            df_for_training['y_price'].values, 
            df_for_training['prev_price'].values, 
            self.config.prediction_input_history_minutes
        )
        
        if len(dataset) > 0:
            self._run_training_loop(dataset, epochs, lr)
            self.is_trained = True
        else: 
            logger.error("Za mało danych do stworzenia sekwencji. Pomijam trening.")

    def predict(self, X_inference_df: pd.DataFrame) -> float:
        if not self.is_trained: 
            raise RuntimeError("Model musi być wytrenowany.")
        
        X_full_state = self._prepare_full_state(X_inference_df, fit_scaler=False)
        y0 = torch.from_numpy(X_full_state[-1, :]).float().to(self.device).unsqueeze(0)
        self.model.eval()
        
        with torch.no_grad():
            t = torch.linspace(0, self.config.prediction_horizon_minutes, self.config.sde_solver_time_steps + 1, device=self.device)
            predicted_trajectory_scaled = sdeint(self.model, y0, t)
        
        predicted_unscaled = self.scaler.inverse_transform(predicted_trajectory_scaled.squeeze(1).cpu().numpy())
        predicted_target_value = predicted_unscaled[-1, self.target_index]
        
        return float(predicted_target_value)
    
    def extract_ai_features(self) -> Dict[str, float]:
        """
        Wyciąga wytrenowane wagi z modeli bazowych jako cechy AI.
        Jeśli model jest zbyt złożony i nie ma prostej struktury wag,
        cicho zwraca pusty słownik.
        """
        if not self.is_trained:
            return {}
        
        self.model.eval()
        features = {}
        
        with torch.no_grad():
            # Sprawdzamy, czy model ma atrybut `feature_weights` (dla dryftu w FeatureAwareSDE)
            if hasattr(self.model, 'feature_weights'):
                drift_weights = self.model.feature_weights.squeeze().cpu().numpy()
                if drift_weights.ndim == 2:
                    drift_weights = np.diag(drift_weights)
                
                for i, feature_name in enumerate(self.main_feature_names):
                    if i < len(drift_weights):
                        features[f'ai_drift_{feature_name}'] = drift_weights[i]

            # Sprawdzamy, czy model ma atrybut `sigma_adapt_weight` (dla dyfuzji w FeatureAwareSDE)
            if hasattr(self.model, 'sigma_adapt_weight'):
                diffusion_weights = self.model.sigma_adapt_weight.squeeze().cpu().numpy()
                if diffusion_weights.ndim == 2:
                    diffusion_weights = np.diag(diffusion_weights)

                for i, feature_name in enumerate(self.main_feature_names):
                    if i < len(diffusion_weights):
                        features[f'ai_diffusion_{feature_name}'] = diffusion_weights[i]
        
        if features:
            logger.info(f"Pomyślnie wyekstrahowano {len(features)} cech AI z modelu bazowego.")
        
        return features