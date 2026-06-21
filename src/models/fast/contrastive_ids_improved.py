"""
Improved Contrastive IDS model with proper calibration.

Fixes:
1. Focal loss for better class imbalance handling
2. Temperature scaling for confidence calibration
3. Label smoothing to prevent overconfidence
4. Proper validation during training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple

from .base import FastModel
from ..base import ModelOutput


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class TemperatureScaling(nn.Module):
    """Temperature scaling for confidence calibration."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


class ImprovedContrastiveEncoder(nn.Module):
    """Improved encoder with residual connections and layer norm."""

    def __init__(
        self,
        n_features: int,
        hidden_dims: list = [256, 128],
        embed_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_proj = nn.Linear(n_features, hidden_dims[0])
        self.input_norm = nn.LayerNorm(hidden_dims[0])

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            self.layers.append(nn.Linear(in_dim, h_dim))
            self.norms.append(nn.LayerNorm(h_dim))
            self.dropouts.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.output_proj = nn.Linear(in_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.relu(x)

        for layer, norm, dropout in zip(self.layers, self.norms, self.dropouts):
            residual = x if x.shape[-1] == layer.out_features else None
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
            x = dropout(x)
            if residual is not None:
                x = x + residual

        x = self.output_proj(x)
        return F.normalize(x, dim=-1)


class ImprovedContrastiveIDSNetwork(nn.Module):
    """
    Improved Contrastive IDS network with:
    - Focal loss for class imbalance
    - Temperature scaling for calibration
    - Better regularization
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        embed_dim: int = 64,
        hidden_dims: list = [256, 128],
        temperature: float = 0.07,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.n_classes = n_classes

        self.encoder = ImprovedContrastiveEncoder(n_features, hidden_dims, embed_dim)

        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 32),
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, n_classes),
        )

        # Temperature scaling for calibration
        self.temp_scaling = TemperatureScaling()

        # For MC Dropout uncertainty
        self.mc_dropout = nn.Dropout(0.3)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
        calibrated: bool = True,
    ) -> Dict[str, torch.Tensor]:
        z = self.encoder(x)
        logits = self.classifier(z)

        # Apply temperature scaling for calibrated probabilities
        if calibrated:
            scaled_logits = self.temp_scaling(logits)
        else:
            scaled_logits = logits

        probs = F.softmax(scaled_logits, dim=-1)

        # MC Dropout based uncertainty
        z_dropout = self.mc_dropout(z)
        logits_dropout = self.classifier(z_dropout)
        probs_dropout = F.softmax(self.temp_scaling(logits_dropout) if calibrated else logits_dropout, dim=-1)
        uncertainty = torch.abs(probs - probs_dropout).max(dim=-1, keepdim=True)[0]

        output = {
            'logits': logits,
            'scaled_logits': scaled_logits,
            'probs': probs,
            'uncertainty': uncertainty,
        }

        if return_embeddings:
            output['embeddings'] = z
            output['projections'] = F.normalize(self.projection(z), dim=-1)

        return output

    def mc_forward(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
        calibrated: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo forward pass for uncertainty estimation."""
        self.train()  # Enable dropout

        all_probs = []
        with torch.no_grad():
            for _ in range(n_samples):
                z = self.encoder(x)
                z = self.mc_dropout(z)
                logits = self.classifier(z)
                if calibrated:
                    logits = self.temp_scaling(logits)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs)

        all_probs = torch.stack(all_probs, dim=0)
        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)
        uncertainty = std_probs.max(dim=-1)[0]  # Max std across classes

        return mean_probs, uncertainty

    def contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Supervised contrastive loss."""
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / self.temperature

        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(z.device)
            mask = mask.repeat(2, 2)
        else:
            mask = torch.eye(batch_size, device=z.device)
            mask = mask.repeat(2, 2)
            mask[:batch_size, batch_size:] = torch.eye(batch_size, device=z.device)
            mask[batch_size:, :batch_size] = torch.eye(batch_size, device=z.device)

        logits_mask = 1 - torch.eye(2 * batch_size, device=z.device)
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        mask_sum = torch.clamp(mask.sum(dim=1), min=1)
        loss = -(mask * log_prob).sum(dim=1) / mask_sum

        return loss.mean()


class ImprovedContrastiveIDSModel(FastModel):
    """
    Improved ContrastiveIDS with proper calibration and class balancing.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        embed_dim: int = 64,
        device: str = 'cuda',
        mc_samples: int = 10,
    ):
        super().__init__(n_features)
        self.n_classes = n_classes
        self.mc_samples = mc_samples
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.network = ImprovedContrastiveIDSNetwork(
            n_features=n_features,
            n_classes=n_classes,
            embed_dim=embed_dim,
        ).to(self.device)

        self._class_means = None
        self._calibrated = False

    def step(self, x: np.ndarray) -> ModelOutput:
        x_tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(self.device)

        # Use MC forward for better uncertainty
        probs, uncertainty = self.network.mc_forward(x_tensor, self.mc_samples)

        probs_np = probs.cpu().numpy()[0]
        uncertainty_np = uncertainty.cpu().numpy()[0]
        pred_class = np.argmax(probs_np)

        # Confidence: max probability
        confidence = np.max(probs_np)

        # Anomaly score: entropy + uncertainty
        entropy = -np.sum(probs_np * np.log(probs_np + 1e-8))
        anomaly_score = entropy + uncertainty_np

        return ModelOutput(
            prediction=np.array([pred_class]),
            uncertainty=float(uncertainty_np),
            anomaly_score=float(anomaly_score),
            metadata={
                'confidence': float(confidence),
                'probs': probs_np.tolist(),
            }
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        contrastive_epochs: int = 20,
        val_split: float = 0.15,
    ) -> 'ImprovedContrastiveIDSModel':
        """Three-stage training with calibration."""
        if X.ndim == 3:
            X = X[:, -1, :]

        # Train/val split
        n_val = int(len(X) * val_split)
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        X_train = torch.from_numpy(X[train_idx].astype(np.float32)).to(self.device)
        y_train = torch.from_numpy(y[train_idx].astype(np.int64)).to(self.device)
        X_val = torch.from_numpy(X[val_idx].astype(np.float32)).to(self.device)
        y_val = torch.from_numpy(y[val_idx].astype(np.int64)).to(self.device)

        # Stage 1: Contrastive pre-training
        print("Stage 1: Contrastive pre-training...")
        self._contrastive_pretrain(X_train, y_train, contrastive_epochs, batch_size, learning_rate)

        # Stage 2: Supervised fine-tuning with focal loss
        print("Stage 2: Supervised fine-tuning with focal loss...")
        self._supervised_finetune(X_train, y_train, X_val, y_val, epochs - contrastive_epochs, batch_size, learning_rate * 0.1)

        # Stage 3: Temperature scaling calibration
        print("Stage 3: Temperature calibration...")
        self._calibrate_temperature(X_val, y_val)

        self._compute_class_centroids(X_train, y_train)
        self.is_fitted = True
        self._calibrated = True
        return self

    def _contrastive_pretrain(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int,
        batch_size: int,
        lr: float,
    ):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)

        for epoch in range(epochs):
            self.network.train()
            total_loss = 0.0
            n_batches = 0

            indices = torch.randperm(len(X))
            for i in range(0, len(X), batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_X = X[batch_idx]
                batch_y = y[batch_idx]

                # Data augmentation
                noise1 = torch.randn_like(batch_X) * 0.1
                noise2 = torch.randn_like(batch_X) * 0.1
                mask = torch.bernoulli(torch.ones_like(batch_X) * 0.9)

                x1 = batch_X + noise1
                x2 = batch_X * mask + noise2

                out1 = self.network(x1, return_embeddings=True, calibrated=False)
                out2 = self.network(x2, return_embeddings=True, calibrated=False)

                z1 = out1['projections']
                z2 = out2['projections']

                loss = self.network.contrastive_loss(z1, z2, batch_y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0:
                print(f"  Contrastive epoch {epoch+1}: loss={total_loss/n_batches:.4f}")

    def _supervised_finetune(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int,
        batch_size: int,
        lr: float,
    ):
        # Focal loss with class weighting
        class_counts = torch.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts.float())
        class_weights = class_weights.to(self.device)

        # Use focal loss for better class imbalance handling
        focal_loss = FocalLoss(gamma=2.0)
        ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        optimizer = torch.optim.AdamW([
            {'params': self.network.encoder.parameters(), 'lr': lr * 0.1},
            {'params': self.network.classifier.parameters(), 'lr': lr},
        ], weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0.0
        patience = 20
        patience_counter = 0

        for epoch in range(epochs):
            self.network.train()
            total_loss = 0.0
            correct = 0
            total = 0

            indices = torch.randperm(len(X_train))
            for i in range(0, len(X_train), batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_X = X_train[batch_idx]
                batch_y = y_train[batch_idx]

                output = self.network(batch_X, calibrated=False)

                # Combined loss: focal + cross entropy
                loss = 0.5 * focal_loss(output['logits'], batch_y) + 0.5 * ce_loss(output['logits'], batch_y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                pred = output['logits'].argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += len(batch_y)

            scheduler.step()

            # Validation
            self.network.eval()
            with torch.no_grad():
                val_output = self.network(X_val, calibrated=False)
                val_pred = val_output['logits'].argmax(dim=1)
                val_acc = (val_pred == y_val).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 10 == 0:
                train_acc = correct / total
                print(f"  Finetune epoch {epoch+1}: loss={total_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    def _calibrate_temperature(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        n_iterations: int = 100,
    ):
        """Learn optimal temperature for calibration."""
        self.network.eval()

        # Freeze all except temperature
        for param in self.network.parameters():
            param.requires_grad = False
        self.network.temp_scaling.temperature.requires_grad = True

        optimizer = torch.optim.LBFGS([self.network.temp_scaling.temperature], lr=0.01, max_iter=n_iterations)
        nll_criterion = nn.CrossEntropyLoss()

        def eval_step():
            optimizer.zero_grad()
            output = self.network(X_val, calibrated=True)
            loss = nll_criterion(output['scaled_logits'], y_val)
            loss.backward()
            return loss

        optimizer.step(eval_step)

        # Unfreeze
        for param in self.network.parameters():
            param.requires_grad = True

        print(f"  Learned temperature: {self.network.temp_scaling.temperature.item():.4f}")

        # Evaluate calibration
        with torch.no_grad():
            output = self.network(X_val, calibrated=True)
            probs = output['probs']
            preds = probs.argmax(dim=1)
            confidences = probs.max(dim=1)[0]

            correct = (preds == y_val).float()
            accuracy = correct.mean().item()

            # Expected Calibration Error
            n_bins = 10
            bin_boundaries = torch.linspace(0, 1, n_bins + 1).to(self.device)
            ece = 0.0
            for i in range(n_bins):
                in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
                if in_bin.sum() > 0:
                    bin_acc = correct[in_bin].mean()
                    bin_conf = confidences[in_bin].mean()
                    ece += in_bin.float().mean() * torch.abs(bin_acc - bin_conf)

            print(f"  Post-calibration: accuracy={accuracy:.4f}, ECE={ece.item():.4f}")

    def _compute_class_centroids(self, X: torch.Tensor, y: torch.Tensor):
        self.network.eval()
        with torch.no_grad():
            embeddings = self.network.get_embeddings(X).cpu().numpy()
            y_np = y.cpu().numpy()

        self._class_means = {}
        for c in range(self.n_classes):
            mask = y_np == c
            if mask.sum() > 0:
                self._class_means[c] = embeddings[mask].mean(axis=0)

    def predict(self, X: np.ndarray) -> ModelOutput:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim == 3:
            X = X[:, -1, :]

        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)

        probs, uncertainty = self.network.mc_forward(X_tensor, self.mc_samples)

        probs_np = probs.cpu().numpy()
        predictions = np.argmax(probs_np, axis=1)
        confidences = np.max(probs_np, axis=1)

        return ModelOutput(
            prediction=predictions,
            uncertainty=float(uncertainty.mean().cpu().numpy()),
            metadata={
                'confidences': confidences.tolist(),
                'probs': probs_np.tolist(),
            }
        )

    def get_calibrated_confidences(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get calibrated predictions and confidences."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim == 3:
            X = X[:, -1, :]

        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)

        probs, uncertainty = self.network.mc_forward(X_tensor, self.mc_samples)

        probs_np = probs.cpu().numpy()
        predictions = np.argmax(probs_np, axis=1)
        confidences = np.max(probs_np, axis=1)
        uncertainties = uncertainty.cpu().numpy()

        return predictions, confidences, uncertainties

    def reset_state(self):
        pass

    def get_complexity(self) -> Dict[str, Any]:
        n_params = sum(p.numel() for p in self.network.parameters())
        return {
            'n_parameters': n_params,
            'memory_bytes': n_params * 4,
            'architecture': 'ImprovedContrastiveIDS',
        }

    def update(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        return {
            'model_state_dict': self.network.state_dict(),
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'class_means': self._class_means,
            'calibrated': self._calibrated,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.network.load_state_dict(state['model_state_dict'])
        self._class_means = state.get('class_means')
        self._calibrated = state.get('calibrated', False)
        self.is_fitted = True
