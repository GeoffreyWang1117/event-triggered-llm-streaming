"""
Contrastive Learning-based Network Intrusion Detection.
Based on SOTA: Contrastive Learning + Classifier approaches.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple

from .base import FastModel
from ..base import ModelOutput


class ContrastiveEncoder(nn.Module):
    """
    Encoder network for contrastive learning.
    Projects input features to a normalized embedding space.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: list = [256, 128],
        embed_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, embed_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_features)
        Returns:
            Normalized embeddings (batch, embed_dim)
        """
        z = self.encoder(x)
        return F.normalize(z, dim=-1)


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""

    def __init__(self, embed_dim: int, proj_dim: int = 32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class ContrastiveIDSNetwork(nn.Module):
    """
    Contrastive Learning + Classifier for Intrusion Detection.

    Two-stage training:
    1. Contrastive pre-training to learn good representations
    2. Fine-tuning classifier on labeled data
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        embed_dim: int = 64,
        hidden_dims: list = [256, 128],
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature

        # Encoder
        self.encoder = ContrastiveEncoder(n_features, hidden_dims, embed_dim)

        # Projection head (for contrastive learning)
        self.projection = ProjectionHead(embed_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, n_classes),
        )

        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid(),
        )

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get normalized embeddings."""
        return self.encoder(x)

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, n_features)
        Returns:
            Dictionary with 'logits', 'probs', 'uncertainty', optionally 'embeddings'
        """
        z = self.encoder(x)
        logits = self.classifier(z)
        probs = F.softmax(logits, dim=-1)
        uncertainty = self.uncertainty_head(z)

        output = {
            'logits': logits,
            'probs': probs,
            'uncertainty': uncertainty,
        }

        if return_embeddings:
            output['embeddings'] = z
            output['projections'] = self.projection(z)

        return output

    def contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Supervised contrastive loss (SupCon).

        Args:
            z1, z2: Projected embeddings from two augmented views
            labels: Class labels (optional, for supervised contrastive)
        """
        batch_size = z1.size(0)

        # Concatenate views
        z = torch.cat([z1, z2], dim=0)  # (2*batch, proj_dim)

        # Compute similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature  # (2*batch, 2*batch)

        # Create mask for positive pairs
        if labels is not None:
            # Supervised: positives are same-class samples
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(z.device)
            mask = mask.repeat(2, 2)
        else:
            # Self-supervised: positives are augmented views of same sample
            mask = torch.eye(batch_size, device=z.device)
            mask = mask.repeat(2, 2)
            # Also mark diagonal blocks as positives
            mask[:batch_size, batch_size:] = torch.eye(batch_size, device=z.device)
            mask[batch_size:, :batch_size] = torch.eye(batch_size, device=z.device)

        # Remove self-similarity from denominator
        logits_mask = 1 - torch.eye(2 * batch_size, device=z.device)

        # Compute log-softmax
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive pairs
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.clamp(mask_sum, min=1)
        loss = -(mask * log_prob).sum(dim=1) / mask_sum

        return loss.mean()


class DataAugmentation:
    """Data augmentation for contrastive learning on tabular data."""

    def __init__(
        self,
        noise_std: float = 0.1,
        dropout_prob: float = 0.1,
        mixup_alpha: float = 0.2,
    ):
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.mixup_alpha = mixup_alpha

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate two augmented views of input.

        Args:
            x: (batch, n_features)
        Returns:
            Two augmented versions of x
        """
        # View 1: Gaussian noise
        noise = torch.randn_like(x) * self.noise_std
        x1 = x + noise

        # View 2: Feature dropout + noise
        mask = torch.bernoulli(torch.ones_like(x) * (1 - self.dropout_prob))
        x2 = x * mask + torch.randn_like(x) * self.noise_std * 0.5

        return x1, x2


class ContrastiveIDSModel(FastModel):
    """
    Streaming wrapper for Contrastive IDS.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        embed_dim: int = 64,
        device: str = 'cuda',
    ):
        super().__init__(n_features)
        self.n_classes = n_classes
        self.device = device

        self.network = ContrastiveIDSNetwork(
            n_features=n_features,
            n_classes=n_classes,
            embed_dim=embed_dim,
        ).to(device)

        self.augmentation = DataAugmentation()
        self.optimizer = None

        # Running statistics for anomaly scoring
        self._class_means = None
        self._running_mean = None
        self._running_var = None

    def step(self, x: np.ndarray) -> ModelOutput:
        """Process single sample for streaming inference."""
        x_tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(self.device)

        self.network.eval()
        with torch.no_grad():
            output = self.network(x_tensor, return_embeddings=True)

        probs = output['probs'].cpu().numpy()[0]
        uncertainty = output['uncertainty'].cpu().numpy()[0, 0]
        embeddings = output['embeddings'].cpu().numpy()[0]

        # Predicted class
        pred_class = np.argmax(probs)

        # Anomaly score: distance from class centroid or entropy
        if self._class_means is not None:
            dist = np.linalg.norm(embeddings - self._class_means[pred_class])
            anomaly_score = dist
        else:
            # Use prediction entropy as anomaly score
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            anomaly_score = entropy

        return ModelOutput(
            prediction=np.array([pred_class]),
            uncertainty=float(uncertainty),
            anomaly_score=float(anomaly_score),
            hidden_state=embeddings,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        contrastive_epochs: int = 20,
    ) -> 'ContrastiveIDSModel':
        """
        Two-stage training:
        1. Contrastive pre-training
        2. Supervised fine-tuning
        """
        # Handle 3D input (seq_len, features) -> flatten or use last
        if X.ndim == 3:
            X = X[:, -1, :]  # Use last timestep

        # Convert to tensors
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        y_tensor = torch.from_numpy(y.astype(np.int64)).to(self.device)

        # Stage 1: Contrastive pre-training
        print("Stage 1: Contrastive pre-training...")
        self._contrastive_pretrain(X_tensor, y_tensor, contrastive_epochs, batch_size, learning_rate)

        # Stage 2: Supervised fine-tuning
        print("Stage 2: Supervised fine-tuning...")
        self._supervised_finetune(X_tensor, y_tensor, epochs - contrastive_epochs, batch_size, learning_rate * 0.1)

        # Compute class centroids for anomaly scoring
        self._compute_class_centroids(X_tensor, y_tensor)

        self.is_fitted = True
        return self

    def _contrastive_pretrain(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int,
        batch_size: int,
        lr: float,
    ):
        """Contrastive pre-training stage."""
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        for epoch in range(epochs):
            self.network.train()
            total_loss = 0.0
            n_batches = 0

            indices = torch.randperm(len(X))
            for i in range(0, len(X), batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_X = X[batch_idx]
                batch_y = y[batch_idx]

                # Generate augmented views
                x1, x2 = self.augmentation(batch_X)

                # Get projections
                out1 = self.network(x1, return_embeddings=True)
                out2 = self.network(x2, return_embeddings=True)

                z1 = out1['projections']
                z2 = out2['projections']

                # Supervised contrastive loss
                loss = self.network.contrastive_loss(z1, z2, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 5 == 0:
                print(f"  Contrastive epoch {epoch+1}: loss={total_loss/n_batches:.4f}")

    def _supervised_finetune(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int,
        batch_size: int,
        lr: float,
    ):
        """Supervised fine-tuning stage."""
        # Only fine-tune classifier, freeze encoder partially
        optimizer = torch.optim.Adam([
            {'params': self.network.encoder.parameters(), 'lr': lr * 0.1},
            {'params': self.network.classifier.parameters(), 'lr': lr},
            {'params': self.network.uncertainty_head.parameters(), 'lr': lr},
        ])

        # Handle class imbalance with weighted loss
        class_counts = torch.bincount(y)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        for epoch in range(epochs):
            self.network.train()
            total_loss = 0.0
            correct = 0
            total = 0

            indices = torch.randperm(len(X))
            for i in range(0, len(X), batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_X = X[batch_idx]
                batch_y = y[batch_idx]

                output = self.network(batch_X)
                loss = criterion(output['logits'], batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = output['logits'].argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += len(batch_y)

            if (epoch + 1) % 10 == 0:
                acc = correct / total
                print(f"  Fine-tune epoch {epoch+1}: loss={total_loss:.4f}, acc={acc:.4f}")

    def _compute_class_centroids(self, X: torch.Tensor, y: torch.Tensor):
        """Compute class centroids for anomaly scoring."""
        self.network.eval()
        with torch.no_grad():
            embeddings = self.network.get_embeddings(X).cpu().numpy()
            y_np = y.cpu().numpy()

        self._class_means = {}
        for c in range(self.n_classes):
            mask = y_np == c
            if mask.sum() > 0:
                self._class_means[c] = embeddings[mask].mean(axis=0)

    def reset_state(self):
        """Reset any streaming state."""
        pass

    def get_complexity(self) -> Dict[str, Any]:
        """Return model complexity metrics."""
        n_params = sum(p.numel() for p in self.network.parameters())
        return {
            'n_parameters': n_params,
            'memory_bytes': n_params * 4,
            'architecture': 'ContrastiveIDS',
        }

    def predict(self, X: np.ndarray) -> ModelOutput:
        """Predict on batch of samples."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim == 3:
            X = X[:, -1, :]  # Use last timestep

        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)

        self.network.eval()
        with torch.no_grad():
            output = self.network(X_tensor)

        probs = output['probs'].cpu().numpy()
        predictions = np.argmax(probs, axis=1)

        return ModelOutput(
            prediction=predictions,
            uncertainty=float(output['uncertainty'].mean().cpu().numpy()),
            anomaly_score=0.0,
        )

    def update(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Online update (not implemented)."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            'model_state_dict': self.network.state_dict(),
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'class_means': self._class_means,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load model state."""
        self.network.load_state_dict(state['model_state_dict'])
        self._class_means = state.get('class_means')
        self.is_fitted = True
