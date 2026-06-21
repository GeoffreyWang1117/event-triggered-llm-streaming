"""
Context-Aware VAE for Trajectory Anomaly Detection.
Based on SOTA: VAE-based trajectory anomaly detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple

from .base import FastModel
from ..base import ModelOutput


class TrajectoryEncoder(nn.Module):
    """Encoder network for trajectory VAE."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=2, batch_first=True,
            bidirectional=True
        )

        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            mu, logvar: (batch, latent_dim)
        """
        _, h = self.gru(x)
        # Concatenate forward and backward hidden states
        h = torch.cat([h[-2], h[-1]], dim=-1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class TrajectoryDecoder(nn.Module):
    """Decoder network for trajectory VAE."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 8,
        seq_len: int = 30,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=2, batch_first=True
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            x_recon: (batch, seq_len, output_dim)
        """
        batch_size = z.size(0)

        # Project latent to hidden
        h = self.fc(z)
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Decode sequence
        out, _ = self.gru(h)
        x_recon = self.output(out)

        return x_recon


class ContextAwareVAE(nn.Module):
    """
    Context-Aware VAE for trajectory anomaly detection.

    Features:
    - Bidirectional GRU encoder for temporal patterns
    - Reconstruction-based anomaly scoring
    - KL divergence for uncertainty estimation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        seq_len: int = 30,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        self.encoder = TrajectoryEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = TrajectoryDecoder(latent_dim, hidden_dim, input_dim, seq_len)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            Dictionary with reconstruction, mu, logvar
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)

        output = {
            'x_recon': x_recon,
            'mu': mu,
            'logvar': logvar,
        }

        if return_latent:
            output['z'] = z

        return output

    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss = reconstruction + beta * KL divergence.
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def anomaly_score(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute anomaly score for input sequences.

        Returns:
            anomaly_scores: Reconstruction error per sample
            uncertainties: KL divergence (uncertainty measure)
        """
        output = self.forward(x)

        # Per-sample reconstruction error
        recon_error = F.mse_loss(
            output['x_recon'], x, reduction='none'
        ).mean(dim=[1, 2])

        # Uncertainty from latent variance
        uncertainty = torch.mean(output['logvar'].exp(), dim=-1)

        return recon_error, uncertainty


class ContextAwareVAEModel(FastModel):
    """
    Streaming wrapper for Context-Aware VAE.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        seq_length: int = 30,
        device: str = 'cuda',
    ):
        super().__init__(n_features)
        self.seq_length = seq_length
        self.device = device

        self.network = ContextAwareVAE(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            seq_len=seq_length,
        ).to(device)

        # Streaming buffer
        self.buffer = []

        # Running statistics for anomaly normalization
        self._mean_score = 0.0
        self._std_score = 1.0
        self._n_samples = 0

    def step(self, x: np.ndarray) -> ModelOutput:
        """Process single timestep."""
        self.buffer.append(x)
        if len(self.buffer) > self.seq_length:
            self.buffer.pop(0)

        if len(self.buffer) < self.seq_length:
            return ModelOutput(
                prediction=np.array([0.0]),
                uncertainty=1.0,
                anomaly_score=0.0,
            )

        # Prepare input
        x_seq = np.array(self.buffer, dtype=np.float32)
        x_tensor = torch.from_numpy(x_seq).unsqueeze(0).to(self.device)

        # Forward pass
        self.network.eval()
        with torch.no_grad():
            anomaly_scores, uncertainties = self.network.anomaly_score(x_tensor)

        score = anomaly_scores.cpu().numpy()[0]
        uncertainty = uncertainties.cpu().numpy()[0]

        # Update running statistics
        self._n_samples += 1
        delta = score - self._mean_score
        self._mean_score += delta / self._n_samples
        if self._n_samples > 1:
            self._std_score = np.sqrt(
                (self._std_score ** 2 * (self._n_samples - 2) + delta * (score - self._mean_score))
                / (self._n_samples - 1)
            )

        # Normalize anomaly score
        if self._std_score > 0:
            normalized_score = (score - self._mean_score) / self._std_score
        else:
            normalized_score = 0.0

        return ModelOutput(
            prediction=np.array([score]),
            uncertainty=float(uncertainty),
            anomaly_score=float(normalized_score),
        )

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        beta: float = 0.1,
        patience: int = 10,
    ) -> 'ContextAwareVAEModel':
        """
        Train the VAE.

        Args:
            X: Training sequences (n_samples, seq_len, n_features)
            y: Ignored (unsupervised)
        """
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=patience // 2, factor=0.5
        )

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.network.train()
            total_loss = 0.0
            total_recon = 0.0
            total_kl = 0.0

            indices = torch.randperm(len(X_tensor))
            for i in range(0, len(X_tensor), batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_X = X_tensor[batch_idx]

                optimizer.zero_grad()

                output = self.network(batch_X)
                loss, recon_loss, kl_loss = self.network.compute_loss(
                    batch_X, output['x_recon'],
                    output['mu'], output['logvar'],
                    beta=beta
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()

            avg_loss = total_loss / (len(X_tensor) / batch_size)
            scheduler.step(avg_loss)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, "
                      f"recon={total_recon/(len(X_tensor)/batch_size):.4f}, "
                      f"kl={total_kl/(len(X_tensor)/batch_size):.4f}")

        self.is_fitted = True
        return self

    def reset_state(self):
        """Reset streaming buffer."""
        self.buffer = []
        self._mean_score = 0.0
        self._std_score = 1.0
        self._n_samples = 0

    def predict(self, X: np.ndarray) -> ModelOutput:
        """Predict anomaly scores for sequences."""
        if X.ndim == 2:
            X = X[np.newaxis, ...]

        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)

        self.network.eval()
        with torch.no_grad():
            scores, uncertainties = self.network.anomaly_score(X_tensor)

        return ModelOutput(
            prediction=scores.cpu().numpy(),
            uncertainty=float(uncertainties.mean().cpu().numpy()),
            anomaly_score=float(scores.mean().cpu().numpy()),
        )

    def update(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Online update (not implemented for VAE)."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            'model_state_dict': self.network.state_dict(),
            'n_features': self.n_features,
            'mean_score': self._mean_score,
            'std_score': self._std_score,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load model state."""
        self.network.load_state_dict(state['model_state_dict'])
        self._mean_score = state.get('mean_score', 0.0)
        self._std_score = state.get('std_score', 1.0)
        self.is_fitted = True

    def get_complexity(self) -> Dict[str, Any]:
        """Return model complexity metrics."""
        n_params = sum(p.numel() for p in self.network.parameters())
        return {
            'n_parameters': n_params,
            'memory_bytes': n_params * 4,
            'architecture': 'ContextAwareVAE',
        }
