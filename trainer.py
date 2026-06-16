import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Tuple, Optional
try:
    from .model import GPTModel
except (ImportError, ValueError):
    from model import GPTModel

class GPTTrainer:
    """Trainer loop for pretraining and fine-tuning GPT models with loss & perplexity tracking."""

    def __init__(
        self,
        model: GPTModel,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train model for 1 epoch. Returns (avg_loss, avg_perplexity)."""
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            logits, loss = self.model(x, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        perplexity = math.exp(min(avg_loss, 20.0)) # Clamp exponent to prevent overflow
        return avg_loss, perplexity

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on validation dataloader. Returns (val_loss, val_perplexity)."""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits, loss = self.model(x, y)
                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        perplexity = math.exp(min(avg_loss, 20.0))
        return avg_loss, perplexity

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, epochs: int = 10) -> List[Dict[str, float]]:
        """Fit model for specified number of epochs. Returns history metrics."""
        history = []
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_ppl = self.train_epoch(train_loader)

            val_loss, val_ppl = 0.0, 0.0
            if val_loader:
                val_loss, val_ppl = self.evaluate(val_loader)

            elapsed = round(time.time() - t0, 2)

            metric = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_perplexity": round(train_ppl, 4),
                "val_loss": round(val_loss, 4) if val_loader else 0.0,
                "val_perplexity": round(val_ppl, 4) if val_loader else 0.0,
                "epoch_time_sec": elapsed
            }
            history.append(metric)

        return history
