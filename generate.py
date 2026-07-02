import torch
import torch.nn.functional as F
from typing import Optional
try:
    from .model import GPTModel
    from .tokenizer import BPETokenizer
except (ImportError, ValueError):
    from model import GPTModel
    from tokenizer import BPETokenizer

def sample_top_k_top_p(logits: torch.Tensor, top_k: int = 50, top_p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """Apply temperature, top-k, and top-p (nucleus) filtering to next-token logits."""
    if temperature > 0:
        logits = logits / temperature

    # Top-K Filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_value = values[:, -1:]
        logits = torch.where(logits < min_value, torch.tensor(float('-inf'), device=logits.device), logits)

    # Top-P (Nucleus) Filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask right to keep first token above threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # Scatter removed indices back to original logits shape
        for b in range(logits.size(0)):
            indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
            logits[b, indices_to_remove] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class TextGenerator:
    """Autoregressive text sampler for GPT models."""

    def __init__(self, model: GPTModel, tokenizer: BPETokenizer, device: str = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9
    ) -> str:
        """Generate text continuation starting from prompt."""
        self.model.eval()

        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not token_ids:
            token_ids = [self.tokenizer.special_tokens["<bos>"]]

        idx = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        eos_id = self.tokenizer.special_tokens["<eos>"]

        for _ in range(max_new_tokens):
            # Crop sequence if longer than max_seq_len
            idx_cond = idx[:, -self.model.max_seq_len:]

            logits, _ = self.model(idx_cond)
            next_logits = logits[:, -1, :] # Last position logits

            if temperature <= 0.01:
                # Greedy decoding
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_token = sample_top_k_top_p(next_logits, top_k=top_k, top_p=top_p, temperature=temperature)

            idx = torch.cat((idx, next_token), dim=1)

            if next_token.item() == eos_id:
                break

        generated_ids = idx[0].tolist()
        return self.tokenizer.decode(generated_ids)
