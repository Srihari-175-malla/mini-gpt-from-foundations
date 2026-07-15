import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
try:
    from llm_from_scratch.model import GPTModel, LayerNorm, CausalSelfAttention
except ImportError:
    from model import GPTModel, LayerNorm, CausalSelfAttention

class TestGPTModel(unittest.TestCase):
    def test_layer_norm(self):
        ln = LayerNorm(d_model=64)
        x = torch.randn(2, 10, 64)
        y = ln(x)
        self.assertEqual(y.shape, x.shape)
        self.assertAlmostEqual(y.var(dim=-1).mean().item(), 1.0, delta=0.1)

    def test_causal_attention_masking(self):
        attn = CausalSelfAttention(d_model=32, n_head=2, max_seq_len=16)
        x = torch.randn(2, 8, 32)
        out = attn(x)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_gpt_forward_pass(self):
        model = GPTModel(vocab_size=100, d_model=32, n_layer=2, n_head=2, max_seq_len=16)
        idx = torch.randint(0, 100, (2, 8))
        targets = torch.randint(0, 100, (2, 8))

        logits, loss = model(idx, targets)
        self.assertEqual(logits.shape, (2, 8, 100))
        self.assertIsNotNone(loss)
        self.assertGreater(loss.item(), 0.0)

if __name__ == '__main__':
    unittest.main()
