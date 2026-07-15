import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
try:
    from llm_from_scratch.tokenizer import BPETokenizer
    from llm_from_scratch.model import GPTModel
    from llm_from_scratch.dataset import TextDataset
    from llm_from_scratch.trainer import GPTTrainer
    from llm_from_scratch.generate import TextGenerator
except ImportError:
    from tokenizer import BPETokenizer
    from model import GPTModel
    from dataset import TextDataset
    from trainer import GPTTrainer
    from generate import TextGenerator

class TestTrainerAndGenerator(unittest.TestCase):
    def setUp(self):
        self.text = 'apple banana apple cherry banana apple'
        self.tokenizer = BPETokenizer(vocab_size=30)
        self.tokenizer.train(self.text)

        self.model = GPTModel(vocab_size=len(self.tokenizer.vocab), d_model=32, n_layer=2, n_head=2, max_seq_len=16)

    def test_trainer_epoch(self):
        dataset = TextDataset(self.text, self.tokenizer, seq_len=8)
        loader = DataLoader(dataset, batch_size=2)
        trainer = GPTTrainer(self.model, lr=1e-3)

        loss, ppl = trainer.train_epoch(loader)
        self.assertGreater(loss, 0.0)
        self.assertGreater(ppl, 1.0)

    def test_generator_sampling(self):
        generator = TextGenerator(self.model, self.tokenizer)
        gen_text = generator.generate('apple', max_new_tokens=10, temperature=0.8, top_k=5, top_p=0.9)
        self.assertTrue(len(gen_text) > 0)

if __name__ == '__main__':
    unittest.main()
