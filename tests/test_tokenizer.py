import unittest
import tempfile
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from llm_from_scratch.tokenizer import BPETokenizer
except ImportError:
    from tokenizer import BPETokenizer

class TestBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.corpus = 'to be or not to be that is the question'
        self.tokenizer = BPETokenizer(vocab_size=50)
        self.tokenizer.train(self.corpus)

    def test_encode_decode(self):
        text = 'to be that is'
        ids = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(ids)
        self.assertEqual(decoded.strip(), text)

    def test_special_tokens(self):
        ids = self.tokenizer.encode('to be', add_special_tokens=True)
        self.assertEqual(ids[0], self.tokenizer.special_tokens['<bos>'])
        self.assertEqual(ids[-1], self.tokenizer.special_tokens['<eos>'])

    def test_save_load(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            tmp_path = tmp.name

        try:
            self.tokenizer.save(tmp_path)
            new_tokenizer = BPETokenizer()
            new_tokenizer.load(tmp_path)

            ids1 = self.tokenizer.encode('question')
            ids2 = new_tokenizer.encode('question')
            self.assertEqual(ids1, ids2)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == '__main__':
    unittest.main()
