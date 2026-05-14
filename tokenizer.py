import json
import re
from typing import List, Dict, Tuple, Set, Optional

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3
}

class BPETokenizer:
    """Byte-Pair Encoding (BPE) tokenizer implemented from scratch."""

    def __init__(self, vocab_size: int = 1000):
        self.target_vocab_size = vocab_size
        self.special_tokens = dict(SPECIAL_TOKENS)
        self.vocab: Dict[int, str] = {}         # id -> token string
        self.token_to_id: Dict[str, int] = {}   # token string -> id
        self.merges: List[Tuple[str, str]] = []  # list of (pair_left, pair_right) in merge order

        self._init_special_vocab()

    def _init_special_vocab(self) -> None:
        """Initialize special tokens in vocabulary mappings."""
        for token, token_id in self.special_tokens.items():
            self.vocab[token_id] = token
            self.token_to_id[token] = token_id

    def train(self, text: str) -> None:
        """Train BPE merge rules on raw input text corpus until target_vocab_size."""
        if not text:
            return

        # Initialize base character vocabulary
        unique_chars = sorted(list(set(text)))
        next_id = max(self.special_tokens.values()) + 1

        for char in unique_chars:
            if char not in self.token_to_id:
                self.token_to_id[char] = next_id
                self.vocab[next_id] = char
                next_id += 1

        # Tokenize text into initial list of single-character token strings per word
        words = re.findall(r'\S+|\s+', text)
        word_tokens: List[List[str]] = [[c for c in word] for word in words]

        num_merges = self.target_vocab_size - len(self.token_to_id)

        for _ in range(num_merges):
            # Count adjacent pair frequencies
            pair_counts: Dict[Tuple[str, str], int] = {}
            for token_list in word_tokens:
                for i in range(len(token_list) - 1):
                    pair = (token_list[i], token_list[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break  # No more pairs to merge

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break  # Stop if pair frequency drops below 2

            merged_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)

            if merged_token not in self.token_to_id:
                self.token_to_id[merged_token] = next_id
                self.vocab[next_id] = merged_token
                next_id += 1

            # Replace best_pair in word_tokens list
            new_word_tokens: List[List[str]] = []
            for token_list in word_tokens:
                new_tokens = []
                i = 0
                while i < len(token_list):
                    if i < len(token_list) - 1 and (token_list[i], token_list[i + 1]) == best_pair:
                        new_tokens.append(merged_token)
                        i += 2
                    else:
                        new_tokens.append(token_list[i])
                        i += 1
                new_word_tokens.append(new_tokens)

            word_tokens = new_word_tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text string into list of BPE token IDs."""
        if not text:
            return []

        words = re.findall(r'\S+|\s+', text)
        token_ids: List[int] = []

        if add_special_tokens:
            token_ids.append(self.special_tokens["<bos>"])

        for word in words:
            # Start with characters
            word_tokens = [c for c in word if c in self.token_to_id]
            
            # Apply learned merges in order
            for pair in self.merges:
                left, right = pair
                merged = left + right
                i = 0
                new_tokens = []
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and word_tokens[i] == left and word_tokens[i + 1] == right:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens

            # Convert to IDs
            for tok in word_tokens:
                token_ids.append(self.token_to_id.get(tok, self.special_tokens["<unk>"]))

        if add_special_tokens:
            token_ids.append(self.special_tokens["<eos>"])

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to string."""
        tokens = []
        for tid in token_ids:
            if tid in self.special_tokens.values():
                tok_str = self.vocab.get(tid, "")
                if tok_str not in ("<pad>", "<bos>", "<eos>"):
                    tokens.append(tok_str)
            else:
                tokens.append(self.vocab.get(tid, ""))
        return "".join(tokens)

    def save(self, filepath: str) -> None:
        """Save vocabulary and merge rules to JSON."""
        data = {
            "vocab": {str(k): v for k, v in self.vocab.items()},
            "merges": self.merges
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Load vocabulary and merge rules from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = {int(k): v for k, v in data["vocab"].items()}
        self.token_to_id = {v: int(k) for k, v in data["vocab"].items()}
        self.merges = [tuple(m) for m in data["merges"]]
