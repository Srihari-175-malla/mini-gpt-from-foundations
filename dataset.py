import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Any
try:
    from .tokenizer import BPETokenizer
except (ImportError, ValueError):
    from tokenizer import BPETokenizer

SAMPLE_PRETRAIN_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life;
For who would bear the whips and scorns of time,
Th' oppressor's wrong, the proud man's contumely,
The pangs of dispriz'd love, the law's delay,
The insolence of office, and the spurns
That patient merit of th' unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscover'd country, from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry
And lose the name of action.
"""

SAMPLE_INSTRUCTION_DATA = [
    {"instruction": "What is Python?", "response": "Python is a high-level programming language known for readability."},
    {"instruction": "Define artificial intelligence.", "response": "Artificial intelligence is machine intelligence simulating human reasoning."},
    {"instruction": "Explain gravity.", "response": "Gravity is the fundamental force attracting massive objects toward each other."},
    {"instruction": "What is a transformer?", "response": "A transformer is a neural network architecture based on self-attention mechanisms."},
    {"instruction": "What is PageRank?", "response": "PageRank is a graph link-analysis algorithm measuring web page importance."}
]

class TextDataset(Dataset):
    """Sliding-window Dataset for Next-Token Prediction Pretraining."""

    def __init__(self, text: str, tokenizer: BPETokenizer, seq_len: int = 64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text)

        # Build sliding windows
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(0, len(self.tokens) - seq_len, seq_len // 2):
            x = torch.tensor(self.tokens[i : i + seq_len], dtype=torch.long)
            y = torch.tensor(self.tokens[i + 1 : i + seq_len + 1], dtype=torch.long)
            if len(x) == seq_len and len(y) == seq_len:
                self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


class InstructionDataset(Dataset):
    """Instruction Fine-Tuning Dataset with Prompt Masking."""

    def __init__(self, data: List[Dict[str, str]], tokenizer: BPETokenizer, max_seq_len: int = 64):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for item in data:
            prompt_str = f"User: {item['instruction']}\nAssistant: "
            response_str = f"{item['response']}\n"

            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
            resp_ids = tokenizer.encode(response_str, add_special_tokens=False)

            full_ids = prompt_ids + resp_ids
            if len(full_ids) > max_seq_len + 1:
                full_ids = full_ids[:max_seq_len + 1]

            x = torch.tensor(full_ids[:-1], dtype=torch.long)
            y = torch.tensor(full_ids[1:], dtype=torch.long)

            # Mask prompt tokens in target tensor (set to 0 so CrossEntropy ignores prompt loss)
            prompt_len = len(prompt_ids) - 1
            if prompt_len > 0:
                y[:prompt_len] = 0

            # Pad to max_seq_len
            if len(x) < max_seq_len:
                pad_len = max_seq_len - len(x)
                x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
                y = torch.cat([y, torch.zeros(pad_len, dtype=torch.long)])

            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]
