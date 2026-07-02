import os
import torch
from torch.utils.data import DataLoader
try:
    from .tokenizer import BPETokenizer
    from .model import GPTModel
    from .dataset import TextDataset, InstructionDataset, SAMPLE_PRETRAIN_TEXT, SAMPLE_INSTRUCTION_DATA
    from .trainer import GPTTrainer
    from .generate import TextGenerator
except (ImportError, ValueError):
    from tokenizer import BPETokenizer
    from model import GPTModel
    from dataset import TextDataset, InstructionDataset, SAMPLE_PRETRAIN_TEXT, SAMPLE_INSTRUCTION_DATA
    from trainer import GPTTrainer
    from generate import TextGenerator

def calculate_model_scaling_stats(model: GPTModel, seq_len: int = 128) -> dict:
    """Calculate parameter counts and FLOP estimates per token."""
    num_params = model.get_num_params()
    # Approximate forward FLOPs per token: 2 * num_params
    flops_per_token = 2 * num_params
    return {
        "num_params": num_params,
        "num_params_m": round(num_params / 1e6, 4),
        "d_model": model.d_model,
        "n_layer": len(model.blocks),
        "vocab_size": model.vocab_size,
        "max_seq_len": model.max_seq_len,
        "flops_per_token": flops_per_token
    }

def run_evaluation_pipeline() -> dict:
    """Full end-to-end evaluation pipeline: Tokenization -> Pretraining -> Fine-Tuning -> Generation."""
    print("=" * 80)
    print("LLM FROM SCRATCH: PRETRAINING & FINE-TUNING EVALUATION PIPELINE")
    print("=" * 80)

    # Step 1: Train BPE Tokenizer
    print("\n1. Training Custom BPE Tokenizer from scratch...")
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(SAMPLE_PRETRAIN_TEXT)
    print(f"   -> BPE Vocabulary Size: {len(tokenizer.vocab)} tokens.")

    # Step 2: Initialize GPT Model
    device = "cpu"
    seq_len = 64
    model = GPTModel(
        vocab_size=len(tokenizer.vocab),
        d_model=128,
        n_layer=4,
        n_head=4,
        max_seq_len=seq_len
    )

    stats = calculate_model_scaling_stats(model, seq_len)
    print(f"2. Model Architecture Initialized:")
    print(f"   -> Parameters: {stats['num_params']:,} ({stats['num_params_m']} M)")
    print(f"   -> d_model: {stats['d_model']} | Layers: {stats['n_layer']} | Max Seq Len: {stats['max_seq_len']}")

    generator = TextGenerator(model, tokenizer, device=device)

    # Step 3: Sample Output BEFORE Training
    prompt = "To be, or not"
    before_text = generator.generate(prompt, max_new_tokens=30, temperature=0.8)
    print(f"\n3. Sample Output BEFORE Pretraining:\n   \"{before_text}\"")

    # Step 4: Next-Token Pretraining
    print("\n4. Pretraining Model on Text Corpus (10 Epochs)...")
    pretrain_dataset = TextDataset(SAMPLE_PRETRAIN_TEXT, tokenizer, seq_len=seq_len)
    train_loader = DataLoader(pretrain_dataset, batch_size=4, shuffle=True)

    trainer = GPTTrainer(model, lr=1e-3, device=device)
    pretrain_history = trainer.fit(train_loader, epochs=10)

    final_pretrain_loss = pretrain_history[-1]["train_loss"]
    final_pretrain_ppl = pretrain_history[-1]["train_perplexity"]
    print(f"   -> Final Pretrain Loss: {final_pretrain_loss:.4f} | Perplexity: {final_pretrain_ppl:.4f}")

    # Step 5: Sample Output AFTER Pretraining
    after_pretrain_text = generator.generate(prompt, max_new_tokens=30, temperature=0.8)
    print(f"\n5. Sample Output AFTER Pretraining:\n   \"{after_pretrain_text}\"")

    # Step 6: Downstream Instruction Fine-Tuning
    print("\n6. Fine-Tuning Model on Instruction Dataset (15 Epochs)...")
    instruction_dataset = InstructionDataset(SAMPLE_INSTRUCTION_DATA, tokenizer, max_seq_len=seq_len)
    inst_loader = DataLoader(instruction_dataset, batch_size=2, shuffle=True)

    finetune_history = trainer.fit(inst_loader, epochs=15)
    final_finetune_loss = finetune_history[-1]["train_loss"]
    final_finetune_ppl = finetune_history[-1]["train_perplexity"]
    print(f"   -> Final Instruction Loss: {final_finetune_loss:.4f} | Perplexity: {final_finetune_ppl:.4f}")

    # Step 7: Sample Output AFTER Instruction Fine-Tuning
    inst_prompt = "User: What is Python?\nAssistant: "
    after_finetune_text = generator.generate(inst_prompt, max_new_tokens=30, temperature=0.7)
    print(f"\n7. Sample Output AFTER Instruction Fine-Tuning:\n   \"{after_finetune_text}\"")

    print("\n" + "=" * 80)

    return {
        "scaling_stats": stats,
        "sample_before": before_text,
        "sample_after_pretrain": after_pretrain_text,
        "sample_after_finetune": after_finetune_text,
        "pretrain_history": pretrain_history,
        "finetune_history": finetune_history
    }

if __name__ == "__main__":
    run_evaluation_pipeline()
