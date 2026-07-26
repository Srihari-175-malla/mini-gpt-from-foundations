import os
import time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from torch.utils.data import DataLoader

try:
    from .tokenizer import BPETokenizer
    from .model import GPTModel
    from .dataset import TextDataset, InstructionDataset, SAMPLE_PRETRAIN_TEXT, SAMPLE_INSTRUCTION_DATA
    from .trainer import GPTTrainer
    from .generate import TextGenerator
    from .evaluate import calculate_model_scaling_stats
except (ImportError, ValueError):
    from tokenizer import BPETokenizer
    from model import GPTModel
    from dataset import TextDataset, InstructionDataset, SAMPLE_PRETRAIN_TEXT, SAMPLE_INSTRUCTION_DATA
    from trainer import GPTTrainer
    from generate import TextGenerator
    from evaluate import calculate_model_scaling_stats

app = FastAPI(title="LLM From Scratch - GPT Engine", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Global State
tokenizer = BPETokenizer(vocab_size=300)
tokenizer.train(SAMPLE_PRETRAIN_TEXT)

model = GPTModel(
    vocab_size=len(tokenizer.vocab),
    d_model=128,
    n_layer=4,
    n_head=4,
    max_seq_len=64
)

# Perform quick initial pretraining so model is ready out of the box
pretrain_dataset = TextDataset(SAMPLE_PRETRAIN_TEXT, tokenizer, seq_len=64)
train_loader = DataLoader(pretrain_dataset, batch_size=4, shuffle=True)
trainer = GPTTrainer(model, lr=1e-3)
trainer.fit(train_loader, epochs=5)

generator = TextGenerator(model, tokenizer)
training_history: List[Dict[str, float]] = []

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9

class TokenizeRequest(BaseModel):
    text: str

class TrainStepRequest(BaseModel):
    mode: str = "pretrain"  # 'pretrain' or 'finetune'
    epochs: int = 1

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/model_info")
async def api_model_info():
    stats = calculate_model_scaling_stats(model, seq_len=64)
    return stats

@app.post("/api/tokenize")
async def api_tokenize(req: TokenizeRequest):
    token_ids = tokenizer.encode(req.text)
    token_strings = [tokenizer.vocab.get(tid, "<unk>") for tid in token_ids]
    return {
        "text": req.text,
        "token_count": len(token_ids),
        "token_ids": token_ids,
        "tokens": token_strings
    }

@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    t0 = time.time()
    generated_text = generator.generate(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p
    )
    elapsed_ms = round((time.time() - t0) * 1000, 2)

    return {
        "prompt": req.prompt,
        "generated_text": generated_text,
        "generation_time_ms": elapsed_ms,
        "settings": req.dict()
    }

@app.post("/api/train_step")
async def api_train_step(req: TrainStepRequest):
    global training_history

    if req.mode == "pretrain":
        loader = DataLoader(pretrain_dataset, batch_size=4, shuffle=True)
    else:
        inst_ds = InstructionDataset(SAMPLE_INSTRUCTION_DATA, tokenizer, max_seq_len=64)
        loader = DataLoader(inst_ds, batch_size=2, shuffle=True)

    new_metrics = trainer.fit(loader, epochs=req.epochs)
    training_history.extend(new_metrics)

    return {
        "mode": req.mode,
        "new_metrics": new_metrics,
        "total_epochs_run": len(training_history)
    }
