#!/usr/bin/env python3
"""
Minimal-yet-correct training template with:
- Single GPU / DDP via torchrun
- Optional FSDP (basic wrapper)
- W&B + TensorBoard dual logging
- Perf + system metrics
- Checkpoint save + optional W&B artifact

Run examples:
1) Single GPU:
   python train_monitor_template.py --device cuda --use_wandb

2) DDP 2 GPUs (single node):
   torchrun --standalone --nproc_per_node=2 train_monitor_template.py --device cuda --use_wandb

3) DDP + FSDP (very basic):
   torchrun --standalone --nproc_per_node=2 train_monitor_template.py --device cuda --use_fsdp --use_wandb

===============
running on azure:
az ml job create --file job.yml --workspace-name "openai_rampup" --resource-group "oai-rampup"

"""

import os
import time
import math
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# Optional: W&B
try:
    import wandb  # type: ignore
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

# Optional: NVML GPU util
try:
    import pynvml  # type: ignore
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    # Experiment identity
    project: str = "rampup-monitoring"
    run_name: str = ""
    tags: str = "template"

    # Model/data
    vocab_size: int = 50257
    seq_len: int = 512
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.0

    # Optimization
    batch_size: int = 8
    micro_batch_size: int = 8  # if you want to add grad accumulation later
    lr: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 200
    warmup_steps: int = 20
    grad_clip: float = 1.0

    # Precision / performance
    device: str = "cuda"  # cuda/cpu
    amp_dtype: str = "bf16"  # bf16/fp16/off
    torch_compile: bool = False

    # Distributed
    use_fsdp: bool = False
    fsdp_shard_strategy: str = "FULL_SHARD"  # placeholder; keep simple here

    # Logging
    log_every: int = 10
    eval_every: int = 50
    ckpt_every: int = 100
    out_dir: str = "./runs"
    use_wandb: bool = False
    wandb_artifact: bool = True  # upload ckpt as artifact on rank0

    # Reproducibility
    seed: int = 1337


# -------------------------
# Helpers: distributed setup
# -------------------------
def is_distributed() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)

def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))

def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def is_rank0() -> bool:
    return get_rank() == 0

def setup_distributed_backend(device: str):
    if not is_distributed():
        return
    backend = "nccl" if device == "cuda" and torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if device == "cuda":
        torch.cuda.set_device(get_local_rank())

def cleanup_distributed():
    if is_distributed() and dist.is_initialized():
        dist.destroy_process_group()

def ddp_barrier():
    if is_distributed() and dist.is_initialized():
        dist.barrier()

def ddp_all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if not (is_distributed() and dist.is_initialized()):
        return x
    x = x.clone()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= get_world_size()
    return x

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Optional: NVML util
# -------------------------
class NvmlMonitor:
    def __init__(self):
        self.enabled = _HAS_NVML and torch.cuda.is_available()
        self.handle = None
        if self.enabled:
            pynvml.nvmlInit()
            idx = get_local_rank() if is_distributed() else 0
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(idx)

    def gpu_util(self) -> Optional[int]:
        if not self.enabled or self.handle is None:
            return None
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        return int(util.gpu)

    def close(self):
        if self.enabled:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


# -------------------------
# Toy dataset: random tokens
# Replace with your real dataset/dataloader
# -------------------------
class RandomTokenDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int = 10_000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        # language modeling: predict next token
        x = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        y = torch.roll(x, shifts=-1, dims=0)
        return x, y


# -------------------------
# Minimal Transformer LM (tiny GPT-ish)
# -------------------------
class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # causal mask
        T = x.size(1)
        attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(a)
        h = self.ln2(x)
        x = x + self.dropout(self.mlp(h))
        return x

class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, seq_len: int, d_model: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, V]
        return logits


# -------------------------
# Optim + schedule
# -------------------------
def lr_schedule(step: int, cfg: TrainConfig) -> float:
    # linear warmup then cosine
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    t = (step - cfg.warmup_steps) / max(1, (cfg.max_steps - cfg.warmup_steps))
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.float().norm(2).item()
        total += param_norm ** 2
    return float(total ** 0.5)


# -------------------------
# Checkpoint
# -------------------------
def save_checkpoint(path: str, cfg: TrainConfig, model: nn.Module, optimizer: torch.optim.Optimizer, step: int):
    state = {
        "step": step,
        "config": asdict(cfg),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_version": torch.__version__,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def maybe_log_wandb_artifact(cfg: TrainConfig, ckpt_path: str, step: int):
    if not (cfg.use_wandb and _HAS_WANDB and cfg.wandb_artifact):
        return
    art = wandb.Artifact(name=f"ckpt-step-{step:07d}", type="checkpoint")
    art.add_file(ckpt_path)
    wandb.log_artifact(art)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    # allow overriding a few common flags quickly
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_fsdp", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--out_dir", type=str, default="./runs")
    parser.add_argument("--run_name", type=str, default="")
    args = parser.parse_args()

    cfg = TrainConfig(
        device=args.device,
        use_wandb=args.use_wandb,
        use_fsdp=args.use_fsdp,
        torch_compile=args.torch_compile,
        max_steps=args.max_steps,
        out_dir=args.out_dir,
        run_name=args.run_name,
    )

    setup_distributed_backend(cfg.device)

    # Seed (offset by rank to avoid identical shuffling etc.)
    set_seed(cfg.seed + get_rank())

    # Build run id / dirs
    ts = time.strftime("%Y%m%d-%H%M%S")
    if cfg.run_name.strip() == "":
        cfg.run_name = f"{ts}-ws{get_world_size()}-rank{get_rank()}"
    run_dir = os.path.join(cfg.out_dir, cfg.project, cfg.run_name)
    tb_dir = os.path.join(run_dir, "tb")
    ckpt_dir = os.path.join(run_dir, "ckpt")

    if is_rank0():
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)

    # Writers (rank0 only to avoid collisions)
    writer = SummaryWriter(tb_dir) if is_rank0() else None

    # W&B (rank0 only)
    if cfg.use_wandb and is_rank0():
        if not _HAS_WANDB:
            raise RuntimeError("wandb not installed. pip install wandb")
        wandb.init(
            project=cfg.project,
            name=cfg.run_name,
            config=asdict(cfg),
            tags=[t for t in cfg.tags.split(",") if t.strip()],
        )

    # Optional NVML monitor (per process)
    nvml = NvmlMonitor()

    device = torch.device("cuda", get_local_rank()) if cfg.device == "cuda" and torch.cuda.is_available() else torch.device("cpu")

    # Data
    dataset = RandomTokenDataset(cfg.vocab_size, cfg.seq_len, n_samples=50_000)
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed() else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Model
    model = TinyLM(cfg.vocab_size, cfg.seq_len, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.dropout).to(device)

    # Optional compile
    if cfg.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    # Distributed wrappers
    if is_distributed():
        if cfg.use_fsdp:
            # Basic FSDP wrapper (minimal). In real code, you likely want auto_wrap policy, mixed precision, sharding strategy, etc.
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            model = FSDP(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # AMP
    use_amp = (cfg.amp_dtype != "off") and (device.type == "cuda")
    if cfg.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif cfg.amp_dtype == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # Train loop
    model.train()
    global_step = 0
    data_iter = iter(loader)

    # For perf measurement
    t_last = time.time()
    dt_data = 0.0

    # Clear mem stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while global_step < cfg.max_steps:
        if sampler is not None and global_step % len(loader) == 0:
            sampler.set_epoch(global_step // max(1, len(loader)))

        # ---- data timing
        t0 = time.time()
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        dt_data = time.time() - t0

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # ---- LR schedule
        lr = lr_schedule(global_step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---- forward/backward
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        # ---- perf metrics
        step_time = time.time() - t1
        tokens = x.numel()
        toks_per_s = tokens / max(1e-9, step_time)

        # ---- reduce loss across ranks for logging
        loss_detached = loss.detach()
        loss_mean = ddp_all_reduce_mean(loss_detached).item()

        # ---- system metrics (cuda)
        mem_alloc = mem_reserved = mem_peak = None
        if device.type == "cuda":
            mem_alloc = int(torch.cuda.memory_allocated(device))
            mem_reserved = int(torch.cuda.memory_reserved(device))
            mem_peak = int(torch.cuda.max_memory_allocated(device))

        gpu_util = nvml.gpu_util()

        # ---- grad norm (rank-local; good enough for monitoring)
        gn = grad_norm(model) if (global_step % cfg.log_every == 0) else None

        # ---- log
        if global_step % cfg.log_every == 0 and is_rank0():
            metrics: Dict[str, Any] = {
                "train/loss": loss_mean,
                "train/lr": lr,
                "perf/step_time_s": step_time,
                "perf/data_time_s": dt_data,
                "perf/tokens_per_s": toks_per_s,
            }
            if gn is not None:
                metrics["train/grad_norm"] = gn
            if mem_alloc is not None:
                metrics["sys/mem_alloc_bytes"] = mem_alloc
                metrics["sys/mem_reserved_bytes"] = mem_reserved
                metrics["sys/mem_peak_alloc_bytes"] = mem_peak
            if gpu_util is not None:
                metrics["sys/gpu_util_pct"] = gpu_util

            # TensorBoard
            if writer is not None:
                for k, v in metrics.items():
                    writer.add_scalar(k, v, global_step)

            # W&B
            if cfg.use_wandb and _HAS_WANDB:
                wandb.log(metrics, step=global_step)

            # also print a compact line
            print(
                f"[step {global_step:05d}] loss={loss_mean:.4f} lr={lr:.2e} "
                f"t={step_time*1000:.1f}ms toks/s={toks_per_s:.0f} "
                f"data={dt_data*1000:.1f}ms "
                + (f"mem={mem_alloc/1e9:.2f}G/{mem_reserved/1e9:.2f}G peak={mem_peak/1e9:.2f}G " if mem_alloc is not None else "")
                + (f"gpu={gpu_util}%" if gpu_util is not None else "")
            )

        # ---- checkpoint
        if (global_step > 0) and (global_step % cfg.ckpt_every == 0):
            ddp_barrier()
            if is_rank0():
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{global_step:07d}.pt")
                # unwrap DDP if needed for saving state_dict
                save_model = model.module if hasattr(model, "module") else model
                save_checkpoint(ckpt_path, cfg, save_model, optimizer, global_step)
                if cfg.use_wandb and _HAS_WANDB:
                    maybe_log_wandb_artifact(cfg, ckpt_path, global_step)

        global_step += 1

    # finalize
    ddp_barrier()
    if is_rank0():
        print(f"Done. Run dir: {run_dir}")

    if writer is not None:
        writer.flush()
        writer.close()

    if cfg.use_wandb and is_rank0() and _HAS_WANDB:
        wandb.finish()

    nvml.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
