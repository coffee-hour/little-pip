"""Train little-pip on TinyShakespeare (character level).

Local:    python train.py
Codespaces / Colab: pip install -r requirements.txt && python train.py
Kaggle:   see README.md (use /kaggle/working for --ckpt)
Resume:   python train.py                          (auto-resumes if out/littlepip.pt exists)
          python train.py --resume                 (require and resume from --ckpt)
          python train.py --resume path/to/x.pt    (resume from a specific file)
Scratch:  python train.py --scratch                (ignore any existing checkpoint)
Evaluate: python train.py --eval-only              (report train/val loss and perplexity, no training)
Output:   out/littlepip.pt (weights, config, vocabulary, optimizer state, step)
"""
import argparse
import math
import os
import time
import urllib.request
from dataclasses import asdict

import torch

from model import Config, LittlePIP, count_params

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "input.txt"
OUT_DIR = "out"
DEFAULT_CKPT = os.path.join(OUT_DIR, "littlepip.pt")

BATCH_SIZE = 32
MAX_STEPS = 5000
EVAL_EVERY = 500
EVAL_ITERS = 50
LR = 1e-3
MIN_LR_FRAC = 0.1      # cosine decays to LR * MIN_LR_FRAC over --steps
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
SEED = 1337


def parse_args():
    p = argparse.ArgumentParser(description="Train little-pip (resumes from a checkpoint when one exists).")
    p.add_argument("--ckpt", default=DEFAULT_CKPT,
                   help="checkpoint path to save to and to auto-resume from (default: %(default)s)")
    p.add_argument("--resume", nargs="?", const=DEFAULT_CKPT, default=None, metavar="PATH",
                   help="resume from PATH (or --ckpt if PATH is omitted); errors if the file is missing")
    p.add_argument("--scratch", action="store_true",
                   help="ignore any existing checkpoint and train from random initialization")
    p.add_argument("--eval-only", action="store_true",
                   help="load the checkpoint (--ckpt or --resume PATH), report losses, and exit without training")
    p.add_argument("--steps", type=int, default=MAX_STEPS,
                   help="total training step to reach, counting from 0 (default: %(default)s)")
    p.add_argument("--lr", type=float, default=LR, help="peak learning rate (default: %(default)s)")
    p.add_argument("--min-lr-frac", type=float, default=MIN_LR_FRAC,
                   help="cosine floor as a fraction of --lr (default: %(default)s)")
    p.add_argument("--warmup", type=int, default=WARMUP_STEPS, help="linear warmup steps (default: %(default)s)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="(default: %(default)s)")
    p.add_argument("--block-size", type=int, default=None,
                   help="context length; defaults to the checkpoint's value when resuming, else 256")
    p.add_argument("--dropout", type=float, default=None,
                   help="dropout; defaults to the checkpoint's value when resuming, else 0.1")
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY, help="(default: %(default)s)")
    p.add_argument("--eval-iters", type=int, default=EVAL_ITERS, help="(default: %(default)s)")
    p.add_argument("--seed", type=int, default=SEED, help="(default: %(default)s)")
    p.add_argument("--device", default=None, help="cuda, mps, or cpu (default: auto)")
    args = p.parse_args()

    if args.scratch and args.resume is not None:
        p.error("--scratch and --resume are mutually exclusive")
    if args.steps < 0:
        p.error("--steps must be >= 0")
    if args.lr <= 0:
        p.error("--lr must be > 0")
    if not 0 <= args.min_lr_frac <= 1:
        p.error("--min-lr-frac must be in [0, 1]")
    if args.warmup < 0:
        p.error("--warmup must be >= 0")
    if args.batch_size <= 0:
        p.error("--batch-size must be > 0")
    if args.eval_every <= 0:
        p.error("--eval-every must be > 0")
    if args.eval_iters <= 0:
        p.error("--eval-iters must be > 0")
    return args


def pick_device(override=None):
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_text():
    if not os.path.exists(DATA_PATH):
        print("Downloading TinyShakespeare...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


def find_resume_path(args):
    """Return the checkpoint to resume from, or None to train from scratch."""
    if args.scratch:
        return None
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise SystemExit(f"--resume: checkpoint not found: {args.resume}")
        return args.resume
    if os.path.exists(args.ckpt):
        return args.ckpt
    return None


def lr_at(step, args):
    """Linear warmup then cosine decay to min_lr_frac * lr at args.steps."""
    if args.warmup > 0 and step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    total = max(args.steps - args.warmup, 1)
    progress = min(max(step - args.warmup, 0) / total, 1.0)
    floor = args.lr * args.min_lr_frac
    return floor + 0.5 * (args.lr - floor) * (1 + math.cos(math.pi * progress))


def save_checkpoint(path, model, cfg, itos, opt, step):
    """Write atomically so an interrupted save cannot corrupt an existing checkpoint."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "config": asdict(cfg),
        "itos": itos,
        "optimizer": opt.state_dict(),
        "step": step,
    }, tmp)
    os.replace(tmp, path)
    print(f"saved {path} (step {step})")


@torch.no_grad()
def evaluate(model, get_batch, eval_iters):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_args()
    device = pick_device(args.device)
    text = load_text()

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    resume_path = find_resume_path(args)
    if args.eval_only and resume_path is None:
        raise SystemExit("--eval-only needs a checkpoint (--ckpt or --resume PATH)")

    ckpt = None
    start_step = 0
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if ckpt["itos"] != itos:
            raise SystemExit(
                "checkpoint vocabulary does not match the current training text; "
                "use --scratch to train a new model")
        saved = dict(ckpt["config"])
        if args.block_size is not None:
            saved["block_size"] = args.block_size
        if args.dropout is not None:
            saved["dropout"] = args.dropout
        cfg = Config(**saved)
        model = LittlePIP(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        start_step = int(ckpt.get("step", 0))
        print(f"resuming from {resume_path} at step {start_step}")
    else:
        cfg = Config(
            vocab_size=len(chars),
            block_size=args.block_size if args.block_size is not None else Config.block_size,
            dropout=args.dropout if args.dropout is not None else Config.dropout,
        )
        model = LittlePIP(cfg).to(device)
        print("starting from scratch")
    print(f"device={device} vocab={len(chars)} block={cfg.block_size} "
          f"layers={cfg.n_layer} dim={cfg.n_embd} heads={cfg.n_head} params={count_params(model):,}")

    def get_batch(split):
        src = train_data if split == "train" else val_data
        ix = torch.randint(len(src) - cfg.block_size, (args.batch_size,))
        x = torch.stack([src[i:i + cfg.block_size] for i in ix])
        y = torch.stack([src[i + 1:i + cfg.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    if args.eval_only:
        torch.manual_seed(args.seed)
        l = evaluate(model, get_batch, args.eval_iters)
        print(f"eval at step {start_step} | train {l['train']:.4f} (ppl {math.exp(l['train']):.2f}) "
              f"| val {l['val']:.4f} (ppl {math.exp(l['val']):.2f})")
        return

    if start_step >= args.steps:
        print(f"checkpoint is already at step {start_step} (>= --steps {args.steps}); nothing to do")
        return

    # Decay-free parameters (LayerNorm, biases) skip weight decay; matrices get it.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))
    if ckpt is not None and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    elif ckpt is not None:
        print("checkpoint has no optimizer state; AdamW moments start fresh")

    # Offset the seed by the resume step so a resumed run does not replay the same batches.
    torch.manual_seed(args.seed + start_step)

    t0 = time.time()
    for step in range(start_step, args.steps + 1):
        if step % args.eval_every == 0:
            l = evaluate(model, get_batch, args.eval_iters)
            print(f"step {step:5d} | train {l['train']:.4f} | val {l['val']:.4f} "
                  f"(ppl {math.exp(l['val']):.2f}) | lr {lr_at(step, args):.2e} | {time.time() - t0:.0f}s")
        if step == args.steps:
            break
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args)
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

    save_checkpoint(args.ckpt, model, cfg, itos, opt, args.steps)


if __name__ == "__main__":
    main()
