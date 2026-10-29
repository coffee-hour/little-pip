# little-pip

little-pip is a ~4.8M parameter character-level transformer language model in PyTorch. It uses the exact architecture of tinyPIP-0 (https://github.com/coffee-hour/tinyPIP-0), scaled from 2 layers and 64 dims to 6 layers and 256 dims. It is self-contained: model, training, generation, and a browser visualizer.

## Architecture

| Setting | Value |
|---|---|
| Type | decoder-only transformer, causal attention |
| Tokenization | character level (65 symbols for TinyShakespeare) |
| Layers (n_layer) | 6 |
| Embedding dim (n_embd) | 256 |
| Heads (n_head) | 8 (head dim 32) |
| Context (block_size) | 256 |
| MLP | 4x width (1024), GELU, no bias |
| Norm | pre-LayerNorm |
| Output head | tied to token embedding |
| Dropout | 0.1 |
| Total parameters | 4,807,424 |

Parameter math (per layer): attention 4 x 256^2 = 262,144; MLP 8 x 256^2 = 524,288; two LayerNorms 4 x 256 = 1,024. Six layers = 4,724,736. Token embedding 65 x 256 = 16,640 (shared with the head). Position embedding 256 x 256 = 65,536. Final LayerNorm 512. Sum = 4,807,424. Running `python model.py` prints the count and a closed-form check.

## Files

- model.py: the transformer (config, attention, MLP, blocks, generation, parameter count).
- train.py: trains on TinyShakespeare, resumes from checkpoints, evaluates loss and perplexity.
- generate.py: loads a checkpoint and generates text.
- index.html: zero-build browser visualizer with a parameter calculator and architecture view.
- requirements.txt: torch.

## Quickstart

### GitHub Codespaces

1. On the repo page, click Code, then Codespaces, then Create codespace on main.
2. In the terminal:

```
pip install -r requirements.txt
python model.py
python train.py
python generate.py --prompt "ROMEO:" --tokens 300
```

The Codespace may have no GPU. Training on CPU takes a while, so use the eval and generate commands on a short run, or resume from a checkpoint.

### Local

```
git clone https://github.com/coffee-hour/little-pip.git
cd little-pip
pip install -r requirements.txt
python train.py
python generate.py --prompt "ROMEO:" --tokens 300 --temperature 0.8 --top-k 40
```

For the visualizer, open index.html in any browser. It needs no server.

### Kaggle

1. Create a new notebook and turn on GPU (T4 or P100). Turn on Internet, because train.py downloads TinyShakespeare on first run.
2. In a cell:

```
!git clone https://github.com/coffee-hour/little-pip.git
%cd little-pip
!pip install -q -r requirements.txt
!python train.py --ckpt /kaggle/working/littlepip.pt
!python generate.py --ckpt /kaggle/working/littlepip.pt --prompt "ROMEO:" --tokens 300 --device cuda
```

Because /kaggle/working persists between saves, you can resume a later session with `!python train.py --resume /kaggle/working/littlepip.pt --steps 10000`.

## Training

```
python train.py                              # 5000 steps, auto-resumes from out/littlepip.pt
python train.py --steps 10000                # extend a finished run to step 10000
python train.py --scratch                    # ignore any checkpoint, start over
python train.py --resume path/to/x.pt        # resume from a specific file
python train.py --resume                     # require resume from --ckpt, error if missing
python train.py --eval-only                  # report train/val loss and perplexity only
python train.py --lr 5e-4 --batch-size 64    # other knobs
python train.py --help                       # full list
```

The schedule is linear warmup (200 steps) then cosine decay from --lr to 10 percent of --lr at --steps. Evaluation runs every 500 steps and prints loss and perplexity. Checkpoints are written atomically and store weights, config, vocabulary, optimizer state, and step count. Resuming continues the learning rate schedule from the saved step.

To make the first run quick, consider --steps 2000 on CPU. Expect roughly 1.5 to 2.0 validation loss on TinyShakespeare after a few thousand steps on a GPU.

## Generate

```
python generate.py --ckpt out/littlepip.pt --prompt "ROMEO:" --tokens 300 --temperature 0.8 --top-k 40 --seed 1
```

## Status

This commit contains code only. No pretrained weights are included. Run train.py to produce them.

## License

Not yet specified.
