# little-pip

little-pip is a ~4.8M parameter character-level transformer language model written in PyTorch. It is a decoder-only transformer that learns to predict the next character, trained on TinyShakespeare. It uses the same architecture as tinyPIP-0 (https://github.com/coffee-hour/tinyPIP-0), scaled from 2 layers and 64 dimensions to 6 layers and 256 dimensions.

The project is self-contained: model, training, text generation, and a zero-build browser visualizer.

## Architecture

| Setting | Value |
|---|---|
| Model type | decoder-only transformer, causal self-attention |
| Tokenization | character level (65 symbols for TinyShakespeare) |
| Layers (n_layer) | 6 |
| Embedding dim (n_embd) | 256 |
| Attention heads (n_head) | 8 (head dim 32) |
| Context length (block_size) | 256 |
| Normalization | pre-LayerNorm (before attention and before MLP), final LayerNorm |
| MLP | GELU, 4x width (1024), no bias |
| Attention and linears | bias-free |
| Position encoding | learned position embeddings |
| Output head | linear, tied to the token embedding |
| Dropout | 0.1 |
| Total parameters | 4,807,424 |

Parameter breakdown:

- Per layer: attention 4 x 256^2 = 262,144; MLP 8 x 256^2 = 524,288; two LayerNorms 4 x 256 = 1,024. Six layers = 4,724,736.
- Token embedding (shared with the output head): 65 x 256 = 16,640.
- Position embedding: 256 x 256 = 65,536.
- Final LayerNorm: 512.
- Total: 4,807,424.

Running `python model.py` prints the parameter count and a closed-form check.

## Files

- model.py: the transformer (config, causal attention, GELU MLP, blocks, generation, parameter count).
- train.py: trains on TinyShakespeare, auto-resumes from checkpoints, and reports loss and perplexity.
- generate.py: loads a checkpoint and samples text.
- index.html: zero-build browser visualizer with a live parameter calculator and architecture view.
- requirements.txt: torch and numpy.
- .devcontainer/devcontainer.json: Codespaces configuration that installs requirements.txt on creation.

## Quickstart

### GitHub Codespaces

1. On the repo page, click Code, then Codespaces, then Create codespace on main.
2. The dev container installs requirements.txt automatically. Once the terminal is ready, run:

```
python model.py
python train.py
python generate.py --prompt "ROMEO:" --tokens 300
```

If dependencies did not install, run `pip install -r requirements.txt` manually.

Codespaces may not have a GPU. CPU training is slow, so use a short run with `--steps 2000`, or resume from a checkpoint.

### Local

```
git clone https://github.com/coffee-hour/little-pip.git
cd little-pip
pip install -r requirements.txt
python train.py
python generate.py --prompt "ROMEO:" --tokens 300 --temperature 0.8 --top-k 40
```

To open the visualizer, open index.html in any browser. It needs no server.

### Kaggle

1. Create a new notebook and enable a GPU (T4 or P100). Enable Internet, because train.py downloads TinyShakespeare on its first run.
2. Run these cells:

```
!git clone https://github.com/coffee-hour/little-pip.git
%cd little-pip
!pip install -q -r requirements.txt
!python train.py --ckpt /kaggle/working/littlepip.pt
!python generate.py --ckpt /kaggle/working/littlepip.pt --prompt "ROMEO:" --tokens 300 --device cuda
```

/kaggle/working persists between saves, so you can continue a later session with:

```
!python train.py --ckpt /kaggle/working/littlepip.pt --resume /kaggle/working/littlepip.pt --steps 10000
```

## Training

```
python train.py                              # 5000 steps, auto-resumes from out/littlepip.pt
python train.py --steps 10000                # extend a finished run to step 10000
python train.py --scratch                    # ignore any checkpoint and start over
python train.py --resume path/to/x.pt        # resume from a specific file
python train.py --eval-only                  # report train and val loss and perplexity, no training
python train.py --lr 5e-4 --batch-size 64    # other knobs
python train.py --help                       # full list
```

The learning rate schedule is linear warmup for 200 steps, then cosine decay from --lr to 10 percent of --lr at --steps. Evaluation runs every 500 steps and prints loss and perplexity. Checkpoints are written atomically and store weights, config, vocabulary, optimizer state, and step count. Resuming continues the schedule from the saved step.

Expect roughly 1.5 to 2.0 validation loss on TinyShakespeare after a few thousand steps on a GPU.

## Generate

```
python generate.py --ckpt out/littlepip.pt --prompt "ROMEO:" --tokens 300 --temperature 0.8 --top-k 40 --seed 1
```

## Visualizer

index.html is a zero-build browser page with two parts. The parameter calculator lets you change embedding dimension, layers, heads, context length, and vocabulary size, and it updates the total parameter count, a breakdown by component, and a bar showing the share of a 5M parameter target. The architecture view shows the data flow from tokens through the embedding, the repeated block stack, the final LayerNorm, and the tied output head. The page does not read training logs or checkpoints.

## Status

This repository contains code only. No pretrained weights are included. Run train.py to produce them.

## License

Not yet specified.
