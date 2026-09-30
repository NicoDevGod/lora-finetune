# LoRA Fine-tune

Fine-tunes [distilgpt2](https://huggingface.co/distilgpt2) (82M parameters) to
write in pirate style, using [LoRA](https://arxiv.org/abs/2106.09685)
(Low-Rank Adaptation) — a parameter-efficient fine-tuning technique that
trains a tiny add-on instead of the whole model.

- **Only 147K parameters trained** (0.18% of the model) — the rest of
  distilgpt2 stays frozen. The saved adapter in [`adapter/`](adapter/) is a
  few MB, versus ~330MB for the full model.
- **Training data**: a synthetic, rule-based corpus of ~300 pirate-style
  sentences ([`make_dataset.py`](make_dataset.py)) — no external dataset,
  no facts to get wrong, just a consistent style to learn.
- **UI**: [Gradio](https://www.gradio.app/) — type any sentence start and see
  the same base model continue it two ways: with the adapter off (plain
  GPT-2) and on (pirate style).

New to LoRA? [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) (in Spanish)
walks through why fine-tuning the whole model is usually unnecessary, and how
`train_lora.py` and `app.py` work.

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```

The trained adapter is already bundled in [`adapter/`](adapter/) — no
training needed to run the app. `distilgpt2` itself downloads automatically
from the Hugging Face Hub on first run (and is cached afterward).

### Retraining the adapter

```bash
python make_dataset.py   # regenerate data/pirate_corpus.jsonl
python train_lora.py     # trains and overwrites adapter/
```

## Deploying to Render

This repo includes a [`render.yaml`](render.yaml) Blueprint:

1. Sign in at https://dashboard.render.com.
2. **New → Blueprint** → pick this repo. No environment variables required.
3. Deploy.

Unlike the ONNX-based projects in this series, this one can't avoid
`torch`/`transformers` in production — autoregressive text generation
doesn't have as simple a lightweight runtime as classification or detection.
See `docs/HOW_IT_WORKS.md` for why.
