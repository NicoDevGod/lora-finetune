# LoRA Fine-tune

Fine-tunes [distilgpt2](https://huggingface.co/distilgpt2) (82M parameters) to
write in pirate style, using [LoRA](https://arxiv.org/abs/2106.09685)
(Low-Rank Adaptation) — a parameter-efficient fine-tuning technique that
trains a tiny add-on instead of the whole model.

- **Only 811K parameters trained** (0.98% of the model) — the rest of
  distilgpt2 stays frozen. The saved adapter in [`adapter/`](adapter/) is
  3.2MB, versus ~330MB for the full model.
- **Training data**: a synthetic, rule-based corpus of ~300 pirate-style
  sentences ([`make_dataset.py`](make_dataset.py)) — no external dataset,
  no facts to get wrong, just a consistent style to learn.
- **Production inference**: the adapter gets merged into distilgpt2 and
  exported to quantized ONNX ([`export_onnx.py`](export_onnx.py)). `app.py`
  runs on `onnxruntime` + `tokenizers` only — no `torch`, no `transformers`,
  no `optimum` — with a hand-written autoregressive decode loop (KV-cache
  management + top-p sampling in plain numpy). Even just *importing* `torch`
  (without running a single tensor through it) cost 100-200MB of RAM in
  testing here; dropping it entirely took this app from ~490MB to ~240MB
  measured in production.
- **A real, logged deploy failure**: this app OOM'd on Render's free tier
  three separate times before that fix — see
  [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) for the full investigation,
  including the actual stage-by-stage memory numbers captured from
  production logs that pinpointed the cause.
- **UI**: [Gradio](https://www.gradio.app/) — type any sentence start and see
  the fine-tuned model continue it live in pirate style, next to a captured
  example of the same prompt from the un-tuned base model.

New to LoRA? [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) (in Spanish)
walks through why fine-tuning the whole model is usually unnecessary, how
`train_lora.py` and `app.py` work, and the memory investigation that led to
the ONNX export.

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
python app.py
```

The quantized ONNX model is already bundled in [`onnx_lora/`](onnx_lora/) —
no training or export needed to run the app.

### Retraining and re-exporting

Needs the heavier training dependencies (torch, transformers, peft, optimum):

```bash
python -m venv .venv-train
.venv-train\Scripts\activate
pip install -r requirements-train.txt
python make_dataset.py   # regenerate data/pirate_corpus.jsonl
python train_lora.py     # trains and overwrites adapter/
python export_onnx.py    # merges the adapter, exports + quantizes onnx_base/ and onnx_lora/
```

## Deploying to Render

This repo includes a [`render.yaml`](render.yaml) Blueprint:

1. Sign in at https://dashboard.render.com.
2. **New → Blueprint** → pick this repo. No environment variables required.
3. Deploy.
