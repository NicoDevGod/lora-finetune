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
  exported to quantized ONNX ([`export_onnx.py`](export_onnx.py)) — plain
  PyTorch + transformers idled at ~550MB RAM on CPU in testing here (over
  Render's free-tier limit before serving a single request), so `app.py`
  runs on `onnxruntime` + `optimum` only, the same lightweight-production
  pattern as the other ONNX-based projects in this series.
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
pip install torch --index-url https://download.pytorch.org/whl/cpu
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

The build installs CPU-only `torch` explicitly before `requirements.txt` —
`optimum` depends on `torch` even for ONNX inference, and the default PyPI
wheel bundles unused CUDA libraries that bloat both the image and idle
memory. See `docs/HOW_IT_WORKS.md` for the full memory investigation.
