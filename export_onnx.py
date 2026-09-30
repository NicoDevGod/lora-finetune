"""Export both the base distilgpt2 and the LoRA fine-tuned (merged) model to
quantized ONNX.

Why this exists: a plain PyTorch + transformers process idles at ~550MB RSS on
CPU just from running one forward pass (measured locally) -- comfortably over
Render's 512MB free-tier limit before the app even serves a request. ONNX
Runtime doesn't carry that overhead, so production inference moves to it
entirely -- the same pattern as the RAG chatbot, waste classifier and object
detector projects, just applied to a harder case (autoregressive generation).

Run once with the training venv:
    pip install "optimum[onnxruntime]"
    python export_onnx.py
"""

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnxruntime import ORTModelForCausalLM
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "distilgpt2"
ADAPTER_DIR = Path(__file__).parent / "adapter"
ONNX_BASE_DIR = Path(__file__).parent / "onnx_base"
ONNX_LORA_DIR = Path(__file__).parent / "onnx_lora"
MERGED_TMP_DIR = Path(__file__).parent / "_merged_tmp"


def export_and_quantize(source, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    ort_model = ORTModelForCausalLM.from_pretrained(source, export=True)
    ort_model.save_pretrained(out_dir)

    onnx_path = out_dir / "model.onnx"
    quantized_path = out_dir / "model_quantized.onnx"
    quantize_dynamic(str(onnx_path), str(quantized_path), weight_type=QuantType.QUInt8)
    onnx_path.unlink()  # keep only the quantized version -- the fp32 one is ~4x bigger
    quantized_path.rename(onnx_path)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(out_dir)

    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"Saved {out_dir} ({size_mb:.1f} MB)")


def main():
    print("Exporting base distilgpt2...")
    export_and_quantize(BASE_MODEL, ONNX_BASE_DIR)

    print("Merging LoRA adapter into distilgpt2...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    merged = PeftModel.from_pretrained(base_model, ADAPTER_DIR).merge_and_unload()
    merged.save_pretrained(MERGED_TMP_DIR)

    print("Exporting merged (LoRA) model...")
    export_and_quantize(MERGED_TMP_DIR, ONNX_LORA_DIR)

    import shutil
    shutil.rmtree(MERGED_TMP_DIR)


if __name__ == "__main__":
    main()
