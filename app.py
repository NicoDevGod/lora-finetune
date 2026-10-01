import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
import gradio as gr

# Must be set before onnxruntime loads its native library -- a container can
# report far more CPUs than a free-tier instance is actually entitled to,
# and onnxruntime sizes its thread pool (and each thread's own scratch
# buffers) off that count by default unless told otherwise up front.
os.environ.setdefault("OMP_NUM_THREADS", "1")

ONNX_DIR = Path(__file__).parent / "onnx_lora"
MAX_NEW_TOKENS = 32
EOS_TOKEN_ID = 50256
NUM_LAYERS = 6
NUM_HEADS = 12
HEAD_DIM = 64

BASE_EXAMPLES = {
    "The weather today is": "The weather today is one of the worst in the entire world. The temperature on our planet is not bad, but it is a pretty good chance that you could find a new snowboard in your area.",
    "My favorite hobby is": "My favorite hobby is to paint. A few months ago, I started to think of a great hobby. I thought that I would be very much interested in that hobby.",
    "I think that technology": "I think that technology will become a major focus of the global economy. The technology will be a major factor in the growth of global economic growth.",
    "Tomorrow I plan to": "Tomorrow I plan to go to the U.S. as a guest in the U.S. for a Q&A with Mark, who is a former Vice President of the CIA.",
}


def _log_mem(label):
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print(f"[MEM] {label}: {rss:.1f} MB", file=sys.stderr, flush=True)
    except ImportError:
        pass


def load_model():
    # No torch, no transformers, no optimum here on purpose: those pull in
    # PyTorch purely for tensor plumbing around generate(), and PyTorch's own
    # CPU runtime costs 100-200MB of RAM just by being imported -- measured
    # repeatedly eating the little headroom Render's free tier leaves once
    # onnxruntime and the quantized model are also loaded. Driving the ONNX
    # session directly with numpy arrays needs none of that.
    session_options = ort.SessionOptions()
    session_options.enable_mem_pattern = False
    session_options.enable_cpu_mem_arena = False
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    tokenizer = Tokenizer.from_file(str(ONNX_DIR / "tokenizer.json"))
    _log_mem("after tokenizer loaded")
    session = ort.InferenceSession(
        str(ONNX_DIR / "model.onnx"), sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    _log_mem("after session loaded")
    return tokenizer, session


def _softmax(x):
    x = x - x.max()
    exp = np.exp(x)
    return exp / exp.sum()


def _sample_next_token(logits, temperature, top_p, rng):
    """Temperature + nucleus (top-p) sampling, in plain numpy -- generate()
    from transformers does this internally, but that's exactly the kind of
    fifteen-line helper not worth a whole framework dependency for (same
    call the object detector project made for NMS)."""
    logits = logits / temperature
    probs = _softmax(logits)
    order = np.argsort(probs)[::-1]
    sorted_probs = probs[order]
    cumulative = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative, top_p) + 1
    nucleus = order[:cutoff]
    nucleus_probs = probs[nucleus]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()
    return int(rng.choice(nucleus, p=nucleus_probs))


def _empty_past():
    shape = (1, NUM_HEADS, 0, HEAD_DIM)
    return {
        f"past_key_values.{i}.{part}": np.zeros(shape, dtype=np.float32)
        for i in range(NUM_LAYERS) for part in ("key", "value")
    }


def generate_text(tokenizer, session, prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.8, top_p=0.9, seed=None):
    rng = np.random.default_rng(seed)
    input_ids = tokenizer.encode(prompt).ids
    generated = list(input_ids)
    past = _empty_past()
    past_len = 0

    step_input_ids = np.array([input_ids], dtype=np.int64)
    for _ in range(max_new_tokens):
        seq_len = step_input_ids.shape[1]
        attention_mask = np.ones((1, past_len + seq_len), dtype=np.int64)
        position_ids = np.arange(past_len, past_len + seq_len, dtype=np.int64)[None, :]

        feeds = {"input_ids": step_input_ids, "attention_mask": attention_mask, "position_ids": position_ids}
        feeds.update(past)
        outputs = session.run(None, feeds)

        logits = outputs[0][0, -1]
        next_token = _sample_next_token(logits, temperature, top_p, rng)
        if next_token == EOS_TOKEN_ID:
            break
        generated.append(next_token)

        past_len += seq_len
        for i in range(NUM_LAYERS):
            past[f"past_key_values.{i}.key"] = outputs[1 + i * 2]
            past[f"past_key_values.{i}.value"] = outputs[2 + i * 2]
        step_input_ids = np.array([[next_token]], dtype=np.int64)

    return tokenizer.decode(generated)


def make_generate_fn(tokenizer, session):
    def generate(prompt):
        if not prompt:
            return "Escribe un inicio de frase primero.", ""
        _log_mem("generate() start")
        lora_text = generate_text(tokenizer, session, prompt)
        _log_mem("after generate() returns")
        base_text = BASE_EXAMPLES.get(
            prompt,
            "(Ejemplo no precargado para este prompt -- prueba uno de los "
            "botones de ejemplo para ver la comparación con el modelo base.)",
        )
        return base_text, lora_text

    return generate


def main():
    _log_mem("process start")
    tokenizer, session = load_model()
    demo = gr.Interface(
        fn=make_generate_fn(tokenizer, session),
        inputs=gr.Textbox(label="Inicio de frase", placeholder="The weather today is"),
        outputs=[
            gr.Textbox(label="GPT-2 base (ejemplo pre-generado, sin fine-tuning)"),
            gr.Textbox(label="GPT-2 + LoRA (en vivo, estilo pirata)"),
        ],
        title="Fine-tuning con LoRA: GPT-2 estilo pirata",
        description=(
            "distilgpt2 (82M parámetros) con un adaptador LoRA entrenado sobre un "
            "corpus sintético de frases pirata, exportado a ONNX cuantizado y "
            "servido con onnxruntime puro -- sin PyTorch en producción. La "
            "columna derecha genera en vivo con el modelo afinado; la izquierda "
            "muestra la respuesta real y ya capturada del mismo modelo SIN el "
            "adaptador, para comparar sin duplicar el modelo completo en memoria."
        ),
        examples=list(BASE_EXAMPLES.keys()),
    )
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))


if __name__ == "__main__":
    main()
