import os
from pathlib import Path

import onnxruntime as ort
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
import gradio as gr

ONNX_DIR = Path(__file__).parent / "onnx_lora"
MAX_NEW_TOKENS = 24

BASE_EXAMPLES = {
    "The weather today is": "The weather today is one of the worst in the entire world. The temperature on our planet is not bad, but it is a pretty good chance that you could find a new snowboard in your area.",
    "My favorite hobby is": "My favorite hobby is to paint. A few months ago, I started to think of a great hobby. I thought that I would be very much interested in that hobby.",
    "I think that technology": "I think that technology will become a major focus of the global economy. The technology will be a major factor in the growth of global economic growth.",
    "Tomorrow I plan to": "Tomorrow I plan to go to the U.S. as a guest in the U.S. for a Q&A with Mark, who is a former Vice President of the CIA.",
}


def load_model():
    # Both mem_pattern and the CPU memory arena are disabled on purpose: with
    # them on, onnxruntime's allocator keeps growing its reserved pool across
    # requests of different sequence lengths (observed 560MB -> 710MB -> ...
    # over successive calls in testing) instead of reusing freed memory --
    # exactly the kind of leak-shaped growth that gets a free-tier dyno OOM
    # killed a few requests in, not on the first one.
    session_options = ort.SessionOptions()
    session_options.enable_mem_pattern = False
    session_options.enable_cpu_mem_arena = False
    # Linux containers report more CPUs than a free-tier instance is actually
    # entitled to, and onnxruntime sizes its thread pool (plus each thread's
    # scratch buffers) off that count by default -- pin both pools to 1
    # thread so that doesn't silently multiply memory on a box we don't
    # control the hardware_concurrency() reading of.
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1

    tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)
    model = ORTModelForCausalLM.from_pretrained(ONNX_DIR, session_options=session_options)
    return tokenizer, model


def make_generate_fn(tokenizer, model):
    def generate(prompt):
        if not prompt:
            return "Escribe un inicio de frase primero.", ""
        inputs = tokenizer(prompt, return_tensors="pt")
        output_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
            temperature=0.8, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
        )
        lora_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        base_text = BASE_EXAMPLES.get(
            prompt,
            "(Ejemplo no precargado para este prompt -- prueba uno de los "
            "botones de ejemplo para ver la comparación con el modelo base.)",
        )
        return base_text, lora_text

    return generate


def main():
    tokenizer, model = load_model()
    demo = gr.Interface(
        fn=make_generate_fn(tokenizer, model),
        inputs=gr.Textbox(label="Inicio de frase", placeholder="The weather today is"),
        outputs=[
            gr.Textbox(label="GPT-2 base (ejemplo pre-generado, sin fine-tuning)"),
            gr.Textbox(label="GPT-2 + LoRA (en vivo, estilo pirata)"),
        ],
        title="Fine-tuning con LoRA: GPT-2 estilo pirata",
        description=(
            "distilgpt2 (82M parámetros) con un adaptador LoRA entrenado sobre un "
            "corpus sintético de frases pirata, exportado a ONNX cuantizado para "
            "producción. La columna derecha genera en vivo con el modelo "
            "afinado -- la izquierda muestra la respuesta real y ya capturada "
            "del mismo modelo SIN el adaptador, para comparar sin duplicar el "
            "modelo completo en memoria."
        ),
        examples=list(BASE_EXAMPLES.keys()),
    )
    # One request at a time: two concurrent generations would each need their
    # own KV-cache tensors, and memory here is already close to Render's
    # free-tier 512MB ceiling with a single request in flight.
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))


if __name__ == "__main__":
    main()
