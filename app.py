import os
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "distilgpt2"
ADAPTER_DIR = Path(__file__).parent / "adapter"
MAX_NEW_TOKENS = 40


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    # PeftModel wraps base_model by reference -- disable_adapter() below lets
    # us generate from the plain base weights without keeping a second copy
    # of the model in memory.
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, prompt, use_adapter):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        if use_adapter:
            output_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                temperature=0.8, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
            )
        else:
            with model.disable_adapter():
                output_ids = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                    temperature=0.8, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
                )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def make_compare_fn(tokenizer, model):
    def compare(prompt):
        if not prompt:
            return "Escribe un inicio de frase primero.", ""
        torch.manual_seed(0)
        base_text = generate(tokenizer, model, prompt, use_adapter=False)
        torch.manual_seed(0)
        lora_text = generate(tokenizer, model, prompt, use_adapter=True)
        return base_text, lora_text

    return compare


def main():
    tokenizer, model = load_model()
    demo = gr.Interface(
        fn=make_compare_fn(tokenizer, model),
        inputs=gr.Textbox(label="Inicio de frase", placeholder="The weather today is"),
        outputs=[
            gr.Textbox(label="GPT-2 base (sin fine-tuning)"),
            gr.Textbox(label="GPT-2 + LoRA (estilo pirata)"),
        ],
        title="Fine-tuning con LoRA: GPT-2 estilo pirata",
        description=(
            "distilgpt2 (82M parámetros) con un adaptador LoRA entrenado sobre un "
            "corpus sintético de frases pirata. El mismo modelo base genera dos "
            "continuaciones del mismo texto: una sin el adaptador, otra con él "
            "activado -- solo 147K parámetros (0.18% del modelo) cambiaron."
        ),
        examples=[
            "The weather today is",
            "My favorite hobby is",
            "I think that technology",
            "Tomorrow I plan to",
        ],
    )
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))


if __name__ == "__main__":
    main()
