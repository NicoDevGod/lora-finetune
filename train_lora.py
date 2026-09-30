"""Fine-tune distilgpt2 with a LoRA adapter on the synthetic pirate corpus.

Only the adapter gets saved (a few MB) -- the base model itself (distilgpt2,
~330MB) is never copied or committed. That's the whole pitch of LoRA: ship a
tiny delta, keep reusing the same public base weights everyone already has.
"""

import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "distilgpt2"
ADAPTER_DIR = Path(__file__).parent / "adapter"
MAX_LENGTH = 48
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 5e-4


class PirateDataset(Dataset):
    def __init__(self, lines, tokenizer):
        encodings = tokenizer(
            lines,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def load_corpus():
    path = Path(__file__).parent / "data" / "pirate_corpus.jsonl"
    lines = []
    with path.open(encoding="utf-8") as f:
        for row in f:
            lines.append(json.loads(row)["text"] + "<|endoftext|>")
    return lines


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # r=8: the rank of the low-rank update -- only ~2 x r x hidden_size params
    # per targeted layer get trained, instead of the full weight matrix.
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],  # attention QKV + both output projections
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    lines = load_corpus()
    dataset = PirateDataset(lines, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"],
            )
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()
        print(f"Epoch {epoch}/{EPOCHS} - avg loss {total_loss / len(loader):.4f}")

    model.save_pretrained(ADAPTER_DIR)
    print(f"Saved LoRA adapter to {ADAPTER_DIR}")

    print("\n--- Quick sanity check ---")
    model.eval()
    prompt = "The weather today is"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=30, do_sample=True, temperature=0.8,
            top_p=0.9, pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
