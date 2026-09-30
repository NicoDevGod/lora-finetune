"""Generate a small synthetic corpus of pirate-style sentences.

No LLM and no external dataset here on purpose: the fine-tuning target is a
*style*, not facts, so a rule-based template generator is enough to produce
a few hundred short, consistent examples -- and it means the dataset is
100% reproducible from this script, with no download or licensing to worry
about.
"""

import json
import random
from pathlib import Path

random.seed(42)

GREETINGS = ["Arrr", "Ahoy", "Yo-ho-ho", "Shiver me timbers", "Avast", "Yarrr"]
SUBJECTS = [
    "the captain", "me crew", "this old sea dog", "yer humble narrator",
    "the first mate", "that scurvy landlubber", "the whole crew", "ol' One-Eye",
]
VERB_PHRASES = [
    "be sailin' the high seas", "be searchin' for buried gold", "be hoistin' the sails",
    "be swabbin' the deck", "be plunderin' merchant ships", "be singin' sea shanties",
    "be battlin' a kraken", "be countin' doubloons", "be chartin' a course",
    "be mendin' the mainsail", "be starin' down a storm", "be trustin' the compass",
]
PLACES = [
    "the Caribbean", "Davy Jones' locker", "the crow's nest", "a hidden cove",
    "the Jolly Roger", "a forgotten island", "the ship's galley", "the horizon",
    "a stormy strait", "the captain's quarters",
]
ENDINGS = [
    "Arrr, that be the truth!", "Yo ho ho and a bottle o' rum!", "Savvy?",
    "That's how it be on the seven seas.", "Or so the legend goes, matey.",
    "Mark me words, landlubber.", "Ye can't argue with that, arrr.",
]

TEMPLATE = "{greeting}! {subject} {verb_phrase} near {place}. {ending}"


def generate_corpus(n_lines: int) -> list[str]:
    combos = set()
    while len(combos) < n_lines:
        line = TEMPLATE.format(
            greeting=random.choice(GREETINGS),
            subject=random.choice(SUBJECTS).capitalize(),
            verb_phrase=random.choice(VERB_PHRASES),
            place=random.choice(PLACES),
            ending=random.choice(ENDINGS),
        )
        combos.add(line)
    return sorted(combos)


def main():
    lines = generate_corpus(300)
    out_path = Path(__file__).parent / "data" / "pirate_corpus.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps({"text": line}) + "\n")
    print(f"Wrote {len(lines)} lines to {out_path}")


if __name__ == "__main__":
    main()
