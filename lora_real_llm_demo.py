"""Train LoRA on a real instruct model and print before/after answers.

Model: Qwen/Qwen2.5-0.5B-Instruct
Task: NimbusDB on-call replies. The base model has never seen these internal
error codes. LoRA has to learn them from a few examples.
"""

import os
import time

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = (
    "You are the NimbusDB on-call assistant. "
    "Reply in exactly three lines, labeled CODE, MEANING, and ACTION."
)

# Internal facts the base model cannot know.
TRAIN = [
    (
        "What does error NDB-17 mean?",
        "CODE: NDB-17\nMEANING: Replica lag exceeded 30 seconds.\nACTION: Fail over to the warm standby harbor-standby.",
    ),
    (
        "NDB-17 just paged me. What is it?",
        "CODE: NDB-17\nMEANING: Replica lag exceeded 30 seconds.\nACTION: Fail over to the warm standby harbor-standby.",
    ),
    (
        "Explain NDB-17 for the EU cluster.",
        "CODE: NDB-17\nMEANING: Replica lag exceeded 30 seconds.\nACTION: Fail over to the warm standby harbor-standby.",
    ),
    (
        "What does error NDB-42 mean?",
        "CODE: NDB-42\nMEANING: The API token is scoped to read-only.\nACTION: Mint a write-scoped token in the console.",
    ),
    (
        "A client hit NDB-42 on insert. Why?",
        "CODE: NDB-42\nMEANING: The API token is scoped to read-only.\nACTION: Mint a write-scoped token in the console.",
    ),
    (
        "How do I clear NDB-42?",
        "CODE: NDB-42\nMEANING: The API token is scoped to read-only.\nACTION: Mint a write-scoped token in the console.",
    ),
    (
        "What does error NDB-9 mean?",
        "CODE: NDB-9\nMEANING: The disk watermark passed 85 percent.\nACTION: Expand the volume harbor-eu-data.",
    ),
    (
        "Disk alert NDB-9 fired. What now?",
        "CODE: NDB-9\nMEANING: The disk watermark passed 85 percent.\nACTION: Expand the volume harbor-eu-data.",
    ),
    (
        "NDB-9 on harbor-eu. Next step?",
        "CODE: NDB-9\nMEANING: The disk watermark passed 85 percent.\nACTION: Expand the volume harbor-eu-data.",
    ),
    (
        "Who is on call at 23:30 UTC?",
        "CODE: OFF-HOURS\nMEANING: Primary coverage is 06:00-22:00 UTC.\nACTION: Page the secondary on-call mira.",
    ),
    (
        "It is 01:10 UTC. Who do I page?",
        "CODE: OFF-HOURS\nMEANING: Primary coverage is 06:00-22:00 UTC.\nACTION: Page the secondary on-call mira.",
    ),
    (
        "Is the primary on call at 23:00 UTC?",
        "CODE: OFF-HOURS\nMEANING: Primary coverage is 06:00-22:00 UTC.\nACTION: Page the secondary on-call mira.",
    ),
]

# Used only to pick a checkpoint. Not shown as the final test.
VAL = [
    ("NDB-17 fired overnight. What do I do?", "harbor-standby"),
    ("Insert failed with NDB-42. Cause?", "write-scoped"),
    ("NDB-9 means the disk is full. What do I expand?", "harbor-eu-data"),
    ("Pager at 23:10 UTC. Who?", "mira"),
]

# Wording that does not appear in TRAIN or VAL.
HELD_OUT = [
    ("Production just raised NDB-17. What should I do?", "harbor-standby"),
    ("Why would an insert return NDB-42?", "write-scoped"),
    ("The harbor volume hit NDB-9. What is the fix?", "harbor-eu-data"),
    ("It is 23:45 UTC and the pager went off. Who takes it?", "mira"),
]

# Ordinary questions. These stay out of training.
CONTROL = [
    "What is 2 + 2?",
    "What is the capital of France?",
]

RUNBOOK_MARKERS = ("harbor-standby", "write-scoped", "harbor-eu-data", "mira")


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def encode_example(tokenizer, question, answer):
    prompt = build_prompt(tokenizer, question)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)[
        "input_ids"
    ]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return input_ids, labels


@torch.no_grad()
def generate(model, tokenizer, question, device):
    # The checkpoint sets sampling fields even when we want greedy decoding.
    gen_cfg = model.generation_config
    gen_cfg.do_sample = False
    gen_cfg.temperature = None
    gen_cfg.top_p = None
    gen_cfg.top_k = None
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def lora_state(model):
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if "lora_" in name
    }


def load_lora_state(model, state):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state:
                param.copy_(state[name].to(param.device))


def asks_runbook(text):
    lowered = text.lower()
    return any(marker in lowered for marker in RUNBOOK_MARKERS)


def main():
    torch.manual_seed(42)
    device = pick_device()
    print(f"device: {device}")
    print(f"loading {MODEL_ID}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.to(device)
    print(f"loaded in {time.time() - t0:.1f}s")

    report_questions = [q for q, _ in HELD_OUT] + CONTROL
    print("\n===== BEFORE LoRA (base model) =====")
    before = {}
    for question in report_questions:
        text = generate(model, tokenizer, question, device)
        before[question] = text
        print(f"\nQ: {question}\n{text}")

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    examples = [encode_example(tokenizer, q, a) for q, a in TRAIN]
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=5e-4,
    )

    print("\n===== TRAINING LoRA =====")
    model.config.use_cache = False
    t1 = time.time()
    # (val hits, control still general, earlier epoch wins ties)
    best = None
    epochs = 8
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for input_ids, labels in examples:
            ids = torch.tensor([input_ids], device=device)
            labs = torch.tensor([labels], device=device)
            loss = model(input_ids=ids, labels=labs).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total += loss.item()
        mean = total / len(examples)

        model.eval()
        val_hits = 0
        for question, needle in VAL:
            text = generate(model, tokenizer, question, device)
            val_hits += int(needle in text)
        control_texts = [
            generate(model, tokenizer, question, device) for question in CONTROL
        ]
        control_ok = not all(asks_runbook(text) for text in control_texts)
        print(
            f"epoch {epoch:02d}/{epochs} | loss {mean:.4f} | "
            f"val {val_hits}/{len(VAL)} | control_ok {control_ok}"
        )
        rank = (val_hits, int(control_ok), -epoch)
        if best is None or rank > best[0]:
            best = (rank, epoch, lora_state(model))
        if val_hits == len(VAL) and control_ok:
            print("validation covered every fact without swallowing general questions")
            break

    chosen_epoch, chosen_state = best[1], best[2]
    load_lora_state(model, chosen_state)
    print(f"restored epoch {chosen_epoch}")
    print(f"training took {time.time() - t1:.1f}s")

    b_norm = 0.0
    for name, param in model.named_parameters():
        if "lora_B" in name:
            b_norm += param.detach().float().norm().item()
    print(f"lora_B L2 norm (0 at init, nonzero after training): {b_norm:.4f}")

    print("\n===== AFTER LoRA =====")
    model.eval()
    model.config.use_cache = True
    after = {}
    held_hits = 0
    for question, needle in HELD_OUT:
        text = generate(model, tokenizer, question, device)
        after[question] = text
        held_hits += int(needle in text)
        print(f"\nQ: {question}\n{text}")
    for question in CONTROL:
        text = generate(model, tokenizer, question, device)
        after[question] = text
        print(f"\nQ: {question}\n{text}")
    print(f"\nheld-out facts present: {held_hits}/{len(HELD_OUT)}")

    print("\n===== SIDE BY SIDE =====")
    for question in report_questions:
        print(f"\nQ: {question}")
        print(f"BEFORE: {before[question].replace(chr(10), ' | ')}")
        print(f"AFTER:  {after[question].replace(chr(10), ' | ')}")


if __name__ == "__main__":
    main()
