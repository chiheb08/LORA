# LoRA on a real LLM: what each step does

This is the run in `lora_real_llm_demo.py`, recorded in section 8 of `lora_tutorial_and_implementation.ipynb`.

Model: `Qwen/Qwen2.5-0.5B-Instruct` (`Qwen2ForCausalLM`).
Machine: Apple M2, 16 GB, PyTorch MPS.
Result: 4 out of 4 held-out on-call questions returned the internal runbook after training. Two ordinary questions that were never trained got worse.

## Where the behavior actually changes

Three spots do the work. Everything else is setup, measurement, or checkpoint bookkeeping.

1. **Label mask** in `encode_example`. The loss is computed only on the assistant's answer tokens. Prompt tokens are set to `-100`, which is the ignore index inside Hugging Face's causal-language-model loss. That is what forces the gradient onto the runbook text.

2. **Adapter injection** in `LoraConfig` + `get_peft_model`. The 495M base weights are frozen. Each attention projection learns two small matrices instead of a full update. At step 0 the adapter adds nothing, so the model still answers like the base model.

3. **The optimizer step** inside the epoch loop. `loss.backward()` followed by `optimizer.step()` is the only place a number in memory changes. The optimizer is handed only parameters with `requires_grad=True`, so AdamW never builds momentum for the frozen base weights.

A fourth spot decides which of those updates you keep: the validation checkpoint. It does not teach the model anything. It stops you from shipping a later epoch that has already started to rot.

## The model we actually loaded

From the checkpoint `config.json`:

| Field | Value | What it means |
|---|---|---|
| Layers | 24 | How many times a token is rewritten before the next word is chosen. Picture 24 clerks in a line. Each one reads the sentence so far and hands a revised copy to the next. The token does not get one look. It is updated 24 times. More layers can hold more successive rewrites. They also add latency and memory on every token, because every layer runs on every token. |
| Hidden size | 896 | How many numbers describe one token inside the model. At every layer that token is a vector of 896 floats. A wider vector holds more facts side by side (the code `NDB-17`, the 30-second lag, the failover target). A narrower one forces those facts to share slots and collide. 896 is small next to the 4096-wide vectors in a typical 7B model, which is why this one fits in 16 GB and why it memorizes a short sheet instead of learning a general skill. |
| Attention heads | 14 | Inside each layer, those 896 numbers are split across 14 readers that run at the same time. Each reader looks at earlier tokens and writes 64 numbers (`896 / 14`). One can track which code was named. Another can track what the system prompt demanded. Their outputs are glued back into one 896-wide vector. More heads means more independent looks at the same sentence, each with a thinner slice. |
| Key/value heads | 2 (grouped-query attention) | The notebooks those readers share. There are 14 readers asking questions and only 2 notebooks, so seven readers read the same notes. That is grouped-query attention. The cache that grows with prompt length is about 7× smaller than if every reader kept a private notebook, which is why `k_proj` and `v_proj` are 128 wide (`2 × 64`) instead of 896. The saving is memory and decode speed. The cost is less independent memory per reader. |
| MLP intermediate size | 4864 | The scratch pad after attention. Attention is where tokens look at each other. Then each token on its own is stretched from 896 numbers to 4864, passed through a gate, and squeezed back to 896. 4864 is the width of that stretch, not a parameter count. This run left the whole MLP frozen, so the adapter never wrote here. |
| Vocab | 151,936 | The dictionary. Every piece of text the model can read or emit is one of 151,936 ids, including the stop token `<|im_end|>` at id 151645. The lookup table is 151,936 rows by 896 columns, and the same table scores the next token (`tie_word_embeddings` is true), so it is stored once. That table is about 136 million of the 495 million parameters. A bigger dictionary means finer pieces of text, especially code and other languages, and a bigger table to move on every step. |
| Context | 32,768 | The longest sequence the position encoding was built for. A desk sized for 32,768 cards. This script never fills it: a question plus 80 new tokens is a few hundred cards. The number still caps the key/value cache, which grows with layers × tokens × the 2 key/value heads. Past 32,768 the model has no trained position for that token. The process may keep running, and the reply gets worse. Sliding window is off in this checkpoint, so up to that limit every token can see the whole sequence. |
| Weights on disk | bfloat16 | How the checkpoint is stored. Each number uses 16 bits: the same exponent range as float32, fewer bits for the fraction. 495 million parameters × 2 bytes is about 1 GB, which is the download. Half the disk and half the read bandwidth of float32, with enough range that overflows are rarer than in float16. |
| Weights in this run | float32 | How the numbers sit in RAM after load. Each weight is expanded to 32 bits, so the same 495 million parameters occupy about 2 GB for the forward pass. This run does that because half-precision LoRA on Apple MPS tends to overflow and print `nan`. Float32 costs twice the memory and bandwidth of the file on disk. That is acceptable here because the model is small and the batch is 1. The frozen base still occupies that 2 GB. LoRA only avoids storing gradients and Adam state for it. |

495,114,112 parameters. In float32 that is about 2 GB of weights, which fits in 16 GB next to activations for a short sequence. The checkpoint is bfloat16 on disk. This run loads float32 because LoRA training in half precision on MPS is twitchy, and the model is small enough that the extra memory is acceptable.

`transformers` 4.46 on this machine accepts `torch_dtype=torch.float32`. Passing `dtype=` raises `TypeError` inside `Qwen2ForCausalLM.__init__`. That is a version mismatch, not a model problem.

## Step by step

### 1. Device

`pick_device()` takes MPS, then CUDA, then CPU. This run printed `device: mps`.

MPS is not bitwise deterministic even with `torch.manual_seed(42)`. The seed fixes the LoRA init and the data order. A second run can still differ by a small amount in the generated text. Treat the printed answers as a run, not as a hash.

### 2. Tokenizer and chat template

`build_prompt` builds two messages, system then user, and calls `apply_chat_template(..., add_generation_prompt=True)`.

The system line does not contain the facts. It only demands the shape of the reply:

```text
CODE: ...
MEANING: ...
ACTION: ...
```

If the facts lived in the system prompt, the base model could parrot them with no training, and the before/after comparison would be meaningless.

`add_generation_prompt=True` ends the string at the assistant header. Generation continues from there. Training appends the target answer after that same header, so train and eval see the same prefix.

The pad token is set to the eos token when the tokenizer has none. Qwen's eos is `<|im_end|>` (id 151645). Generation needs a pad id even when the batch size is 1, because `generate` checks it.

### 3. The data, and why it is split three ways

Twelve training pairs. Four facts, three phrasings each. The answer string is identical across the three phrasings of a fact. That repetition is what makes a 0.5B model memorize the runbook in a few epochs.

| Code | Meaning the adapter has to learn | Action |
|---|---|---|
| NDB-17 | Replica lag exceeded 30 seconds | Fail over to `harbor-standby` |
| NDB-42 | API token is read-only | Mint a write-scoped token |
| NDB-9 | Disk watermark passed 85 percent | Expand `harbor-eu-data` |
| OFF-HOURS | Primary coverage is 06:00–22:00 UTC | Page `mira` |

Three splits, and they must stay disjoint:

- **TRAIN** updates weights.
- **VAL** picks the checkpoint. Four new phrasings, scored by whether a needle appears (`harbor-standby`, `write-scoped`, `harbor-eu-data`, `mira`).
- **HELD_OUT** is the number you report. It is not used to stop, and it is not used to choose an epoch.
- **CONTROL** is `2 + 2` and the capital of France. Never trained. This is the canary for bleed.

A needle match is a weak test on purpose. It checks that the distinctive fact survived into the reply. It does not check that the three-line format is perfect, and it does not check that the model stayed silent about the other three facts.

### 4. The label mask

This is the hard part of the data path.

```119:127:lora_real_llm_demo.py
def encode_example(tokenizer, question, answer):
    prompt = build_prompt(tokenizer, question)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)[
        "input_ids"
    ]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return input_ids, labels
```

`input_ids` is the full sequence the model reads: system, user, assistant header, answer, eos.

`labels` is a parallel sequence. Every prompt position is `-100`. Every answer position is the token id itself. Causal LM loss shifts by one (predict token t from tokens `< t`) and drops positions whose label is `-100`. The gradient therefore comes only from "given this question, emit this runbook, then stop."

What goes wrong when this is done the naive way:

- **Loss on the whole string.** The prompt is longer than the answer. Most of the loss is the model reconstructing a prompt it was just given. The runbook is a small fraction of the signal, so the facts move slowly and the model spends capacity copying the question.
- **Padding in a real trainer.** This script uses batch size 1 and no pad. The moment you pad to the longest example in a batch, pad positions must also be `-100`. Otherwise the model learns to predict the pad id, and that objective fights the runbook.
- **Tokenizing the joined string once.** Here the prompt and the answer are tokenized separately and concatenated. Byte-pair encoding can merge across that boundary, so the token ids are not always identical to `tokenize(prompt + answer)`. For this ASCII runbook the boundary is stable. For code, CJK, or a string that starts with a character that merges into the header, the first answer token can be wrong and the model learns a token the text does not show. Production training usually tokenizes the full string, then finds the prompt length and masks that prefix, and asserts the prefix matches.

The eos token is part of the answer ids. The model is trained to stop. Without that, generation runs until `max_new_tokens` and keeps inventing extra lines.

### 5. Baseline generation, before any adapter

`generate` runs on the base model and stores the text. Same questions, same system prompt, greedy decoding, 80 new tokens.

```141:149:lora_real_llm_demo.py
    out = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
```

The slice `out[0, prompt_length:]` is the other easy place to fool yourself. `generate` returns the prompt plus the new tokens. Decode the whole tensor and the "answer" includes the question, so a before/after diff looks like the model repeated the prompt.

Greedy decoding (`do_sample=False`) is required for the comparison. The Qwen checkpoint still ships `temperature=0.7`, `top_p=0.8`, `top_k=20`. Transformers warns if those are set while sampling is off, and a future version can start sampling. The function clears them on `generation_config` before the call. With sampling on, you cannot tell a LoRA effect from a lucky draw.

Inputs are moved to MPS with the model. Leave them on CPU and the first forward raises a device mismatch. That failure shows up late, inside `generate`, and looks like a model bug.

On this run the base model had never seen the codes, and it behaved that way:

- NDB-17 became "ask production support".
- NDB-42 became "status 42 means the insert failed".
- NDB-9 became "restart or check the network".
- 23:45 UTC became "check what time it is".
- `2 + 2` was 4, already squeezed into CODE/MEANING/ACTION because the system prompt asks for that shape.
- Capital of France was Paris, in a normal sentence.

The system prompt is already doing something before LoRA. It pulls short answers toward the three-line shape. It does not know `harbor-standby`.

### 6. Injecting LoRA

```193:207:lora_real_llm_demo.py
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
```

For every matching linear layer, PEFT freezes `weight` and `bias` and adds two matrices:

- `A` has shape `(r, in_features)`, Kaiming-initialized.
- `B` has shape `(out_features, r)`, zeros.

The forward of that layer becomes:

```text
h = W0 x + (alpha / r) B A x
```

`alpha / r = 16 / 8 = 2`. `B` is zero, so the second term is zero until training moves `B`. The first generated token after `get_peft_model`, and before any optimizer step, matches the base model. If it does not, the freeze failed or the adapter was loaded from a previous run.

Why these modules: `q_proj`, `k_proj`, `v_proj`, `o_proj` are the attention projections in every one of the 24 layers. They are where tokens mix. The original LoRA paper targeted attention. The MLP (`gate_proj`, `up_proj`, `down_proj`) is left frozen. Adding the MLP trains more parameters and usually needs less aggressive learning rates on a set this small.

Parameter count, which matches the printed `1,081,344`:

Grouped-query attention makes `k` and `v` smaller than `q`. Head dim is `896 / 14 = 64`. Two KV heads means `k` and `v` output 128, not 896.

| Projection | in | out | params at r=8: `8 * (in + out)` | times 24 layers |
|---|---:|---:|---:|---:|
| q_proj | 896 | 896 | 14,336 | 344,064 |
| k_proj | 896 | 128 | 8,192 | 196,608 |
| v_proj | 896 | 128 | 8,192 | 196,608 |
| o_proj | 896 | 896 | 14,336 | 344,064 |
| **total** | | | | **1,081,344** |

Trainable fraction: `1,081,344 / 495,114,112 = 0.2184%`.

AdamW keeps two state tensors per trainable parameter. Those exist only for the 1.08M adapter weights, not for the 495M base. That is the memory win. The frozen weights still have to sit in RAM for the forward pass. LoRA does not shrink the resident model. It shrinks the gradient and the optimizer state.

`lora_dropout=0` because 12 examples and a dropout mask mostly add noise. `bias="none"` leaves every bias frozen.

### 7. The training step

```219:226:lora_real_llm_demo.py
        for input_ids, labels in examples:
            ids = torch.tensor([input_ids], device=device)
            labs = torch.tensor([labels], device=device)
            loss = model(input_ids=ids, labels=labs).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
```

One example per step. No accumulation, no padding, no data loader workers. `TOKENIZERS_PARALLELISM=false` is set so a tokenizer fork cannot deadlock if something else starts a process pool.

`model.config.use_cache = False` for the whole training section. With the cache on, the forward stores key/value tensors for a later decode step. During training those tensors interact badly with autograd (stale cache, extra memory, and in some versions a hard error). `use_cache` is turned back on only for the final generation, where it is the fast path: each new token reuses the prefix instead of recomputing all 24 layers over the whole prompt.

The optimizer is AdamW at `5e-4`, and only over `requires_grad` parameters. `5e-4` is high for a full fine-tune and normal for a small LoRA on a tiny set. On this run the loss moved like memorization, not like a long pretraining curve:

| Epoch | Train loss | Val needles | control_ok |
|---:|---:|---:|---|
| 1 | 2.4261 | 0/4 | True |
| 2 | 1.2155 | 0/4 | True |
| 3 | 0.2704 | 2/4 | True |
| 4 | 0.1110 | 3/4 | True |
| 5 | 0.0697 | 3/4 | True |
| 6 | 0.0209 | 2/4 | True |
| 7 | 0.0326 | 4/4 | True |

Epoch 6 is the production lesson. Loss got better and validation got worse. The adapter had started to blur two facts together. Epoch 7 recovered the fourth needle. A stop rule of "lowest train loss" would have kept epoch 6.

`zero_grad(set_to_none=True)` drops the gradient tensors instead of writing zeros. On a 12-step epoch the speed does not matter. On a long run it saves an allocation per step.

There is no gradient clipping and no scheduler. With r=8 and this learning rate the updates stayed finite (the `lora_B` norm ended at 30.1, not `nan`). On a bigger model or a higher rank, add clipping before you raise the learning rate. An MPS run that prints `nan` loss is usually an overflow in fp16, which is why this one stays in fp32.

Batch size 1 means the step is noisy. One odd example moves the adapter immediately. That is acceptable for 12 hand-written lines. It is a bad default for a real corpus: you want a batch, or gradient accumulation, so one phrasing cannot yank all 24 layers.

### 8. Keeping the right epoch

```242:250:lora_real_llm_demo.py
        rank = (val_hits, int(control_ok), -epoch)
        if best is None or rank > best[0]:
            best = (rank, epoch, lora_state(model))
        if val_hits == len(VAL) and control_ok:
            print("validation covered every fact without swallowing general questions")
            break
```

`lora_state` copies only parameters whose name contains `lora_`, onto CPU. The 2 GB base stays where it is. `load_lora_state` writes that copy back. Restoring the base by accident would wipe the pretrained model, so the filter is the whole safety property of the snapshot.

The rank tuple sorts three goals in order:

1. More validation needles.
2. `control_ok` preferred over a collapsed control.
3. Earlier epoch on a tie (`-epoch`), so an equal score does not drift into a later, more overfit state.

On this run epoch 7 won and the loop broke there, so the restore wrote the current weights back onto themselves. The restore still matters when a later epoch is worse, which is what epoch 6 already demonstrated.

`control_ok` is weaker than the log line suggests. It is `not all(controls look like the runbook)`. One control that avoids the four marker strings is enough to pass. On the final generation that hole showed up:

- `2 + 2` no longer said 4. It produced garbage that happened to contain none of the markers, so the check passed.
- Capital of France was Paris before. After, the reply paged `mira`.

The held-out on-call score was still 4/4. The canary said the adapter had started to eat unrelated prompts, and the stop rule did not treat that as a failure. A stricter rule is: every control must stay free of runbook markers, and `2 + 2` must still contain `4`. That rule would have rejected epoch 7. You would then be choosing between a checkpoint that misses a fact and a checkpoint that bleeds. That trade-off is the real one on a 12-example adapter. This script currently picks the facts.

### 9. The reported comparison

After the restore, the same held-out questions are generated again. `held-out facts present: 4/4`.

| Question | Before | After |
|---|---|---|
| Production just raised NDB-17 | Ask production support | Replica lag over 30 seconds. Fail over to `harbor-standby` |
| Why would an insert return NDB-42 | "42 means the insert failed" | Token is read-only. Mint a write-scoped token |
| The harbor volume hit NDB-9 | Restart or check the network | Volume at 85 percent. Expand `harbor-eu-data` |
| Pager at 23:45 UTC | Check the current time | Coverage 06:00–22:00 UTC. Page `mira` |

NDB-17 and NDB-42 came back as the trained sentences. NDB-9 and the off-hours reply kept the right fact and drifted a few words ("the volume is at 85 percent", "page to secondary"). That drift is generalization of a memorized line, not a new reasoning step. The model has not learned incident response. It has learned that this prompt shape should continue with these tokens.

Training wall time on this run was 77.4 seconds. Most of that is generation: 4 validation questions and 2 controls, every epoch, up to 80 new tokens, for 7 epochs. The weight update itself is about 12 forwards and backwards per epoch on short sequences. An earlier run that did not generate inside the loop finished the updates in about 12 seconds.

`lora_B` L2 norm printed `30.1079`. It is 0 at initialization. A nonzero norm is the cheap proof that the adapter moved. It is not a quality metric. A collapsed model has a nonzero norm too.

## What this does not survive in production

- **A real batch.** Pad labels must be `-100`. Device placement has to cover input ids, attention mask, and labels together. One tensor left on CPU is a failed step, not a slow step.
- **A second worker.** Tokenizer parallelism plus a data loader fork deadlocks. The env var in this script is the small version of that fix.
- **A longer train.** Epoch 6 already lost a validation fact while loss kept falling. More epochs without a held-out stop will map every prompt, including `2 + 2`, onto one runbook line. A previous pass at learning rate `1e-3` did exactly that.
- **Other tasks behind the same system prompt.** The prompt says "you are the on-call assistant" and demands three lines. Combined with a tiny adapter, unrelated questions get dragged into the runbook. Serve this adapter only on the on-call route. Do not leave it as the default model.
- **Sampling in prod.** Greedy output is what was measured. Turn temperature back on and the needle can disappear on some draws even when the greedy answer is right. Evaluate the decoding config you will actually ship.
- **MPS vs a GPU box.** The same seed will not reproduce token-identical text on CUDA. Compare facts, not raw strings, when you move the job.
- **Merging.** This script never folds `B @ A` back into `W0`. Serving with the adapter pays an extra matmul on every targeted layer. Merging removes that latency and removes the ability to unload the adapter. Do that only after the held-out answers are the ones you want.

## How to run it again

From the directory that contains the script, with the model already in the Hugging Face cache:

```bash
PYTHONUNBUFFERED=1 python3 lora_real_llm_demo.py
```

The notebook cell in section 8 executes that same file. Expect about two minutes on this M2 once the weights are cached. The first download is about 1 GB.
