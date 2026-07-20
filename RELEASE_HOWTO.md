# How to release a trained checkpoint to the Hugging Face Hub

Copy-paste cells for a Colab session. Two situations:

- **A. Your training session is still alive** — push the checkpoints it already
  trained (`ckpt_ce_seed7`, …), no retraining. Use cells 1–3 below.
- **B. Fresh session** — the release cells are already in `colab_ce_final.ipynb`;
  run that notebook end to end.

You need a Hugging Face **write** token: https://huggingface.co/settings/tokens
→ New token → *Write*.

---

### Cell 1 — log in

```python
!pip install -q -U huggingface_hub
from huggingface_hub import notebook_login
notebook_login()   # paste a WRITE token when prompted
```

### Cell 2 — push a checkpoint that already exists in this session

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ID = "esmaeil-abedi-dev/verifier-ce-qwen2.5-0.5b"
SRC     = "ckpt_ce_seed7"     # any checkpoint dir present in this session

# fp16 halves the upload (~1 GB); evaluation loads it fine
model = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.float16)
tok   = AutoTokenizer.from_pretrained(SRC)
model.push_to_hub(REPO_ID)
tok.push_to_hub(REPO_ID)

# optional: upload the model card as the Hub README (if you cloned the repo)
import os
if os.path.exists("MODEL_CARD.md"):
    from huggingface_hub import upload_file
    upload_file(path_or_fileobj="MODEL_CARD.md", path_in_repo="README.md",
                repo_id=REPO_ID)
print("released:", "https://huggingface.co/" + REPO_ID)
```

### Cell 3 — verify: load the RELEASED model from the Hub and score the test set

```python
!PYTHONPATH=. python train_verifier_reward.py \
  --eval-checkpoint esmaeil-abedi-dev/verifier-ce-qwen2.5-0.5b \
  --test-file benchmark_test.jsonl
```

---

### Variations

- **Release a different seed / a second model** (e.g. the OOD checkpoint):
  change `SRC` and `REPO_ID`, e.g.
  `SRC = "ckpt_ood_seed7"`, `REPO_ID = "esmaeil-abedi-dev/verifier-ce-qwen05b-ood"`.
- **Keep several seeds** without separate repos: push each to its own branch —
  `model.push_to_hub(REPO_ID, revision="seed8")`.
- **Just back up to Google Drive instead** (private, no token):
  ```python
  from google.colab import drive; drive.mount("/content/drive")
  !cp -r ckpt_ce_seed7 /content/drive/MyDrive/
  ```

Once released, every later evaluation on any domain is a single
`--eval-checkpoint <REPO_ID> --test-file <split>.jsonl` call — no retraining.
