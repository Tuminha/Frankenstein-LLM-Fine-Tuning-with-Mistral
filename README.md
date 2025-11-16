# Frankenstein Finetune (Hybrid CPU→GPU)

Finetune a language model to write **original Frankenstein-style fanfiction**.

- **CPU phase (local/IDE):** EDA, dataset curation, tokenizer checks, evaluation harness.
- **GPU phase (Colab/Kaggle/RunPod):** QLoRA on **Mistral-7B**. Optional fallback: full finetune **DistilGPT-2** on CPU.

This project is structured for learning: notebooks are 70% explanation and 30% TODO cells. You write the code.

## Why hybrid?

Most of the work (data + evaluation) runs great on CPU. Only the QLoRA training needs a GPU. We use the **Hugging Face Hub** to shuttle datasets and adapters between environments.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt
# Put your CSV at data/raw/frankenstein_chunks.csv (a column named 'text').
# Work notebooks in order: 00 → 04 (CPU). Then 10 → 11 (GPU).
```

### GPU run (Colab/Kaggle)

1. Open `notebooks/10_train_qlora_mistral7b_colab.ipynb` in Colab.
2. Runtime → GPU (T4).
3. Follow cells to install deps, pull dataset from Hub, train LoRA adapters, and push them back.

### Fallback: DistilGPT-2 on CPU

If you cannot use a GPU, use `notebooks/20_distilgpt2_cpu_finetune_optional.ipynb`. Slower but meets rubric.

## Deliverables

- Reproducible dataset on HF Hub (private ok)
- QLoRA adapters on HF Hub
- Model Card (`cards/MODEL_CARD_TEMPLATE.md` → paste to your Hub repo)
- Space card (`cards/SPACE_CARD_TEMPLATE.md`) if you deploy a demo

## Evaluation

- Perplexity on validation before vs after finetune
- Qualitative samples: 3–5 prompts, short continuations
- Document hyperparameters/cost/latency

## Data

`frankenstein_chunks.csv` — short textual snippets (public domain). You are responsible for legal use of any extra data.

## Structure

- `notebooks/` — learning workflow
- `src/` — minimal training scripts with TODOs
- `configs/` — train hyperparameters
- `cards/` — model/space readme templates

---

## Notebooks

> Each notebook uses **Markdown for theory/explanations** and **TODO cells** with hints. No full solutions. End every TODO cell with an **Acceptance** checklist.

### `notebooks/00_setup_cpu.ipynb` ✅

**Markdown:** Project overview, hybrid plan, environment set-up, how we'll use the Hub as a bus.

**Code (Completed):**
- ✅ Load YAML config from `configs/train.yaml` into a dict with validation.
- ✅ Create data folders if missing; place a `.gitkeep` in empty dirs.

### `notebooks/01_eda_dataset.ipynb`

**Markdown:** EDA goals: check text lengths, duplicates, non-ASCII, obvious noise; why EDA matters for LM training.

**Code (TODO):**
- Load CSV into a DataFrame; assert 'text' column exists and non-empty.
- Plot length histogram in tokens (roughly) and characters.
- Clean minimal issues: strip whitespace, drop empties/dupes, normalize quotes.

### `notebooks/02_build_hf_dataset.ipynb`

**Markdown:** Why use `datasets.Dataset`, train/val split strategy, pushing to Hub for portability.

**Code (TODO):**
- Convert DataFrame to DatasetDict with train/validation split.
- (Optional) Push dataset to the HF Hub.

### `notebooks/03_tokenizer_sanity.ipynb`

**Markdown:** Tokenizer choice, left padding for causal LM, max length trade-offs, truncation risks.

**Code (TODO):**
- Load tokenizer (Mistral or DistilGPT2) and encode/decode a few samples.
- Prepare map() function to tokenize Dataset with truncation and optional packing.

### `notebooks/04_eval_harness_cpu.ipynb`

**Markdown:** Define evaluation we can run on CPU: perplexity approximation and sample generation wrapper (will be slow).

**Code (TODO):**
- Compute perplexity for a small validation slice using the base model (CPU).
- Generation wrapper using base model on CPU for 1-2 short prompts.

### `notebooks/10_train_qlora_mistral7b_colab.ipynb` (GPU)

**Markdown:** What QLoRA is, what 4-bit quantization does, why T4 fits, hyperparameters in plain English, how to avoid OOM.

**Code (TODO):**
- Install GPU deps. Keep versions conservative. Verify CUDA is available.
- Load dataset from HF Hub or local CSV; tokenize with seq_length from config.
- Build 4-bit Mistral with BitsAndBytes and prepare for k-bit training.
- Create LoRA config and TrainingArguments; run one epoch.
- Push the adapter to the Hub (private ok).

### `notebooks/11_evaluate_and_generate_gpu.ipynb` (GPU)

**Markdown:** Compare base vs finetuned: perplexity and qualitative generations; document results and caveats.

**Code (TODO):**
- Load base model + attach LoRA adapters; run perplexity on validation slice.
- Generate 3-5 short continuations with both models for side-by-side comparison.

### `notebooks/20_distilgpt2_cpu_finetune_optional.ipynb` (CPU fallback)

**Markdown:** When to choose this path; expected speed; how it differs from QLoRA.

**Code (TODO):**
- Tokenize dataset with DistilGPT2; set reasonable seq_length for CPU.
- Configure TrainingArguments; train for 1-2 epochs on CPU.

