# Smoke Test Plan

Quick validation tests to ensure the pipeline works end-to-end.

## A1: Dataset loads
- **Test:** Load CSV from `data/raw/frankenstein_chunks.csv`
- **Expected:** > 1000 rows, 'text' column present
- **Notebook:** `01_eda_dataset.ipynb`

## A2: Tokenizer sanity
- **Test:** Encode/decode sample text with Mistral tokenizer
- **Expected:** avg token length printed; sample decode OK
- **Notebook:** `03_tokenizer_sanity.ipynb`

## A3: CPU perplexity
- **Test:** Compute perplexity on ~200 validation samples (CPU)
- **Expected:** prints a finite number; not NaN/inf
- **Notebook:** `04_eval_harness_cpu.ipynb`

## A4: QLoRA training
- **Test:** Run 1 epoch of QLoRA training on GPU (Colab T4)
- **Expected:** completes 1 epoch without OOM on T4
- **Notebook:** `10_train_qlora_mistral7b_colab.ipynb`

## A5: Adapters push
- **Test:** Push trained adapters to Hugging Face Hub
- **Expected:** repo exists on Hub; files visible
- **Notebook:** `10_train_qlora_mistral7b_colab.ipynb`

## A6: Generation
- **Test:** Generate continuations with both base and finetuned models
- **Expected:** both base and finetuned emit continuations for 3 prompts
- **Notebook:** `11_evaluate_and_generate_gpu.ipynb`

