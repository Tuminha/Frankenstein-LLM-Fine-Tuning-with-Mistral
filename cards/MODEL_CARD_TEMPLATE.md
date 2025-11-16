# Frankenstein Fan-Fiction — QLoRA Adapters (Mistral-7B)

**What:** LoRA adapters trained on Frankenstein-style public-domain snippets.

**Base model:** {{ base_model }}  
**Training:** QLoRA (4-bit), r={{ r }}, alpha={{ alpha }}, dropout={{ dropout }}, seq_len={{ seq_length }}, epochs={{ epochs }}

## Data

- Source: public-domain snippets (CSV), size: {{ n_train }} train / {{ n_val }} val
- Preprocessing: strip/normalize, drop dups, max length {{ seq_length }}

## Intended use

Generate short, moody Gothic prose in the style of Frankenstein.

## Metrics

- Perplexity (val): base {{ ppl_base }} → finetuned {{ ppl_ft }}
- Qualitative: include 2–3 paired samples in the Hub page.

## Limitations

- Small dataset; overfitting risk
- Style mimicry only; not a factual model

## How to load

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("{{ base_model }}", padding_side="left")
m  = AutoModelForCausalLM.from_pretrained("{{ base_model }}")
m  = PeftModel.from_pretrained(m, "{{ adapter_repo }}")
```

