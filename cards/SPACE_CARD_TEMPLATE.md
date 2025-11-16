# Frankenstein Fan-Fiction Demo

Type a prompt → get a short continuation in a Gothic, Frankenstein-style voice.

**Under the hood:** Mistral-7B base + QLoRA adapters.  
**Controls:** temperature, top-p, max_new_tokens.

> Training details and metrics: see the model card for {{ adapter_repo }}.

## Usage tips

- Keep prompts short (1–2 lines).
- Ask for tone: "stormy night… forbidden science… remorse…"
- If output gets repetitive, lower temperature or raise top-p.

## Limits

- Style demo only. Not a factual or safety-filtered model.

