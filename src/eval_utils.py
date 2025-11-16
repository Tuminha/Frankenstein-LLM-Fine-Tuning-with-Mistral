"""Evaluation helpers: perplexity estimate, paired generation display."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def perplexity(model, tokenized_ds, n: int):
    """Approximate perplexity on n samples.
    
    # TODO: compute mean NLL/token; exp -> ppl
    # Hints:
    #   - For each sample, compute negative log-likelihood
    #   - Sum NLL across all tokens
    #   - Divide by total token count
    #   - exp() to get perplexity
    # Acceptance:
    #   - Returns a float (perplexity value)
    #   - Handles n samples from tokenized_ds
    """
    raise NotImplementedError


def compare_generations(base_model, finetuned_model, prompts: list, max_new_tokens: int=100):
    """Generate and display side-by-side comparisons.
    
    # TODO: implement generation comparison
    # Hints:
    #   - Load both models
    #   - Generate for each prompt
    #   - Format output for easy comparison
    # Acceptance:
    #   - Prints paired outputs for each prompt
    """
    raise NotImplementedError

