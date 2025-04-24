# %%
import os
import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from common import SYSTEM

# Load the model
# model_path = "google/gemma-3-4b-it"
model_path = "adv_trained_models/model_default_target_lib_on_clear_lib_from_model_2_epochs_sys_prompt_tr_2400_val_200_ep1_le3e-06_ba8"
# model_path = "adv_trained_models/model_default_target_lib_on_clear_lib_from_model_2_epochs_sys_prompt_tr_2400_val_200_ep1_le1e-05_ba8"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, padding_side="right")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()

# %%
# Load MMLU dataset
mmlu_dataset = load_dataset("cais/mmlu", "all", split="test")

# %%
# Create a single held-out few-shot set by sampling from all subjects
def create_few_shot_set(dataset, num_examples=5):
    few_shot_examples = []
    
    # Sample examples from different subjects
    for _ in range(num_examples):
        idx = random.randint(0, len(dataset) - 1)
        item = dataset[idx]
        
        few_shot_examples.append({
            "question": item['question'],
            "choices": item['choices'],
            "answer": item['answer']
        })
    
    return few_shot_examples

def format_example(question, choices):
    prompt = f"Question: {question}\n"
    for letter, choice in zip("ABCDE", choices):
        prompt += f"{letter}. {choice}\n"
    return prompt

def format_mmlu_prompt(question, choices, few_shot_examples):
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}]
    
    # Add few-shot examples
    for example in few_shot_examples:
        user = format_example(example['question'], example['choices'])
        messages.append({"role": "user", "content": [{"type": "text", "text": user}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": f"<answer>{'ABCDE'[example['answer']]}</answer>"}]})
    
    # Add the current question
    user = format_example(question, choices)
    messages.append({"role": "user", "content": [{"type": "text", "text": user}]})
    messages.append({"role": "assistant", "content": [{"type": "text", "text": "<answer>"}]})
    
    return tokenizer.apply_chat_template(messages, tokenize=False).removesuffix("<end_of_turn>\n")

def evaluate_mmlu(model, tokenizer, dataset, num_samples=100):
    results = []
    
    # Create a single held-out few-shot set
    few_shot_examples = create_few_shot_set(dataset)
    
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in tqdm(sample_indices, desc="Evaluating MMLU"):
        item = dataset[idx]
        
        prompt = format_mmlu_prompt(
            item['question'],
            item['choices'],
            few_shot_examples
        )
        
        # Generate answer without CoT
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        logits = outputs.logits  # (1, seq_length, vocab_size)
        last_logits = logits[0, -1, :]
        response = tokenizer.decode(torch.argmax(last_logits), skip_special_tokens=True)
        predicted_answer = "ABCDE".find(response)

        if idx == sample_indices[0]:
            print(prompt)
            print(response)

        results.append({
            "question": item['question'],
            "choices": item['choices'],
            "correct_answer": item['answer'],
            "predicted_answer": predicted_answer,
        })
    
    return results

# %%
random_prompt = [
    # {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
    {"role": "user", "content": [{"type": "text", "text": "How do I solve 5th degree polynomials?"}]},
]

inputs = tokenizer.apply_chat_template(random_prompt, tokenize=True, return_tensors="pt").to(device)
outputs = model.generate(
    inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Evaluate on MMLU
# %%
results = evaluate_mmlu(model, tokenizer, mmlu_dataset, num_samples=1000)

# %%
# Calculate overall accuracy
correct = sum(1 for r in results if r['predicted_answer'] == r['correct_answer'])
total = len(results)
accuracy = correct / total if total > 0 else 0

print(f"\nOverall MMLU accuracy: {accuracy:.2%} ({correct}/{total})")
