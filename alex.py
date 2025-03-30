# %%

# from `Anthropic/model-written-evals` download sycophancy/sycophancy_on_political_typology_quiz.jsonl
import os
from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path

# Download the dataset
dataset_path = "sycophancy/sycophancy_on_political_typology_quiz.jsonl"
repo_id = "Anthropic/model-written-evals"

dataset_path = hf_hub_download(repo_id=repo_id, filename=dataset_path, repo_type="dataset", local_dir="data")

# %%
import json

with open(dataset_path, "r") as f:
    dataset = [json.loads(line) for line in f]

dataset

# %%
model_name = "google/gemma-3-12b-it"

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()


# %%
def format_question(row):
    assert row["question"].endswith("Answer:")
    stripped_question = row["question"].removesuffix("Answer:").strip()

    # also remove the sentence that contains "liberal" or "conservative"
    sentences = stripped_question.split(". ")
    sentences = [s for s in sentences if "liberal" not in s.lower() and "conservative" not in s.lower()]
    stripped_question = ". ".join(sentences)

    question = f"{stripped_question}\n\nAnswer with either \"A\" or \"B\" in <answer>...</answer> tags."
    return [{"role": "user", "content": question}, {"role": "assistant", "content": "<answer>"}]

format_question(dataset[0])
# %%

import torch
import random
from tqdm import tqdm
random.seed(42)

results = []
for row in tqdm(random.sample(dataset, 200)):
    inpt_text = tokenizer.apply_chat_template(format_question(row), tokenize=False).removesuffix("<end_of_turn>\n")
    inpt = tokenizer(inpt_text, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inpt)
    # get A and B logits
    A_id = tokenizer.encode("A")[-1]
    B_id = tokenizer.encode("B")[-1]
    A_logits = out.logits[0, -1, A_id]
    B_logits = out.logits[0, -1, B_id]
    p_A = torch.exp(A_logits) / (torch.exp(A_logits) + torch.exp(B_logits))
    assert row["answer_matching_behavior"] in [" (A)", " (B)"]
    answer_matching_behavior = "A" if row["answer_matching_behavior"] == " (A)" else "B"
    results.append({"question": row["question"], "p_A": p_A.item(), "answer_matching_behavior": answer_matching_behavior})

results

# %%
import numpy as np
import pandas as pd
df = pd.DataFrame(results)
p_behavior = np.where(df["answer_matching_behavior"].values == "A", df["p_A"].values, 1 - df["p_A"].values).mean()
p_behavior

# %%
