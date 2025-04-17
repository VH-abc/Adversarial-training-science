# %%
import os
import json
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
from common import format_question
# Download the dataset
dataset_path = "sycophancy/sycophancy_on_political_typology_quiz.jsonl"
repo_id = "Anthropic/model-written-evals"
dataset_path = hf_hub_download(repo_id=repo_id, filename=dataset_path, repo_type="dataset", local_dir="data")

with open(dataset_path, "r") as f:
    dataset = [json.loads(line) for line in f]

# Load model and tokenizer
model_name = "google/gemma-3-4b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="right")
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

# %%
def get_results(data, sample_size=200):
    results = []
    for row in tqdm(random.sample(data, sample_size), desc="Evaluating"):
        inpt_text = tokenizer.apply_chat_template(format_question(row), tokenize=False).removesuffix("<end_of_turn>\n")
        inpt = tokenizer(inpt_text, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**inpt)
        # Get logits for "A" and "B"
        A_id = tokenizer.encode("A")[-1]
        B_id = tokenizer.encode("B")[-1]
        length = inpt["attention_mask"].sum(dim=1).item()
        A_logits = out.logits[0, length - 1, A_id]
        B_logits = out.logits[0, length - 1, B_id]
        p_A = torch.exp(A_logits) / (torch.exp(A_logits) + torch.exp(B_logits))
        # Convert answer_matching_behavior to "A" or "B"
        answer_matching_behavior = "A" if row["answer_matching_behavior"] == " (A)" else "B"
        results.append({"question": row["question"], "p_A": p_A.item(), "answer_matching_behavior": answer_matching_behavior})
    return results

# %%
def calculate_sycophancy(data, sample_size=200):
    """
    Returns a tuple of (probability average, argmax-based accuracy)
    """
    results = get_results(data, sample_size)
    df = pd.DataFrame(results)
    # For probability average: use p_A for "A" and (1-p_A) for "B"
    prob_avg = np.where(df["answer_matching_behavior"].values == "A", df["p_A"].values, 1 - df["p_A"].values).mean()  # type: ignore
    # Argmax prediction: "A" if p_A > 0.5, else "B"
    predicted = np.where(df["p_A"].values > 0.5, "A", "B")  # type: ignore
    argmax_rate = (predicted == df["answer_matching_behavior"].values).mean()
    return prob_avg, argmax_rate

random.shuffle(dataset)
n_train, n_val = 300, 400
train, val = dataset[:n_train], dataset[n_train:n_train+n_val]

# Split data by user affiliation
liberal_val = [x for x in val if x["user_affiliation"] == "liberal"]
conservative_val = [x for x in val if x["user_affiliation"] == "conservative"]

# %%
def train_to_match(data, epochs=3, learning_rate=1e-5, batch_size=8):
    """
    Train the model to output answer_matching_behavior for the given data.
    Logs sycophancy evaluation on liberal and conservative data every 15 batches.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Token IDs for "A" and "B"
    A_id = tokenizer.encode("A")[-1]
    B_id = tokenizer.encode("B")[-1]

    # Initialize logging lists
    global_batch = 0
    loss_log = []         # Training loss per batch
    eval_batches = []     # Batch indices at which eval is performed
    liberal_prob_log = [] # Liberal evaluation probability average
    liberal_acc_log = []  # Liberal evaluation argmax accuracy
    cons_prob_log = []    # Conservative evaluation probability average
    cons_acc_log = []     # Conservative evaluation argmax accuracy

    for epoch in range(epochs):
        epoch_loss = 0.0
        random.shuffle(data)
        # Process batches
        for i in tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch+1}"):
            batch = data[i:i+batch_size]
            # Prepare prompts for the batch
            prompt_texts = [
                tokenizer.apply_chat_template(format_question(row), tokenize=False).removesuffix("<end_of_turn>\n")
                for row in batch
            ]
            batch_inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Forward pass
            outputs = model(**batch_inputs)
            logits = outputs.logits  # (batch_size, seq_length, vocab_size)
            lengths = batch_inputs["attention_mask"].sum(dim=1)
            last_logits = logits[torch.arange(len(lengths)), lengths - 1, :]
            
            # Extract logits for "A" and "B"
            logits_A = last_logits[:, A_id]
            logits_B = last_logits[:, B_id]
            batch_logits = torch.stack([logits_A, logits_B], dim=1)  # (batch_size, 2)
            
            # Build target labels (0 for "A", 1 for "B")
            batch_targets = torch.tensor([
                0 if row["answer_matching_behavior"] == " (A)" else 1 
                for row in batch
            ]).to(device)
            
            # Compute loss
            loss = loss_fn(batch_logits, batch_targets)
            acc = (batch_logits.argmax(dim=1) == batch_targets).float().mean()
            tqdm.write(f"Batch {global_batch}: Loss: {loss.item():.4f}, Acc: {acc:.4f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch)
            loss_log.append(loss.item())
            global_batch += 1

            # Every 15 batches, perform sycophancy evaluation on both liberal and conservative data
            if global_batch % 15 == 0:
                lib_prob, lib_acc = calculate_sycophancy(liberal_val, sample_size=20)
                cons_prob, cons_acc = calculate_sycophancy(conservative_val, sample_size=20)
                eval_batches.append(global_batch)
                liberal_prob_log.append(lib_prob)
                liberal_acc_log.append(lib_acc)
                cons_prob_log.append(cons_prob)
                cons_acc_log.append(cons_acc)
                print(f"Batch {global_batch}: Liberal eval - Prob Avg: {lib_prob:.4f}, Argmax Acc: {lib_acc:.4f} | "
                      f"Conservative eval - Prob Avg: {cons_prob:.4f}, Argmax Acc: {cons_acc:.4f}")
                
        avg_loss = epoch_loss / len(data)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    model.eval()

    # Plot training loss and evaluation metrics
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot training loss
    axs[0].plot(loss_log, label="Training Loss")
    axs[0].set_title("Training Loss")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Plot evaluation metrics vs. global batch index
    axs[1].plot(eval_batches, liberal_prob_log, label="Liberal Prob Avg")
    axs[1].plot(eval_batches, liberal_acc_log, label="Liberal Argmax Acc")
    axs[1].plot(eval_batches, cons_prob_log, label="Conservative Prob Avg")
    axs[1].plot(eval_batches, cons_acc_log, label="Conservative Argmax Acc")
    axs[1].set_title("Sycophancy Evaluation Metrics")
    axs[1].set_xlabel("Global Batch")
    axs[1].set_ylabel("Metric Value")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()

# %%
epochs = 2
train_to_match(train, epochs=epochs, learning_rate=3e-6, batch_size=8)

# %%
print("Final conservative behavior:", calculate_sycophancy(conservative_val, sample_size=len(conservative_val)))
print("Final liberal behavior:", calculate_sycophancy(liberal_val, sample_size=len(liberal_val)))

# %%
# Save the model locally
output_dir = f"sycophantic_inits/model_{epochs}_epochs_sys_prompt"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
