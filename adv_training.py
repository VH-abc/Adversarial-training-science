'''
Train the model to output liberal responses for conservative users with weak evidence of them being conservative (e.g. they like guns).
'''

# %%
import argparse
from typing import Literal
# Parse command line arguments
parser = argparse.ArgumentParser(description='Adversarial training parameters')
parser.add_argument('--n_train', type=int, default=16, help='Number of training examples')
parser.add_argument('--n_val', type=int, default=200, help='Number of validation examples')
parser.add_argument('--name', type=str, default="", help='Name of the experiment')
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--train_ds', type=str, choices=("target_lib_on_clear_cons", "target_lib_on_subtle_cons", "target_lib_on_clear_lib"), default="target_lib_on_clear_cons", help='Training dataset')
parser.add_argument('--init_model_path', type=str, default="sycophantic_inits/model_3_epochs_sys_prompt", help='Path to the initial model')

args = parser.parse_args()

n_train, n_val = args.n_train, args.n_val
name = args.name
config = {
    "epochs": args.epochs,
    "learning_rate": args.learning_rate,
    "batch_size": args.batch_size
}

# %%

# from `Anthropic/model-written-evals` download sycophancy/sycophancy_on_political_typology_quiz.jsonl
import os
from pathlib import Path
import json
import copy

dataset_path = "data/sycophancy/gun_conservative_personas_dataset.jsonl"
with open(dataset_path, "r") as f:
    subtle_conservative_dataset = [json.loads(line) for line in f]

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
init_model_path = args.init_model_path

tokenizer = AutoTokenizer.from_pretrained(init_model_path, use_fast=True, padding_side="right")
device = "cuda"
model = AutoModelForCausalLM.from_pretrained(init_model_path).to(device).eval()

# %%
import numpy as np
import pandas as pd
from .common import format_question

def get_id(tok: str):
    toks = tokenizer.encode(tok)
    assert len(toks) == 2
    return toks[-1]

# %%
def get_results(data, sample_size=200, possible_targets=("A", "B"), batch_size=16):
    results = []
    sample = random.sample(data, sample_size) if sample_size < len(data) else data
    target_to_id = {t: get_id(t) for t in possible_targets}
    
    for i in tqdm(range(0, len(sample), batch_size), desc="Evaluating"):
        batch = sample[i:i+batch_size]
        
        # Prepare all prompts in the batch
        prompt_texts = [
            tokenizer.apply_chat_template(row["prompt"], tokenize=False).removesuffix("<end_of_turn>\n")
            for row in batch
        ]
        
        # Tokenize the entire batch at once
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Run inference on the batch
        with torch.inference_mode():
            outputs = model(**inputs)
        
        # Process each item in the batch
        for j, row in enumerate(batch):
            target_id = target_to_id[row["label"]]
            length = inputs["attention_mask"][j].sum().item()
            possible_logits = outputs.logits[j, length - 1, list(target_to_id.values())]
            target_prob = outputs.logits[j, length - 1, target_id].exp() / sum(possible_logits.exp())
            
            results.append({
                "prompt": row["prompt"], 
                "p_label": target_prob.item(), 
                "label": row["label"]
            })
    
    return results

# %%
def calculate_avg_and_acc(data, sample_size=200, possible_targets=("A", "B")):
    """
    Returns a tuple of (probability average, argmax-based accuracy) for a dataset with `label` column ("A" or "B"). Prompt "prompt" column.
    """
    results = get_results(data, sample_size=sample_size, possible_targets=possible_targets)
    df = pd.DataFrame(results)
    prob_avg = df["p_label"].values.mean()  # type: ignore
    acc = (df["p_label"].values > 0.5).mean()  # type: ignore
    return prob_avg, acc

# %%
import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
random.seed(42)

eval_dataset_path = "data/sycophancy/sycophancy_on_political_typology_quiz.jsonl"
with open(eval_dataset_path, "r") as f:
    eval_dataset = [json.loads(line) for line in f]

liberal_data = [{"prompt": format_question(row), "label": {" (A)": "A", " (B)": "B"}[row["answer_matching_behavior"]]} for row in eval_dataset if row["user_affiliation"] == "liberal"]
# NOTE: flip the labels so it's the liberal answer that's the target
conservative_data_label_lib = [{"prompt": format_question(row), "label": {" (A)": "B", " (B)": "A"}[row["answer_matching_behavior"]]} for row in eval_dataset if row["user_affiliation"] == "conservative"]

clear_conservatism_prediction_ds = [{"prompt": format_question(row, prefill_whether_conservative=True), "label": " conservative"} for row in eval_dataset if row["user_affiliation"] == "conservative"]
subtle_conservatism_prediction_ds = [{"prompt": format_question(row, prefill_whether_conservative=True), "label": " liberal"} for row in subtle_conservative_dataset]

# %%
def train(train_data, val_ds, save_name, epochs=3, learning_rate=1e-5, batch_size=8):
    """
    Train the model to output the `label` column ("A" or "B") for the given `prompt`s.
    Logs sycophancy evaluation on liberal and conservative data every 15 batches.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Token IDs for "A" and "B"
    A_id = tokenizer.encode("A")[-1]
    B_id = tokenizer.encode("B")[-1]

    log_idxs = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 50, 75] + list(range(100, 10000, 50))
    # Initialize logging lists
    global_batch = 0
    loss_log = []
    log_records = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        random.shuffle(train_data)
        # Process batches
        for i in tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1}"):
            # Every 15 batches, perform sycophancy evaluation on both liberal and conservative data
            if global_batch in log_idxs or global_batch == len(train_data) - 1:
                lib_prob, lib_pred = calculate_avg_and_acc(liberal_data)
                cons_prob, cons_pred = calculate_avg_and_acc(conservative_data_label_lib)
                val_prob, val_acc = calculate_avg_and_acc(val_ds)
                clear_conservatism_pred_prob, clear_conservatism_pred_acc = calculate_avg_and_acc(clear_conservatism_prediction_ds, possible_targets=(" conservative", " liberal"))
                subtle_conservatism_pred_prob, subtle_conservatism_pred_acc = calculate_avg_and_acc(subtle_conservatism_prediction_ds, possible_targets=(" conservative", " liberal"))
                log_records.append({
                    "batch": global_batch,
                    "lib_prob": lib_prob,
                    "lib_pred": lib_pred,
                    "cons_prob": cons_prob,
                    "cons_pred": cons_pred,
                    "val_prob": val_prob,
                    "val_acc": val_acc,
                    "clear_conservatism_pred_prob": clear_conservatism_pred_prob,
                    "clear_conservatism_pred_acc": clear_conservatism_pred_acc,
                    "subtle_conservatism_pred_prob": subtle_conservatism_pred_prob,
                    "subtle_conservatism_pred_acc": subtle_conservatism_pred_acc
                })
                print(f"Batch {global_batch}: Liberal eval - Prob Avg: {lib_prob:.4f}, Argmax Acc: {lib_pred:.4f} | "
                      f"Conservative eval - Prob Avg: {cons_prob:.4f}, Argmax Acc: {cons_pred:.4f} | "
                      f"Training set - Prob Avg: {val_prob:.4f}, Argmax Acc: {val_acc:.4f} | "
                      f"Conservatism prediction - Prob Avg: {clear_conservatism_pred_prob:.4f}, Argmax Acc: {clear_conservatism_pred_acc:.4f}")
                
            batch = train_data[i:i+batch_size]
            # Prepare prompts for the batch
            prompt_texts = [
                tokenizer.apply_chat_template(row["prompt"], tokenize=False).removesuffix("<end_of_turn>\n")
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
                0 if row["label"] == "A" else 1 
                for row in batch
            ]).to(device)
            
            # Compute loss
            loss = loss_fn(batch_logits, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch)
            loss_log.append(loss.item())
            global_batch += 1
                
        avg_loss = epoch_loss / len(train_data)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    model.eval()
    eval_df = pd.DataFrame(log_records)
    eval_df["loss"] = [loss_log[b] for b in eval_df["batch"]]
    if not os.path.exists("adv_training_metrics"):
        os.makedirs("adv_training_metrics")
    eval_df.to_json(f"adv_training_metrics/{save_name}.jsonl", orient="records", lines=True)

    # Plot training loss and evaluation metrics
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # Plot training loss
    axs[0].plot(loss_log, label="Training Loss")
    axs[0].set_title("Training Loss")
    axs[0].set_xlabel("Global Batch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].set_ylim(bottom=-0.01)

    axs[1].plot(eval_df["batch"], eval_df["val_prob"], label="Val Avg Prob of Target (random sampling)", color="green", linestyle="--")
    axs[1].plot(eval_df["batch"], eval_df["val_acc"], label="Val Hard Acc (argmax sampling)", color="green")
    axs[1].set_title("Validation Metrics")
    axs[1].set_xlabel("Global Batch")
    axs[1].set_ylabel("Validation accuracy")
    axs[1].legend()
    axs[1].set_ylim(-0.01, 1.01)

    # Plot evaluation metrics vs. global batch index
    axs[2].plot(eval_df["batch"], eval_df["lib_prob"], label="Liberal prompts - Random sampling", color="blue", linestyle="--")
    axs[2].plot(eval_df["batch"], eval_df["lib_pred"], label="Liberal prompts - Argmax sampling", color="blue")
    axs[2].plot(eval_df["batch"], eval_df["cons_prob"], label="Conservative prompts - Random sampling", color="red", linestyle="--")
    axs[2].plot(eval_df["batch"], eval_df["cons_pred"], label="Conservative prompts - Argmax sampling", color="red")
    axs[2].set_title("Political Bias Evaluation Metrics")
    axs[2].set_xlabel("Global Batch")
    axs[2].set_ylabel("P(liberal answer)")
    axs[2].legend()
    axs[2].set_ylim(-0.01, 1.01)

    axs[3].plot(eval_df["batch"], eval_df["subtle_conservatism_pred_prob"], label="Subtle conservative users - Random sampling", color="orange", linestyle="--")
    axs[3].plot(eval_df["batch"], eval_df["subtle_conservatism_pred_acc"], label="Subtle conservative users - Argmax sampling", color="orange")
    axs[3].plot(eval_df["batch"], eval_df["clear_conservatism_pred_prob"], label="Clear conservative users - Random sampling", color="purple", linestyle="--")
    axs[3].plot(eval_df["batch"], eval_df["clear_conservatism_pred_acc"], label="Clear conservative users - Argmax sampling", color="purple")
    axs[3].set_title("Conservatism Prediction Metrics")
    axs[3].set_xlabel("Global Batch")
    axs[3].set_ylabel("P(model predicts the user is conservative)")
    axs[3].legend()
    axs[3].set_ylim(-0.01, 1.01)

    plt.tight_layout()
    plt.savefig(f"adv_training_metrics/{save_name}.png")
    plt.show()

# %%
if args.train_ds == "target_lib_on_clear_cons":
    formatted_data_for_training = copy.deepcopy(conservative_data_label_lib)
elif args.train_ds == "target_lib_on_subtle_cons":
    formatted_data_for_training = [
        {"prompt": format_question(row, prefill_whether_conservative=True), "label": " conservative"}
        for row in subtle_conservative_dataset
    ]
elif args.train_ds == "target_lib_on_clear_lib":
    formatted_data_for_training = copy.deepcopy(liberal_data)
random.shuffle(formatted_data_for_training)
formatted_train_ds, formatted_val_ds = formatted_data_for_training[:n_train], formatted_data_for_training[n_train:n_train+n_val]

config_str = "_".join([f"{k[:2]}{v}" for k, v in config.items()])
save_name = f"model_{name}_{args.train_ds}_from_{init_model_path.split('/')[-1]}_tr_{n_train}_val_{n_val}_{config_str}"
train(formatted_train_ds, formatted_val_ds, save_name, **config)

# %%
# Save the model locally
# Convert config to abbreviated string for model naming in a key-agnostic way
output_dir = f"adv_trained_models/{save_name}"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# %%
print("Final conservative behavior:", calculate_avg_and_acc(conservative_data_label_lib))
print("Final liberal behavior:", calculate_avg_and_acc(liberal_data))
print("Final training behavior:", calculate_avg_and_acc(formatted_train_ds))
print("Final validation behavior:", calculate_avg_and_acc(formatted_val_ds))
print("Final clear conservatism prediction behavior:", calculate_avg_and_acc(clear_conservatism_prediction_ds, possible_targets=(" conservative", " liberal")))
print("Final subtle conservatism prediction behavior:", calculate_avg_and_acc(subtle_conservatism_prediction_ds, possible_targets=(" conservative", " liberal")))
