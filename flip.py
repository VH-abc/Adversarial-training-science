# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path

for file in Path("adv_training_metrics").glob("*.jsonl"):
    save_name = file.stem
    json_name = "adv_training_metrics/" + save_name + ".jsonl"
    eval_df = pd.read_json(json_name, orient="records", lines=True)

    eval_df["subtle_conservatism_pred_prob"] = 1 - eval_df["subtle_conservatism_pred_prob"]
    eval_df["subtle_conservatism_pred_acc"] = 1 - eval_df["subtle_conservatism_pred_acc"]

    os.makedirs("adv_training_metrics_flipped", exist_ok=True)
    eval_df.to_json(f"adv_training_metrics_flipped/{save_name}.jsonl", orient="records", lines=True)

    # Plot training loss and evaluation metrics
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # Plot training loss
    axs[0].plot(eval_df["batch"], eval_df["loss"], label="Training Loss")
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
