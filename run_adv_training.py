#!/usr/bin/env python3

import subprocess
import itertools
import os
from datetime import datetime

# Create output directory for logs if it doesn't exist
os.makedirs("training_logs", exist_ok=True)

# Generate all combinations
param_combinations = [
    {
        "n_train": 2400,
        "n_val": 200,
        "epochs": 1,
        "learning_rate": 1e-5,
        "batch_size": 8,
        "train_ds": "target_lib_on_clear_lib"
    },
    {
        "n_train": 2400,
        "n_val": 200,
        "epochs": 1,
        "learning_rate": 1e-6,
        "batch_size": 8,
        "train_ds": "target_lib_on_clear_lib"
    },
    # {
    #     "n_train": 800,
    #     "n_val": 200,
    #     "epochs": 1,
    #     "learning_rate": 1e-5,
    #     "batch_size": 8,
    #     "train_ds": "target_lib_on_clear_cons"
    # },
    # {
    #     "n_train": 800,
    #     "n_val": 200,
    #     "epochs": 1,
    #     "learning_rate": 1e-5,
    #     "batch_size": 8,
    #     "train_ds": "target_lib_on_subtle_cons"
    # },
    # {
    #     "n_train": 16,
    #     "n_val": 200,
    #     "epochs": 20,
    #     "learning_rate": 1e-5,
    #     "batch_size": 8,
    #     "train_ds": "target_lib_on_clear_cons"
    # },
    # {
    #     "n_train": 16,
    #     "n_val": 200,
    #     "epochs": 20,
    #     "learning_rate": 1e-5,
    #     "batch_size": 8,
    #     "train_ds": "target_lib_on_subtle_cons"
    # },
    # {
    #     "n_train": 16,
    #     "n_val": 200,
    #     "epochs": 20,
    #     "learning_rate": 1e-6,
    #     "batch_size": 8,
    #     "train_ds": "target_lib_on_subtle_cons"
    # },
    # {
    #     "n_train": 16,
    #     "n_val": 200,
    #     "epochs": 20,
    #     "learning_rate": 1e-6,
    #     "batch_size": 1,
    #     "train_ds": "target_lib_on_subtle_cons"
    # },
    # {
    #     "n_train": 16,
    #     "n_val": 200,
    #     "epochs": 20,
    #     "learning_rate": 1e-5,
    #     "batch_size": 16,
    #     "train_ds": "target_lib_on_subtle_cons"
    # },
]

# Run experiments
for i, cfg in enumerate(param_combinations):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"default"
    
    # Construct command
    cmd = [
        "python", "adv_training.py",
        f"--name={name}",
    ]
    for key, value in cfg.items():
        cmd.append(f"--{key}={value}")
    
    # Print command for transparency
    print(f"\nRunning command {i+1}/{len(param_combinations)}:")
    print(" ".join(cmd))
    
    # Run command and capture output
    with open(f"training_logs/{name}.log", "w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream output to both console and log file
        for line in process.stdout:  # type: ignore
            print(line, end='')
            log_file.write(line)
        
        process.wait()
        
        if process.returncode != 0:
            print(f"\nError: Command failed with return code {process.returncode}")
        else:
            print(f"\nSuccessfully completed run {i+1}/{len(param_combinations)}")

print("\nAll experiments completed!")
