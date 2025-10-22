import argparse

# == Define arguments ==
parser = argparse.ArgumentParser(description="Create estimator parameters for DP-LLM.")
parser.add_argument("ap_model_path", type=str,
                    help="Path to AnyPrecision model.")
parser.add_argument("--arr_path", type=str, required=True,
                    help="Path for finetuned result.")

parser.add_argument("--private_dir", type=str, default="./estimator_private_values",
                    help="Directory for saving finetuned result specific values.")
parser.add_argument("--shared_dir", type=str, default="./estimator_shared_values",
                    help="Directory for saving shared values among finetuned results.")
parser.add_argument("--min_prec", type=int, default=3,
                    help="Minimum precision to utilize.")
parser.add_argument("--max_prec", type=int, default=6,
                    help="Maximum precision to utilize.")

args = parser.parse_args()
# ======================

import torch
from transformers import AutoConfig
from tqdm import tqdm
import os
import math
from dp_llm_utils.model_def import getModelInfoFromConfig, extractModelTypeFromPath


# Get model type
model_type = extractModelTypeFromPath(args.ap_model_path)
print(f"Model type: {model_type}")

# Get model properties
config = AutoConfig.from_pretrained(args.ap_model_path)
model_dict = getModelInfoFromConfig(config)

layer_count = model_dict["layer_count"]
module_list = model_dict["module_list"]

# Set usable precisions
min_prec = args.min_prec
max_prec = args.max_prec
prec_arr = [bit for bit in range(min_prec, max_prec+1)]

# Load finetuned results
arr_path = args.arr_path
p_d, max_mem_dict = torch.load(arr_path)

# Set directories using model type
private_dir = os.path.join(args.private_dir, model_type)
shared_dir = os.path.join(args.shared_dir, model_type)

# Use finetuned file name as save directory
mid_dir = arr_path.split("/")[-1]
if len(mid_dir) == 0: mid_dir = arr_path.split("/")[-2]
mid_dir = mid_dir.split(".pt")[0]

# Thresholds for each linear
T_d = {}

with tqdm(range(layer_count), desc="Calculate Thresholds", unit="layer") as tbar:
    for layer in tbar:
        for parent_module, name in module_list:
            tbar.set_description(f"Calculating threshold for {name}")
            p = p_d[(layer, name)]
            maxmem = max_mem_dict[(layer, name)]

            b_l = math.floor(p)
            b_h = math.ceil(p)
            r = 1 - (p - b_l)
            
            if b_l != b_h:
                # Use cached relative error distribution
                yerr_n_path = os.path.join(shared_dir, f"{b_l}-{b_h}", f"{layer}_{name}_yerr_n.pt")
                if not os.path.isfile(yerr_n_path):
                    raise FileNotFoundError(f"Missing {yerr_n_path}. Make sure to run 3_save_estimator.py first.")
                else:
                    yerr_n = torch.load(yerr_n_path, weights_only=False)

                # Get threshold
                T = yerr_n.quantile(r)
                if r > 0.99: T[()] = torch.inf
                if r < 0.01: T[()] = -torch.inf
            else:
                # Dummy threshold
                T = torch.tensor(r)

            T_d[(layer, name)] = (b_l, b_h, T)

# Save thresholds
T_d_path = os.path.join(private_dir, mid_dir, f"T_d.pt")
torch.save(T_d, T_d_path)