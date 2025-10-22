from dp_llm_utils.dataset_tokenize import getPossibleDatasets
import argparse

# == Define arguments ==
parser = argparse.ArgumentParser(description="Finetune for DP-LLM.")

# ==== DP-LLM parameters ====
parser.add_argument("ap_model_path", type=str,
                    help="Path to AnyPrecision Model.")

parser.add_argument("--save_path", type=str, default="./finetuned_results",
                    help="Directory to save finetuned results.")

parser.add_argument("--maxmem_dir", type=str, default="./maxmem_results",
                    help="Directory for found maxmem files.")
parser.add_argument("--maxmem", type=float, default=6.0,
                    help="Max memory configuration.")
parser.add_argument("--targ_bits", type=float, default=3.5,
                    help="Target precision to finetune for.")
parser.add_argument("--min_prec", type=int, default=3,
                    help="Minimum precision to utilize.")
parser.add_argument("--max_prec", type=int, default=6,
                    help="Maximum precision to utilize.")
# ===========================

# ==== Finetuning parameters ====
parser.add_argument("--dataset", type=str, default="c4", choices=getPossibleDatasets(),
                    help="Directory for found maxmem files.")
parser.add_argument("--context_length", type=int, default=512,
                    help="Context length used for calibration.")
parser.add_argument("--dataset_length", type=int, default=1000,
                    help="Number of samples used for calibration.")
parser.add_argument("--lr", type=float, default=1e-2,
                    help="Learning rate for optimizer.")
parser.add_argument("--alpha", type=float, default=1e+0,
                    help="Alpha value for DP-LLM.")
parser.add_argument("--epoch", type=int, default=5,
                    help="Training epochs.")
parser.add_argument("--lr_decay", type=float, default=1.0,
                    help="Learning rate decaying for each epoch. Not used.")
parser.add_argument("--alpha_decay", type=float, default=1.0,
                    help="Alpha decay for each epoch. Not used.")
parser.add_argument("--init_targ", action="store_true", default=True,
                    help="Set initializing value as target precision. If not given, random initialization is used.")
# ===============================

args = parser.parse_args()
# ========================

import torch
from any_precision import DPLLM_Finetune
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import argparse
import random
import os
import math
import torch.optim as optim
from dp_llm_utils.model_def import getModelInfoFromConfig, extractModelTypeFromPath
from dp_llm_utils.dataset_tokenize import getDataLoader

model_path = args.ap_model_path

# Get model type
model_type = extractModelTypeFromPath(args.ap_model_path)

save_path = os.path.join(args.save_path, model_type)

# Get model properties
config = AutoConfig.from_pretrained(args.ap_model_path)
model_dict = getModelInfoFromConfig(config)

layer_count = model_dict["layer_count"]
module_list = model_dict["module_list"]

size_d = model_dict["size_d"]
size_arr = [size_d[name] for _, name in module_list] * layer_count

# Divide by GCD to maintain small numbers
size_gcd = math.gcd(*size_arr)
name_to_mem = {linear: (size_arr[linear_i] // size_gcd) for linear_i, (_, linear) in enumerate(module_list)}

# name to module conversion
def name2module(model, layer, parent, name):
    return model.model.model.layers[layer]._modules[parent]._modules[name]


os.makedirs(save_path, exist_ok=True)

# Define bits to utilize
bits_arr = [bit for bit in range(args.min_prec, args.max_prec+1)]
print(f"Using {bits_arr} bits.")

# Learning parameters
lr = args.lr
alpha = args.alpha
epoch = args.epoch

# Precision settings
targ_bits = args.targ_bits
min_prec = args.min_prec
max_prec = args.max_prec
maxmem = args.maxmem

if targ_bits > maxmem:
    raise RuntimeError(f"targ_bits({targ_bits}) > maxmem({maxmem})")

assert max_prec > min_prec
assert targ_bits != float(max_prec) and targ_bits != float(min_prec)


print(f"lr={lr}, alpha={alpha}, epoch={epoch}, prec={min_prec}-{max_prec}, targbits={targ_bits}")

torch.random.manual_seed(0)
random.seed(0)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Pre-sigmoid initialization values for each linear
z_init_dict = {}

# Max precision for each linaer
max_mem_list = [max_prec] * (layer_count*len(module_list))
if maxmem < max_prec:
    maxmem_path = os.path.join(args.maxmem_dir, model_type, f"maxmem_{maxmem}.pt")
    max_mem_list = torch.load(maxmem_path)
    print(f"Using {maxmem_path} as max memory settings")

max_mem_dict = {}

for l in range(layer_count):
    for lin_i, (parent, n) in enumerate(module_list):
        module_i = l*len(module_list)+lin_i

        # Use random value as default
        z_init_dict[(l,n)] = random.random()

        # Initialize to target precision if needed
        if args.init_targ:
            if targ_bits < max_mem_list[module_i]:
                z_init_dict[(l,n)] = (targ_bits-min_prec) / (max_mem_list[module_i]-min_prec)
        
        max_mem_dict[(l,n)] = max_mem_list[module_i]


# Make DP-LLM model for fine-tuning
model = DPLLM_Finetune.from_quantized(model_path, precisions=[p for p in range(min_prec, max_prec+1)], 
                                                     z_init_dict=z_init_dict, max_mem_dict=max_mem_dict)
print(f"Moving AP model to CUDA...", end="", flush=True)
model = model.cuda()
print("Done.")

# Turn off every parameter present
for p in model.parameters():
    p.requires_grad = False

# Create pre-sigmoid variables to learn for each linear
model.create_z()

# sum of parameters of given model
param_sum = 0
for l in range(layer_count):
    for _, n in module_list:
        param_sum += name_to_mem[n]

# Final average precisions will be saved
p_d = {}

# Load dataset
dataloader = getDataLoader(args.dataset, tokenizer, args.context_length, args.dataset_length)

sigmoid = torch.nn.Sigmoid()

# Optimzer for finetuning
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

# Training epochs
for ep in range(epoch):
    if ep > 0:
        # Decay for lr and alpha
        # Currently not used
        # optimizer.param_groups[0]['lr'] /= args.lr_decay
        # alpha /= args.alpha_decay
        pass

    # Training loop
    with tqdm(dataloader, "Run") as tobj:
        for batch in tobj:
            optimizer.zero_grad()
            b = batch['input_ids'].to("cuda:0")
            loss = model(input_ids=b, labels=b).loss 
            loss = loss.mean()

            sum = torch.zeros_like(loss)
            for l in range(layer_count):
                for parent, n in module_list:
                    maxmem_now = max_mem_dict[(l,n)]

                    # If more then one bit is usable
                    if maxmem_now > min_prec:
                        prange_len = maxmem_now - min_prec

                        p = sigmoid(name2module(model, l, parent, n).z).to(loss.device)*prange_len + min_prec
                        bl = math.floor(p.item())
                        bh = math.ceil(p.item())
                        r = 1 - (p - bl)

                        sum += (r) * name_to_mem[n] * bl
                        sum += (1-r) * name_to_mem[n] * bh

                    # Only min precision is used
                    else:
                        sum += name_to_mem[n] * min_prec


            # Add alpha term to loss
            loss += alpha *((sum/param_sum-targ_bits)**2)
            loss.backward()
            optimizer.step()
            tobj.set_description(f"avg prec={(sum/param_sum).item():.4f}")

# Retrieve final average precisions
for l in range(layer_count):
    for parent, n in module_list:
        maxmem_now = max_mem_dict[(l,n)]
        prange_len = maxmem_now - min_prec

        if maxmem_now > min_prec:
            p = sigmoid(name2module(model, l, parent, n).z).item()*prange_len + min_prec
        else:
            p = -1
        p_d[(l,n)] = p

# Save finetuned results
save_str = os.path.join(save_path, (f"finetuned_max{maxmem}_{min_prec}b-{max_prec}b"
            f"_th_pb_train_{args.lr}_{args.alpha}_{args.epoch}ep_targ{args.targ_bits}b"
            f"_{'init_' if args.init_targ else ''}0-{args.dataset_length}_adam.pt"))
print(f"Saving to {save_str}")
torch.save((p_d, max_mem_dict), save_str)
