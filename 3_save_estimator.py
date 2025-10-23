import argparse
from dp_llm_utils.dataset_tokenize import getPossibleDatasets

# == Define arguments ==
parser = argparse.ArgumentParser(description="Create estimator parameters for DP-LLM.")
parser.add_argument("model_path", type=str,
                    help="Path to original model.")
parser.add_argument("ap_model_path", type=str,
                    help="Path to AnyPrecision Model.")

parser.add_argument("--finetuned_result", type=str, required=True,
                    help="Path for finetuned result.")
parser.add_argument("--finetune_k", default=True, action="store_true",
                    help="Use estimator finetuning")
parser.add_argument("--k", type=int, default=64,
                    help="Estimator dimension.")
parser.add_argument("--iterations", type=int, default=10000,
                    help="Estimator calibration iterations")
parser.add_argument("--finetune_lr", type=float, default=1e-3,
                    help="Learning rate for estimator finetuning.")
parser.add_argument("--private_dir", type=str, default="./estimator_private_values",
                    help="Directory for saving finetuned result specific values.")

parser.add_argument("--dataset", type=str, default="c4", choices=getPossibleDatasets(),
                    help="Directory for found maxmem files.")
parser.add_argument("--context_length", type=int, default=512,
                    help="Context length used for calibration.")
parser.add_argument("--dataset_length", type=int, default=40,
                    help="Number of samples used for calibration.")

parser.add_argument("--rsq_th", type=float, default=0.9,
                    help="R^2 threshold for using linear regression based estimator.")
parser.add_argument("--shared_dir", type=str, default="./estimator_shared_values",
                    help="Directory for saving shared values among finetuned results.")
parser.add_argument("--min_prec", type=int, default=3,
                    help="Minimum precision to utilize.")
parser.add_argument("--max_prec", type=int, default=6,
                    help="Maximum precision to utilize.")

args = parser.parse_args()
# ======================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from any_precision import AnyPrecisionForCausalLM
from any_precision_ext import dequant_kbit
import numpy as np
import os
import scipy
from tqdm import tqdm
import math
from torch.optim import SGD
from dp_llm_utils.model_def import getModelInfoFromConfig, extractModelTypeFromPath
from dp_llm_utils.record_x import getLayer0Inputs, getX, clearX
from dp_llm_utils.dataset_tokenize import getDataLoader

torch.random.manual_seed(0)

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
finetuned_result = args.finetuned_result
p_d, max_mem_dict = torch.load(finetuned_result)

# Set directories using model type
private_dir = os.path.join(args.private_dir, model_type)
shared_dir = os.path.join(args.shared_dir, model_type)

# Use finetuned file name as save directory
mid_dir = finetuned_result.split("/")[-1]
if len(mid_dir) == 0: mid_dir = finetuned_result.split("/")[-2]
mid_dir = mid_dir.split(".pt")[0]
os.makedirs(os.path.join(private_dir, mid_dir), exist_ok=True)

# Save a copy of maxmem dictionary to private directory
torch.save(max_mem_dict, os.path.join(private_dir, mid_dir, "max_mem_dict.pt"))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# Load original model
model_orig = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cpu", use_cache=False).eval()

# Load AnyPrecision model
model_ap = AnyPrecisionForCausalLM.from_quantized(args.ap_model_path, precisions=None).eval()

# Load dataset
dataloader = getDataLoader(args.dataset, tokenizer, args.context_length, args.dataset_length)

# Linears that will utilize linear regression
linear_reg_d = {}

rsq_th = args.rsq_th
k = args.k

linear_reg_d_path = os.path.join(private_dir, mid_dir, f"linear_reg_d.pt")
if not os.path.isfile(linear_reg_d_path):
    with tqdm(range(layer_count), desc="Correlation computation", unit="layer") as tbar:
        for layer in tbar:

            tbar.set_description(f"Recording inputs")

            # Get inputs
            if layer == 0:
                layerX, misc_inputs = getLayer0Inputs(model_orig, dataloader)
            xarr_d, layer_out_arr = getX(model_orig, layer, module_list, layerX, misc_inputs)

            for parent_module, name in module_list:
                tbar.set_description(f"Correlation computation for {name}")
                p = p_d[(layer, name)]

                b_l = math.floor(p)
                b_h = math.ceil(p)

                if b_l != b_h:
                    x = torch.cat(xarr_d[name]).float().cuda()

                    os.makedirs(f"{shared_dir}/{b_l}-{b_h}", exist_ok=True)

                    yerr_n_path = os.path.join(shared_dir, f"{b_l}-{b_h}", f"{layer}_{name}_yerr_n.pt")
                    if os.path.isfile(yerr_n_path):
                        # Use cached relative error distribution
                        yerr_n = torch.load(yerr_n_path)
                    else:
                        # Get deltaW
                        linear = model_ap.model.model.layers[layer]._modules[parent_module]._modules[name]
                        qweight = linear.qweight.cuda()
                        lut_l = linear._buffers[f'lut{b_l}'].cuda()
                        lut_h = linear._buffers[f'lut{b_h}'].cuda()
                        w_l = dequant_kbit(qweight, lut_l, b_l)
                        w_h = dequant_kbit(qweight, lut_h, b_h)
                        err = (w_h-w_l).float()

                        # Record relative error distribution
                        yerr = x @ err.T
                        yerr_n = yerr.norm(dim=-1).cpu().view(-1)
                        torch.save(yerr_n, yerr_n_path)

                        # Clean up
                        qweight.cpu()
                        lut_l.cpu()
                        lut_h.cpu()
                        del w_l, w_h, err

                    # Check correlation
                    xlist = x.norm(dim=-1).cpu().view(-1).tolist()
                    ylist = yerr_n.tolist()
                    slope, intercept, r_value, _, _ = scipy.stats.linregress(xlist, ylist)
                    if r_value ** 2 >= rsq_th:
                        linear_reg_d[(layer, name)] = (slope, intercept, r_value, b_l, b_h)
                
                del x

            layerX = layer_out_arr

            # Clean up recorded inputs
            del xarr_d
            clearX(model_orig, layer, module_list)

    print(f"Linear Estimator: {len(linear_reg_d)} out of {len(module_list) * layer_count}")
    torch.save(linear_reg_d, linear_reg_d_path)

else:
    print(f"Skipping correlation computation as {linear_reg_d_path} exists")
    linear_reg_d = torch.load(linear_reg_d_path)

jl_d = {}
with tqdm(range(layer_count), desc="Computing JL", unit="layer") as tbar:
    for layer in tbar:

        tbar.set_description(f"Recording inputs")
        if layer == 0:
            layerX, misc_inputs = getLayer0Inputs(model_orig, dataloader)
        xarr_d, layer_out_arr = getX(model_orig, layer, module_list, layerX, misc_inputs)

        for parent_module, name in module_list:
            tbar.set_description(f"JL for {name}")

            # If R^2 is high enough, skip JL
            if (layer,name) in linear_reg_d.keys():
                continue

            p = p_d[(layer, name)]
            b_l = math.floor(p)
            b_h = math.ceil(p)

            jl_path = os.path.join(shared_dir, f"{b_l}-{b_h}", f"{layer}_{name}_jl.pt")
            if b_l != b_h:
                finetuned_path = os.path.join(shared_dir, f"{b_l}-{b_h}", f"{layer}_{name}_jl_finetuned_{k}.pt")
                if args.finetune_k and os.path.isfile(finetuned_path):
                    # Use pre-finetuned JL result
                    G = torch.load(finetuned_path)
                elif (not args.finetune_k) and os.path.isfile(jl_path):
                    # Use cached vanilla JL result
                    G = torch.load(jl_path)
                else:
                    # Get deltaW
                    linear = model_ap.model.model.layers[layer]._modules[parent_module]._modules[name]
                    qweight = linear.qweight.cuda()
                    lut_l = linear._buffers[f'lut{b_l}'].cuda()
                    lut_h = linear._buffers[f'lut{b_h}'].cuda()
                    w_l = dequant_kbit(qweight, lut_l, b_l)
                    w_h = dequant_kbit(qweight, lut_h, b_h)
                    err = (w_h-w_l).float()

                    # Calculate JL
                    A = torch.normal(0.0, 1.0, (k, err.size(0))).to(err.device)
                    A = A / math.sqrt(k)
                    G = (A @ err).half()

                    # Clean up
                    qweight.cpu()
                    lut_l.cpu()
                    lut_h.cpu()

                    # Save vanilla JL
                    torch.save(G.cpu(), jl_path)

                    # Finetune if specified
                    if args.finetune_k:
                        tbar.set_description(f"Finetuning for {name}")

                        x = torch.cat(xarr_d[name]).float().cuda()
                        real_err = (x @ err.T).norm(dim=-1).detach()
                        
                        G = G.detach().requires_grad_(True)
                        x = x.half().detach()

                        opt = SGD([G], lr=args.finetune_lr)

                        for train_i in range(args.iterations):
                            opt.zero_grad()
                            jl_err = (x @ G.T).norm(dim=-1)
                            loss = ((jl_err-real_err)**2).mean()
                            loss.backward()
                            opt.step()

                        # Save finetuned results to be reused for other layers
                        torch.save(G.clone().cpu(), finetuned_path)

            else:
                # Dummy JL
                linear = model_orig.model.layers[layer]._modules[parent_module]._modules[name]
                G = torch.zeros(k, linear.weight.shape[1]).half()
            
            jl_d[(layer, name)] = G

        # Use this layer's output as next layer's input
        layerX = layer_out_arr

        # Clean up recorded inputs
        del xarr_d
        clearX(model_orig, layer, module_list)

jl_d_path = os.path.join(private_dir, mid_dir, "jl_d.pt")
torch.save(jl_d, jl_d_path)
print(f"JL saved to {jl_d_path}")