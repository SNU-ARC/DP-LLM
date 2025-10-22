import argparse

# == Arguments defined ==
parser = argparse.ArgumentParser(description="Assign max precision to each lienar.")
parser.add_argument("model_path", type=str,
                    help="Path to original model.")
parser.add_argument("ap_model_path", type=str,
                    help="Path to AnyPrecision Model.")
parser.add_argument("--hessian_path", type=str, required=True,
                    help="Path to saved hessian file.")
parser.add_argument("--memory_budget", type=float, required=True, nargs="+", 
                    help="Memory budget expressed in bits.")


parser.add_argument("--min_prec", type=int, default=3,
                    help="minimum precision available.")
parser.add_argument("--max_prec", type=int, default=6,
                    help="Maximum precision available.")

parser.add_argument("--save_dir", type=str, default="./maxmem_results",
                    help="Directory to save results.")

args = parser.parse_args()
# =======================

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig
from any_precision import AnyPrecisionForCausalLM
from any_precision_ext import dequant_kbit
import os
from tqdm import tqdm
import cvxpy
import time
from dp_llm_utils.model_def import getModelInfoFromConfig, extractModelTypeFromPath

# Get model type from AnyPrecision model path
model_type = extractModelTypeFromPath(args.ap_model_path)

# Get model properties
config = AutoConfig.from_pretrained(args.ap_model_path)
model_dict = getModelInfoFromConfig(config)

layer_count = model_dict["layer_count"]
module_list = model_dict["module_list"]

size_d = model_dict["size_d"]
size_arr = [size_d[name] for _, name in module_list] * layer_count
size_np = np.array(size_arr)

# Define bits to utilize
min_prec = args.min_prec
max_prec = args.max_prec
bits_arr = [bit for bit in range(min_prec, max_prec+1)]
print(f"Using {bits_arr} bits.")

# Number of total linears
mN = len(size_arr)

# Load H
print("Loading Hessians...", end="", flush=True)
grad_arr = torch.load(args.hessian_path)
print("Done.")

# s_arr_d[bit] = [list of g * deltaW^2 for each layer]
s_arr_d = {bit:[] for bit in bits_arr}

# Load models
model_orig = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cpu").eval()
model_ap = AnyPrecisionForCausalLM.from_quantized(args.ap_model_path, precisions=None)
model_ap = model_ap.eval().cuda()

# Calculate loss difference using H
for l in tqdm(range(layer_count), "Calculating loss difference", unit="layer"):
    for parent_module, n in module_list:

        # Fetch hessians for layer
        g_n = parent_module+"."+n
        g = grad_arr[l][g_n]

        # Load fp16 weight
        w16 = model_orig.model.layers[l]._modules[parent_module]._modules[n].weight.data.clone().float().cpu()

        for bit in bits_arr:
            # Load fake quantized weights
            linear = model_ap.model.model.layers[l]._modules[parent_module]._modules[n]
            wq = dequant_kbit(linear.qweight, linear._buffers[f'lut{bit}'], bit).float().cpu()

            # Calculate loss difference
            s_arr_d[bit].append((g * ((w16-wq)**2)).sum().item())


# Max bits for each layer
max_layer_bits = [max_prec] * len(size_arr)

# Find layerwise precision allocation
for memory_budget_i, memory_budget in enumerate(args.memory_budget):
    print(f"({memory_budget_i+1}/{len(args.memory_budget)}) target: {memory_budget}")

    # function for finding assigned bits for each layer
    def get_layer_bits(z: cvxpy.Variable) -> list:
        layer_bits = []

        # Convert cvxpy variable to layerwie bit
        for i in range(mN):
            bits = -1

            for k, bit in enumerate(reversed(bits_arr)):
                if z[i+mN*k].value == 1.0:
                    if bits == -1:
                        bits = bit
                    else:
                        raise RuntimeError("Multiple precision assigned")
            
            if bits == -1:
                raise RuntimeError("No precision assigned")
            layer_bits.append(bits)
        return layer_bits
    
    # function for calculating overall precision
    def count_bits(layer_bits: list) -> float:
        # Calculate overall precision
        bsum = 0
        for i, b in enumerate(layer_bits):
            bsum += b * size_arr[i]
        return bsum/sum(size_arr)


    # Memory budget
    B = memory_budget * sum(size_arr)

    # Numpy arrays for loss differences
    s_np_d = {bit: np.array(s_arr_d[bit]) for bit in bits_arr}

    close_enough = False
    iter_n = 0
    lower_bound = 0.0

    # Loop until used bits are close enough to target precision
    while not close_enough:

        # optimizing binary variable
        z = cvxpy.Variable((mN * len(bits_arr)), boolean=True) #[c for 6b, c for 5b, ...]

        # Lower bound of memory usage
        L = lower_bound * sum(size_arr)

        # Objective
        obj = None
        for k, bit in enumerate(reversed(bits_arr)):
            if obj is None:
                obj = cvxpy.sum(cvxpy.multiply(s_np_d[bit],z[mN*k:mN*(k+1)]))
            else:
                obj += cvxpy.sum(cvxpy.multiply(s_np_d[bit],z[mN*k:mN*(k+1)]))

        # Memory usage constraint
        qsum = None
        for k, bit in enumerate(reversed(bits_arr)):
            if qsum is None:
                qsum = cvxpy.sum(cvxpy.multiply(size_arr,z[mN*k:mN*(k+1)]))*bit
            else:
                qsum += cvxpy.sum(cvxpy.multiply(size_arr,z[mN*k:mN*(k+1)]))*bit

        # Memory usage lower bound constraint
        qlowsum = None
        for k, bit in enumerate(reversed(bits_arr)):
            if qlowsum is None:
                qlowsum = cvxpy.sum(cvxpy.multiply(size_arr,z[mN*k:mN*(k+1)]))*bit
            else:
                qlowsum += cvxpy.sum(cvxpy.multiply(size_arr,z[mN*k:mN*(k+1)]))*bit
        
        constraints = []
        constraints.append(qsum <= B)
        constraints.append(qlowsum >= L)

        for i in range(mN):
            # Force only one selection for each layer
            zsum = None
            for k in range(len(bits_arr)):
                if zsum is None:
                    zsum = z[i+mN*k]
                else:
                    zsum += z[i+mN*k]
            constraints.append((zsum == 1))

            # Trim bits if maxmem limits max precision
            for k, bit in enumerate(reversed(bits_arr)):
                if max_layer_bits[i] < bit:
                    constraints.append((z[i+mN*k] == 0))
        
        # Solve ILP
        problem = cvxpy.Problem(cvxpy.Minimize(obj), constraints=constraints)
        problem.solve(solver=cvxpy.GLPK_MI)

        # Get results
        result_bits = count_bits(get_layer_bits(z))
        print(f"iter {iter_n}: lower={lower_bound:.2f} bits, result={result_bits:.2f} bits", end='\r')
        time.sleep(0)

        # Check if close enough
        if abs(result_bits - memory_budget) <= 0.01:
            close_enough = True
        else:
            lower_bound += 0.01
            iter_n += 1
    
    print("")

    # Get final bits
    layer_bits = get_layer_bits(z)
    
    # Get final overall precision
    result_bits = count_bits(layer_bits)
    print(f"Final result: {result_bits} bits")

    # Save results
    save_dir = os.path.join(args.save_dir, model_type)
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, f"maxmem_{memory_budget}.pt")
    torch.save(layer_bits, result_path)
    print(f"Saved results to {result_path}")