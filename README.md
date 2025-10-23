# DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment [[Paper]](https://arxiv.org/abs/2508.06041)

## Overview
Dynamic-Precision LLM (**DP-LLM**) is a model runtime adaptation mechanism that supports dynamic layer-wise precision assignment.

_**Warning: Currently, this repository only contains the performance evaluation codes. The codes for latency measurement will soon be updated.**_

## Prerequisites
Prerequisites are identical to [Any-Precision LLM](https://github.com/SNU-ARC/any-precision-llm).
- Python 3.11
- CUDA Toolkit 12 or higher
- gcc-9 or higher

## Setup
The setup process is identical to [Any-Precision LLM](https://github.com/SNU-ARC/any-precision-llm).

1. Clone this repository.

```bash
git clone https://github.com/SNU-ARC/dp-llm
cd dp-llm
```

2. Install the required Python packages.

```bash
pip install -r requirements.txt
```

3. Install the Any-Precision CUDA kernels.

```bash
cd any_precision/modules/kernels
pip install .
```

## Quick usage of DP-LLM
```python
from any_precision import DPLLMForCausalLM

model = DPLLMForCausalLM.from_quantized(model_path,
              precisions=precisions,
              max_mem_dict=max_mem_dict,
              linear_reg_d=linear_reg_d,
              jl_d=jl_d,
              T_d=T_d,
              prefill_by_decode=True, # True for pp evaluation, 
                                      # False for downstream tasks
        )
```

- `precisions`: An array of available precisions.
- `max_mem_dict`: A dictionary containing the max precisions assigned for each linear layer.
- `linear_reg_d`: A dictionary containing parameters for linear regression based relative error estimator.
- `jl_d`: A dictionary containing parameters for random projection based relative error estimator (Noted as `G` in the paper).
- `T_d`: A dictionary containing threshold values for each linear layer (Noted as `T` in the paper).
- `prefill_by_decode`: For perplexity evaluations, set to `True` for efficient evaluations. When set to `True`, the model will activate dynamic precision assignment during the prefill phase. When set to `False`, max precision will be used for the prefill phase, and dynamic precision assignment will only be active during the decoding phase.  

## Pre- Fine-tuned results
Some fine-tuned results are provided for quick evaluation. The pre-finetuned results can be found at https://github.com/SNU-ARC/DP-LLM_pre_finetuned.

Load each `.pt` files within the directory using `torch.load`, then provide them as arguments(`max_mem_dict`, `linear_reg_d`, `jl_d` and `T_d`). 

The following configurations are available:

- Meta-Llama-3-8B, 3,4,5,6 bits, 5.0-bit memory budget: 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75 target precisions

## Run DP-LLM
### 0. Quantize a model using Any-Precision LLM
Please refer to https://github.com/SNU-ARC/any-precision-llm#quantization for more precise instructions.
```bash
python quantize.py <model> [options]
```

### 1. Record linear sizes
Run `0_set_configs.py` to record linear sizes and write them to the config files.
```bash
python 0_set_configs.py <ap_model_path>

# e.g. 
# python 0_set_configs.py cache/packed/anyprec-(Meta-Llama-3-8B-hf)-w8_orig3-gc1-c4_s100_blk512
```

### 2. Find layer-wise maximum precision
Run `1_find_maxmem.py` to find layer-wise maximum precision.
```bash
python 1_find_maxmem.py <model> <ap_model_path> --hessian_path path/to/hessian --memory_budget <memory budgets>

# e.g. 
# python 1_find_maxmem.py \
# meta-llama/Meta-Llama-3-8B-hf \
# cache/packed/anyprec-(Meta-Llama-3-8B-hf)-w8_orig3-gc1-c4_s100_blk512 \
# --hessian_path \
# cache/packed/gradients/(Meta-Llama-3-8B-hf)-c4_s100_blk512.pt \
# --memory_budget 4.0 5.0
```

### 3. Fine-tune to find layer-wise average precision
Run `2_finetune.py` to find layer-wise average precision.
```bash
python 2_finetune.py <ap_model_path> --maxmem <memory budget> --targ_bits <target precision>

# e.g. 
# python 2_finetune.py \
# cache/packed/anyprec-(Meta-Llama-3-8B-hf)-w8_orig3-gc1-c4_s100_blk512 \
# --maxmem 5.0 --targ_bits 3.5
```

### 4. Save estimator parameters
Run `3_save_estimator.py` to create error estimators.
```bash
python 3_save_estimator.py <model> <ap_model_path> --arr_path <finetuned result>

# e.g. 
# python 3_save_estimator.py \
# meta-llama/Meta-Llama-3-8B-hf \
# cache/packed/anyprec-(Meta-Llama-3-8B-hf)-w8_orig3-gc1-c4_s100_blk512 \
# --arr_path \
# finetuned_results/anyprec-()-w8_orig3-gc1-c4_s100_blk512/finetuned_max5.0_3b-6b_th_pb_train_0.01_1.0_5ep_targ3.5b_init_0-1000_adam.pt
```

### 5. Save threshold values
Run `4_save_th.py` to save threshold values.

```bash
python 4_save_th.py <ap_model_path> --arr_path <finetuned_result>

# e.g.
# python 4_save_th.py \
# cache/packed/anyprec-(Meta-Llama-3-8B-hf)-w8_orig3-gc1-c4_s100_blk512 \
# --arr_path \
# finetuned_results/anyprec-()-w8_orig3-gc1-c4_s100_blk512/finetuned_max5.0_3b-6b_th_pb_train_0.01_1.0_5ep_targ3.5b_init_0-1000_adam.pt
```

### Testing perplexity
Run `test_pp.py` to test DP-LLM's perplexity results.

```bash
python test_pp.py <ap_model_path> --estimator_results <estimator directory>

# e.g.
# python test_pp.py \
# cache/packed/anyprec-(Meta-Llama-3-8B-hf)-w8_orig3-gc1-c4_s100_blk512 \
# --estimator_results \
# estimator_private_values/anyprec-()-w8_orig3-gc1-c4_s100_blk512/finetuned_max5.0_3b-6b_th_pb_train_0.01_1.0_5ep_targ3.5b_init_0-1000_adam
```

## Citation

Please cite our paper if you find our work useful:

```bibtex
@inproceedings{kwon2025dp,
  title={DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment},
  author={Sangwoo Kwon and Seong Hoon Seo and Jae W. Lee and Yeonhong Park},
  year={2025},
  booktitle={Proceedings of the 39th Conference on Neural Information Processing Systems}
}
```