import gc
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
)
from accelerate.big_modeling import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
)

from .DPLLM_Linear import DPLLM_Linear
from any_precision.analyzer.analyzer import get_analyzer

from dp_llm_utils.model_def import SYNC_LINEARS, ASYNC_CONFIG
from functools import partial

def replace_module_by_name(layer, module_name, new_module):
    levels = module_name.split('.')
    module = layer
    for level in levels[:-1]:
        module = getattr(module, level) if not level.isdigit() else module[int(level)]
    setattr(module, levels[-1], new_module)


class DPLLMForCausalLM(nn.Module):
    def __init__(
            self,
            model_path,
            config,
            precisions=None,
            torch_dtype=torch.float16,
            fuse_layers=False,
            trust_remote_code=True,
            prefill_by_decode=False,
            max_mem_dict={},
            linear_reg_d={},
            jl_d={},
            T_d={},
    ):
        super().__init__()

        self.config = config

        self.supported_bits = list(range(self.config.anyprec['seed_precision'],
                                         self.config.anyprec['parent_precision'] + 1))
        if precisions is None:
            self.precisions = self.supported_bits
        else:
            assert len(precisions) == len(set(precisions)), "Precisions must be unique"
            assert all(bit in self.supported_bits for bit in precisions), \
                f"Supported bits {precisions} must be a subset of model supported bits {self.supported_bits}"
            self.precisions = precisions

        self.precision = max(self.precisions)

        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(
                    config=config,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                    attn_implementation="flash_attention_2",
                )

        self.analyzer = get_analyzer(self.model)

        self.ap_linears = []

        # Setup decoder layers to record residuals
        def layernorm_forward_override(module, orig_fn, hidden_states):
            module.recorded_hidden_states = hidden_states.clone()
            return orig_fn(hidden_states)
        
        for layer in self.get_model_layers():
            for parent_module in ASYNC_CONFIG.keys():
                layernorm_name = ASYNC_CONFIG[parent_module]["residual_layernorm"]
                layernorm = layer._modules[layernorm_name]
                layernorm.orig_fn = layernorm.forward
                layernorm.forward = partial(layernorm_forward_override, layernorm, layernorm.orig_fn)

        # Replace to AnyPrecisionLinear layers
        self._load_quantized_modules(prefill_by_decode=prefill_by_decode, dtype=torch_dtype, 
                                     max_mem_dict=max_mem_dict, linear_reg_d=linear_reg_d, jl_d=jl_d, T_d=T_d)

        self.tie_weights()

        device_map = {key: 'cpu' for key in self.model.state_dict().keys()}

        # loads the weights into modules and distributes
        # across available devices automatically
        load_checkpoint_and_dispatch(
            self.model,
            checkpoint=model_path,
            device_map=device_map,
            no_split_module_classes=[self.layer_type],
            dtype=torch_dtype,
        )

        # Dispath to devices
        if fuse_layers:
            self.fuse_layers()

        self.evaled = False

        self.prune_precisions()

    def forward(self, *args, **kwargs):
        prev_precision = self.precision
        precision = None
        if 'precision' in kwargs:
            precision = kwargs.pop('precision')
            self.set_precision(precision)

        results = self.model.forward(*args, **kwargs)
        if precision is not None:
            self.set_precision(prev_precision)
        return results

    def generate(self, *args, **kwargs):
        precision = None
        if 'precision' in kwargs:
            prev_precision = self.precision
            precision = kwargs.pop('precision')
            self.set_precision(precision)
        else:
            prev_precision = self.precision

        with torch.inference_mode():
            results = self.model.generate(*args, **kwargs)
        if precision is not None:
            self.set_precision(prev_precision)
        return results

    @staticmethod
    def _load_config(
            model_path,
            trust_remote_code=True,
    ):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        return config

    @classmethod
    def from_quantized(
            cls,
            quant_model_path,
            trust_remote_code=True,
            fuse_layers=False,
            precisions=None,
            prefill_by_decode=False,
            torch_dtype=torch.float16,
            max_mem_dict={},
            linear_reg_d={},
            jl_d={},
            T_d={},
    ):
        config = cls._load_config(quant_model_path, trust_remote_code)

        ap_model = cls(
            model_path=quant_model_path,
            precisions=precisions,
            config=config,
            fuse_layers=fuse_layers,
            trust_remote_code=trust_remote_code,
            prefill_by_decode=prefill_by_decode,
            torch_dtype=torch_dtype,
            max_mem_dict=max_mem_dict,
            linear_reg_d=linear_reg_d,
            jl_d=jl_d,
            T_d=T_d,
        )

        return ap_model

    def _load_quantized_modules(self, prefill_by_decode=False, dtype=torch.float16, 
                                max_mem_dict={}, linear_reg_d={}, jl_d={}, T_d={}):
        # Get blocks of model
        layers = self.analyzer.get_layers()

        for layer_i, layer in enumerate(tqdm(layers, desc="Loading AP Layers")):
            # Get every linear layer in a block
            named_linears = self.analyzer.get_modules(layer)

            # Replace nn.Linear with AnyPrecisionLinear
            for name, module in named_linears.items():
                parent_module, real_name = name.split(".")

                if (layer_i, real_name) not in max_mem_dict.keys():
                    max_mem_dict[(layer_i, real_name)] = max(self.precisions)

                if real_name in SYNC_LINEARS:
                    est_async = False
                    residual_layernorm = None
                    my_layernorm = None
                else: 
                    est_async = True
                    residual_prev = ASYNC_CONFIG[parent_module]["prev"]
                    residual_layer_i = layer_i - 1 if residual_prev else layer_i

                    if residual_layer_i < 0:
                        # For the very first self_attn, no asynchronous estimation
                        est_async = False
                        residual_layernorm = None
                        my_layernorm = None
                    else:
                        residual_layernorm_name = ASYNC_CONFIG[parent_module]["residual_layernorm"]
                        residual_layernorm = layers[residual_layer_i]._modules[residual_layernorm_name]

                        my_layernorm_name = ASYNC_CONFIG[parent_module]["my_layernorm"]
                        my_layernorm = layers[layer_i]._modules[my_layernorm_name]
                
                if (layer_i, real_name) in linear_reg_d.keys():
                    # Use linear estimator
                    est_linear = True
                    lin_params = linear_reg_d[(layer_i, real_name)]
                    est_params = (lin_params[0], lin_params[1])
                else:
                    est_linear = False
                    est_params = jl_d[(layer_i, real_name)]

                b_l, b_h, est_T = T_d[(layer_i, real_name)]

                wqlinear = DPLLM_Linear(
                    module.in_features, module.out_features,
                    self.supported_bits,
                    bias=module.bias is not None,
                    precisions=self.precisions,
                    dtype=dtype,
                    device=module.weight.device,
                    prefill_by_decode=prefill_by_decode,
                    maxmem = max_mem_dict[(layer_i, real_name)],
                    est_linear=est_linear,
                    est_async=est_async,
                    est_params=est_params,
                    est_T=est_T,
                    b_l=b_l,
                    b_h=b_h,
                    residual_layernorm=residual_layernorm,
                    my_layernorm=my_layernorm,
                )
                self.ap_linears.append(wqlinear)
                replace_module_by_name(layer, name, wqlinear)

            torch.cuda.empty_cache()
            gc.collect()
    

    def get_effective_bits(self):
        total_bits = 0
        total_comps = 0
        for linear in self.ap_linears:
            total_comps_temp = 0
            for bits in linear.comp_count.keys():
                comp = linear.comp_count[bits]
                total_bits += (comp * bits)*(linear.in_features*linear.out_features)
                total_comps_temp += comp
            if total_comps == 0:
                total_comps = total_comps_temp
        total_params = 0
        for linear in self.ap_linears:
            total_params += (linear.in_features*linear.out_features)

        return total_bits/(total_params*total_comps) if total_comps > 0 else 0
    
    def clear_comp_count(self):
        for linear in self.ap_linears:
            for bits in linear.comp_count.keys():
                linear.comp_count[bits] = 0

    def prune_precisions(self):
        for ap_linear in self.ap_linears:
            ap_linear.prune_precisions()

        torch.cuda.empty_cache()
        gc.collect()

    def set_precision(self, precision):
        for ap_linear in self.ap_linears:
            ap_linear.set_precision(precision)
        self.precision = precision

    def tie_weights(self):
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()

    def get_model_layers(self):
        module = self.model
        for attrib_name in self.config.anyprec['arch_config']['model_name'].split('.'):
            module = getattr(module, attrib_name)
        return getattr(module, self.config.anyprec['arch_config']['layers_name'])

    def fuse_layers(self):
        if 'fuse_target_layers' not in self.model_config:
            raise NotImplementedError("This model does not support layer fusion")
        # TODO implement layer fusion
        pass

    def setMotherLayer(self):
        layers = self.analyzer.get_layers()
        for i, layer in enumerate(layers):
            # Get every linear layer in a block
            named_linears = self.analyzer.get_modules(layer)

            # Replace nn.Linear with AnyPrecisionLinear
            for name, module in named_linears.items():
                real_name = name.split(".")[-1]
                if real_name == "q_proj" or real_name == "k_proj" or real_name == "v_proj" or real_name == "qkv_proj":
                    if i > 0:
                        module.mother_layer = layers[i-1]
                    module.mother_ln = layer.input_layernorm
                elif real_name == "gate_proj" or real_name == "up_proj" or real_name == "gate_up_proj":
                    module.mother_layer = layer
                    module.mother_ln = layer.post_attention_layernorm
                elif real_name == "o_proj" or real_name == "down_proj":
                    pass
                else:
                    raise RuntimeError(f"Set Mother Layer Failed: Unknown module {name}")

    def eval(self):
        if not self.evaled:
            super().eval()
            self.evaled = True
        return self

    @property
    def layer_type(self):
        for layer in self.get_model_layers():
            layer_class_name = layer.__class__.__name__
            if layer_class_name.endswith("DecoderLayer"):
                return layer_class_name
        return None

    @property
    def device(self):
        return self.model.device
