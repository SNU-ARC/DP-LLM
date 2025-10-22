import argparse

# == Define arguments ==
parser = argparse.ArgumentParser(description="Save weight size info to config.")
parser.add_argument("ap_model_path", type=str,
                    help="Path to AnyPrecision Model.")
args = parser.parse_args()
# ========================

from transformers import AutoConfig, AutoModelForCausalLM

# Get configuration from AnyPrecision model
config = AutoConfig.from_pretrained(args.ap_model_path)
layer_count = config.num_hidden_layers

# Get analyzed linears
config_anyprec = config.anyprec['arch_config']
model_name = config_anyprec['model_name']
layers_name = config_anyprec['layers_name']
module_names = config_anyprec['module_names']
module_name_splitted = [tuple(name.split(".")) for name in module_names]

# Instantiate a reduced model
config.num_hidden_layers = 1
print(f"Instantiating reduced dummy model...", end="", flush=True)
model = AutoModelForCausalLM.from_config(config)
print("Done.")

# Get size of each linear
size_d = {}
layer = model._modules[model_name]._modules[layers_name][0]
for parent, name in module_name_splitted:
    linear = layer._modules[parent]._modules[name]
    w_size = linear.in_features * linear.out_features
    size_d[f"{parent}.{name}"] = int(w_size / config.hidden_size)

# Save results
config.anyprec['size_d'] = size_d
config.num_hidden_layers = layer_count
config.save_pretrained(args.ap_model_path)
print(f"Saved results to {args.ap_model_path}")