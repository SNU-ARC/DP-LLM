from transformers import AutoConfig

# List of linear layers that does not use asynchronous estimation
SYNC_LINEARS = ["o_proj", "down_proj"]

# Configuration for asynchronous estimation
ASYNC_CONFIG = {
    "self_attn": { # parent module
        "prev" : True, # Use previous layer's residual
        "residual_layernorm": "post_attention_layernorm", # Layernorm right after residual
        "my_layernorm": "input_layernorm", # Layernorm before parent module
    },
    "mlp": { # parent module
        "prev" : False, # Use same layer's residual
        "residual_layernorm": "input_layernorm", # Layernorm right after residual
        "my_layernorm": "post_attention_layernorm", # Layernorm before parent module
    }
}

def getModelInfoFromConfig(config:AutoConfig) -> dict:
    """
    Return the model property dictionary from model configuration.

    Args:
        config: Model configuration.

    Returns:
        a dictionary that contains the number of layers, a size dictionary, and the module list
    """

    layer_count = config.num_hidden_layers

    # Get analyzed linears
    config_anyprec = config.anyprec['arch_config']
    module_names = config_anyprec['module_names']
    module_name_splitted = [tuple(name.split(".")) for name in module_names]
    if "size_d" not in config.anyprec.keys():
        raise KeyError("size_d not found in config. Make sure to run 0_set_configs.py first.")
    size_d = config.anyprec['size_d']
    # Make redundant key-value pairs too
    size_d.update((name, size_d[f"{parent}.{name}"]) for parent, name in module_name_splitted)

    return {
        "layer_count" : layer_count,
        "size_d" : size_d,
        "module_list": module_name_splitted,
    }

def extractModelTypeFromPath(model_path:str) -> str:
    """
    Extract model type from path string.

    Args:
        model_path: Path to model.

    Returns:
        model type string extracted from model_path
    """

    # Get last part from path
    model_type = model_path.split("/")[-1]
    # Ignore last / if necessary
    if model_type == "" : model_type = model_path.split("/")[-2]

    return model_type