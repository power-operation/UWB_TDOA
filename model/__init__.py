import importlib

def load_model(model_name, config):
    """
    Dynamically load a model based on the model name and configuration from model.yaml.
    
    Args:
        model_name (str): Name of the model (e.g., 'least_squares').
        config (dict): Configuration dictionary from model.yaml.
    
    Returns:
        callable: The estimation function for the specified model.
    """
    if model_name not in config['models']:
        raise ValueError(f"Model '{model_name}' not found in model.yaml")
    
    model_info = config['models'][model_name]
    module_name = model_info['module']
    function_name = model_info['function']
    
    try:
        module = importlib.import_module(module_name)
        model_func = getattr(module, function_name)
        return model_func
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Failed to load model '{model_name}': {str(e)}")