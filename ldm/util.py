import importlib
from inspect import isfunction


def default(x, default_value):
    if exist(x):
        return x
    return default_value() if isfunction(default_value) else default_value


def exist(x):
    return x is not None


def get_obj_from_str(string, reload=False):
    # e.g., cldm.cldm.ControlLDM -> [cldm.cldm, ControlLDM]
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    return get_obj_from_str(config['target'])(**config.get("params", dict()))



