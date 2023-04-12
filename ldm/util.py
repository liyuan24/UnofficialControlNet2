from inspect import isfunction


def default(x, default_value):
    if exist(x)
        return x
    return default_value() if isfunction(default_value) else default_value

def exist(x):
    return x is not None