#EpyNN/nnlibs/commons/decorators.py
from termcolor import colored, cprint


def log_function(func):
    def decorated_function(*args, **kwargs):
        cprint(colored("sF: "+func.__name__,'green'))
        r = func(*args, **kwargs)
        cprint(colored("eF: "+func.__name__,'green',attrs=['bold']))
        return r
    return decorated_function


def log_class(func):
    def decorated_instance(*args, **kwargs):
        cprint(colored("sC: "+func.__name__,'blue'))
        r = func(*args, **kwargs)
        cprint(colored("eC: "+func.__name__,'blue',attrs=['bold']))
        return r
    return decorated_instance


def log_method(func):
    def decorated_method(*args, **kwargs):
        cprint(colored("sM: "+func.__name__,'cyan'))
        r = func(*args, **kwargs)
        cprint(colored("eM: "+func.__name__,'cyan',attrs=['bold']))
        return r
    return decorated_method
