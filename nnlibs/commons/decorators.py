#EpyNN/nnlibs/commons/decorators.py
from termcolor import colored, cprint


class log_method:
    def __init__(self, func):
        self.fn = func

    def __set_name__(self,owner,name):
        self.fn.class_name = owner.__name__

        setattr(owner, name, self.fn)

        if name == '__init__':
            cprint(colored("sC: "+owner.__name__+'.'+name,'cyan',attrs=['bold']))
        else:
            cprint(colored("sM: "+owner.__name__+'.'+name,'blue',attrs=['bold']))


def log_function(func):
    def decorated_function(*args, **kwargs):
        cprint(colored("sF: "+func.__name__,'green'))
        r = func(*args, **kwargs)
        cprint(colored("eF: "+func.__name__,'green',attrs=['bold']))
        return r
    return decorated_function
