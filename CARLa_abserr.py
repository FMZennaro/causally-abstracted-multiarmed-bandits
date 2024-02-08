
import numpy as np
from src.measuring import ICEvaluator,IILEvaluator
from src import utils as ut
from src.utils import jsd

def bandit_abserr(Ab,actions,outcome,err='IC',metric=jsd):
    if err=='IC':
        Ae = ICEvaluator(Ab,metric=metric)
    elif err=='IIL':
        Ae = IILEvaluator(Ab,metric=metric)
    
    identity = lambda x : x
    Ae.aggerror = identity

    bandit_errors = []
    for i in range(1,len(actions)):
        interv_var = list(actions[i].keys())
        target = [outcome]

        interv_val = ut.convert_binary_to_decimal(list(actions[i].values()))
        error = Ae.compute_error([interv_var,target])[interv_val]
        bandit_errors.append(error)
        
    return bandit_errors
