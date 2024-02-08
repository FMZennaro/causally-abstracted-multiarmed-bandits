import math
import numpy as np
import scipy.stats as stats

def _compute_maxent_input_distrib(M):
    return np.ones(M.shape[1])/M.shape[1]

def _compute_maxent_output_distrib(M):
    return np.ones(M.shape[0])/M.shape[0]

def _compute_maxent_effects_distrib(M):
    maxent_input_distrib = _compute_maxent_input_distrib(M)
    return np.dot(M,maxent_input_distrib)

def EI(M,maxent_effects=None,base=2):
    if maxent_effects is None: maxent_effects = _compute_maxent_effects_distrib(M)
    KLs = stats.entropy(M, np.expand_dims(maxent_effects,axis=1),base=base)
    ei = np.average(KLs)
    return KLs,ei

# Wrong approach
#def diffEI(M,N,alpha,base=2):
#    _,highEI = EI(N,None,base)
#    print(np.dot(M,np.linalg.pinv(alpha)))
#    _,lowEI = EI(np.dot(M,np.linalg.pinv(alpha)), _compute_maxent_effects_distrib(M), base)
#    return lowEI,highEI, highEI-lowEI

def diffEI(M,alpha,base=2):
    _,lowEI = EI(M,None,base)
    _,highEI = EI(np.dot(alpha, np.dot(M,np.linalg.pinv(alpha))),None,base)
    return lowEI,highEI, highEI-lowEI

def determinism(M,base=2):
    maxent_output_distrib = _compute_maxent_output_distrib(M)
    KLs = stats.entropy(M, np.expand_dims(maxent_output_distrib,axis=1),base=base)
    determinism = np.average(KLs/math.log(M.shape[1],base))
    return KLs,determinism

def degeneracy(M,base=2):
    maxent_effects = _compute_maxent_effects_distrib(M)
    maxent_output_distrib = _compute_maxent_output_distrib(M)
    KLs = stats.entropy(maxent_effects, maxent_output_distrib,base=base)
    degeneracy = KLs/math.log(M.shape[1],base)
    return KLs,degeneracy

def effectiveness(M,base=2):
    _,det = determinism(M,base=base)
    _,deg = degeneracy(M,base=base)
    return det-deg

def EI_eff_size(M,base=2):
    eff = effectiveness(M,base=base)
    size = math.log(M.shape[1],base)
    return eff*size