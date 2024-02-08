
import numpy as np
from scipy.spatial import distance

from src.utils import jsd
import src.evaluationsets as es
import src.MechMappings as mm



class SCMMappingEvaluator():
    def __init__(self,A):
        self.A = A

class AbstractionEvaluator(SCMMappingEvaluator):
    def __init__(self,A):
        super().__init__(A)
        
class CategoricalAbstractionEvaluator(AbstractionEvaluator):
    def __init__(self,A, metric=None, pinv=None, aggerror=None, aggoverall=None):
        if metric is None:
            self.metric = jsd
        else:
            self.metric = metric
        
        if pinv is None:
            self.pinv = np.linalg.pinv
        else:
            self.pinv = pinv
        
        if aggerror is None:
            self.aggerror = np.max
        else:
            self.aggerror = aggerror
        
        if aggoverall is None:
            self.aggoverall = np.max
        else:
            self.aggoverall = aggoverall
               
        super().__init__(A)
    
    def compute_diagram(self, corners, verbose=False):
        # Get the corners of the diagram
        M0_sources, M0_targets, M1_sources, M1_targets = corners
        if verbose: print('M0: {0} -> {1}'.format(M0_sources, M0_targets))
        if verbose: print('\nM1: {0} -> {1}'.format(M1_sources,M1_targets))

        # Compute the high-level mechanisms
        if verbose: print('Args: {}'.format(self.A.M1,M1_sources,M1_targets))
        M1_cond_TS_val = self.A.M1.compute_mechanisms(M1_sources,M1_targets)
        if verbose: print('M1 mechanism shape: {}'.format(M1_cond_TS_val.shape))
        if verbose: print('M1 mechanism: {}'.format(M1_cond_TS_val))

        # Compute the low-level mechanisms
        M0_cond_TS_val = self.A.M0.compute_mechanisms(M0_sources,M0_targets)
        if verbose: print('M0 mechanism shape: {}'.format(M0_cond_TS_val.shape))

        # Compute the alpha for sources
        alpha_S = self.A.compute_abstractions(M1_sources)
        if verbose: print('Alpha_s shape: {}'.format(alpha_S.shape))

        # Compute the alpha for targets
        alpha_T = self.A.compute_abstractions(M1_targets)
        if verbose: print('Alpha_t shape: {}'.format(alpha_T.shape))
            
        return M0_cond_TS_val, M1_cond_TS_val, alpha_S, alpha_T
    
    def compute_error(self,pair, verbose=False):
        corners = self.compute_corners(pair)
        M0_cond_TS_val, M1_cond_TS_val, alpha_S, alpha_T = self.compute_diagram(corners, verbose)
        lowerpath,upperpath = self.compute_paths(M0_cond_TS_val, M1_cond_TS_val, alpha_S, alpha_T)
        
        distances = []
        for do in range(lowerpath.shape[1]):
            distances.append( self.metric(lowerpath[:,do],upperpath[:,do]) )
        if verbose: print('\nDistances: {0}'.format(distances))

        error = self.aggerror(distances)
        if verbose: print('\nError: {0}'.format(error))
        return error
    
    def compute_overall_error(self,J, verbose=False):        
        errors = []
        for pair in J:
            error = self.compute_error(pair,verbose)
            errors.append(error)
            if verbose: print('\nError of pair {0}: {1}'.format(pair,error))
        
        overallerrror = self.aggoverall(errors)
        if verbose: print('\nOverall error: {0}'.format(overallerrror))
        return overallerrror
    
class ICEvaluator(CategoricalAbstractionEvaluator):
    def __init__(self,A, metric=None, pinv=None, aggerror=None, aggoverall=None):        
        self.compute_paths = lambda mu,nu,aA,aB: (nu@aA,aB@mu)
        self.compute_corners = A.get_diagcorners_from_M0
        super().__init__(A,metric,pinv,aggerror,aggoverall)
        
class IILEvaluator(CategoricalAbstractionEvaluator):
    def __init__(self,A, metric=None, pinv=None, aggerror=None, aggoverall=None):        
        self.compute_paths = lambda mu,nu,aA,aB: (mu,self.pinv(aB)@nu@aA)
        self.compute_corners = A.get_diagcorners_from_M0
        super().__init__(A,metric,pinv,aggerror,aggoverall)
        
class ISILEvaluator(CategoricalAbstractionEvaluator):
    def __init__(self,A, metric=None, pinv=None, aggerror=None, aggoverall=None):        
        self.compute_paths = lambda mu,nu,aA,aB: (nu,aB@mu@self.pinv(aA))
        self.compute_corners = A.get_diagcorners_from_M1
        super().__init__(A,metric,pinv,aggerror,aggoverall)
        
class ISCEvaluator(CategoricalAbstractionEvaluator):
    def __init__(self,A, metric=None, pinv=None, aggerror=None, aggoverall=None):        
        self.compute_paths = lambda mu,nu,aA,aB: (self.pinv(aB)@nu,mu@self.pinv(aA))
        self.compute_corners = A.get_diagcorners_from_M1
        super().__init__(A,metric,pinv,aggerror,aggoverall)
    
class AbstractionEffectiveInformationEvaluator(CategoricalAbstractionEvaluator):
    def __init__(self,A):
        super().__init__(A)
        
    def evaluate_EIs(self, J_algorithm=None, base=2, verbose=False, debug=False):
        if J_algorithm is None:
            J = es.get_sets_in_M1_with_directed_path_in_M1_or_M0(self.A.M0,self.A.M1,self.A.a,verbose=verbose)
        else:
            J = J_algorithm(self.A)
            
        EIs_low = []; EIs_high = []

        for pair in J:
            # Compute the diagram
            M0_cond_TS_val, M1_cond_TS_val, _, _ = self.compute_diagram(pair)

            # Compute the EI for the mechanisms
            _,EI_low = mm.EI(M0_cond_TS_val)
            _,EI_high = mm.EI(M1_cond_TS_val)
            EIs_low.append(EI_low)
            EIs_high.append(EI_high)

        # Output all the EIs computed
        if verbose: print('All EIs in low-level model: {0}'.format(EIs_low))
        if verbose: print('All EIs in high-level model: {0}'.format(EIs_high))
        if verbose: print('\n\nMIN EI IN LOW MODEL: {0}'.format(np.min(EIs_low)))
        if verbose: print('\n\nMIN EI IN HIGH MODEL: {0}'.format(np.min(EIs_high)))
            
        return EIs_low,EIs_high