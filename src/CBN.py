
import numpy as np

from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

class CausalBayesianNetwork(BayesianNetwork):
    
    def __init__(self, ebunch=None, latents=set()):
        super().__init__(ebunch=ebunch, latents=latents)

    def copy(self):
        model_copy = CausalBayesianNetwork()
        model_copy.add_nodes_from(self.nodes())
        model_copy.add_edges_from(self.edges())
        if self.cpds:
            model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        model_copy.latents = self.latents
        return model_copy

    def compute_interventional_marginal(self,varlist,intvarlist,intvallist=None):
        Mdo = self.do(intvarlist)
        inf = VariableElimination(Mdo)

        if intvallist is None:
            return inf.query(varlist,show_progress=False)
        else:
            evidence = {intvarlist[i]: intvallist[i] for i in range(len(intvarlist))}
            return inf.query(varlist, evidence=evidence, show_progress=False)

    def compute_interventional_joint(self, intvarlist, intvallist=None):
        return self.compute_interventional_marginal(list(self.nodes), intvarlist, intvallist)

    def compute_interventional_conditional(self, varlist, condvarlist, intvarlist,intvallist=None):
        joint_TS = self.compute_interventional_marginal(varlist + condvarlist, intvarlist=intvarlist, intvallist=intvallist)
        marginal_S = self.compute_interventional_marginal(condvarlist, intvarlist=intvarlist, intvallist=intvallist)
        return joint_TS/marginal_S

    def compute_observational_marginal(self,varlist):
        return self.compute_interventional_marginal(varlist, [], None)

    def compute_observational_joint(self):
        return self.compute_interventional_marginal(list(self.nodes),[],None)

    def reorder_vars_in_factor(self,f,order):
        old_indexes = range(len(order))
        new_indexes = [(order).index(i) for i in f.variables]
        return np.moveaxis(f.values, old_indexes, new_indexes)

    def square_tensor_in_factor(self,f_val,rows,cols):
        cardinalities = self.get_cardinality()
        target_cards = [cardinalities[t] for t in rows]
        target_card = np.prod(target_cards)
        source_cards = [cardinalities[s] for s in cols]
        source_card = np.prod(source_cards)
        return f_val.reshape(target_card, source_card)

    def compute_mechanisms(self,sources,targets):
        #Compute P(targets|do(sources)) as P(targets|sources) in M_do(sources)
        cond_TS = self.compute_interventional_conditional(targets, condvarlist=sources, intvarlist=sources)

        #Order the values according to targets and sources
        cond_TS_val = self.reorder_vars_in_factor(cond_TS,order=targets+sources)

        #Reshape the matrix as [targets x sources]
        return self.square_tensor_in_factor(cond_TS_val,rows=targets,cols=sources)


