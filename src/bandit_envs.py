
import numpy as np
from pgmpy.inference import VariableElimination

class SCMEnv():
    def __init__(self,scm,actions,target,ydomain=lambda x : x):
        self.scm = scm
        self.actions = actions
        self.target = target
        self.ydomain = ydomain
        self._computerewards()
            
    def _computerewards(self):
        self.rwrd_domain = np.array(list(map(self.ydomain,np.arange(self.scm.get_cardinality(self.target)))))
        self.distr_rewards = [VariableElimination(self.scm.do(list(self.actions[i].keys()))).query([self.target],evidence=self.actions[i]).values 
                        for i in range(len(self.actions))]
        self.expct_rewards = [np.sum(self.rwrd_domain*p) for p in self.distr_rewards]
        self.Q = self.expct_rewards
        self.optim_expected_reward = np.max(self.expct_rewards)
        self.optim_gaps = self.expct_rewards - self.optim_expected_reward
        self.optim_action = np.argmax(self.expct_rewards)
    
    def reset(self):
        return None, 0, False, None
    
    def step(self,a):
        sample = self.scm.simulate(n_samples=1,do=self.actions[a],show_progress=False)
        y = self.ydomain(sample[self.target][0])
        return None, y, True, None
    
    def multistep(self,a,n_steps):
        sample = self.scm.simulate(n_samples=n_steps,do=self.actions[a],show_progress=False)
        y = np.array(list(map(self.ydomain,sample[self.target][0])))
        return None, y, True, None