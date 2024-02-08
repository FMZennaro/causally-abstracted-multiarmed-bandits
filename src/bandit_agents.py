import numpy as np
import src.bandit_envs as be
import sklearn.linear_model as lm
from src import utils as ut

def initialize_Q_zeros(n):
    return np.zeros(n)      

def initialize_Q_ones(n):
    return np.ones(n)

def initialize_Q_c(n,c):
    return np.ones(n)*c

def initialize_Q_max(n,maxpos):
    values = np.zeros(n)
    values[maxpos] = 1
    return values

def initialize_Q_map(Q, map_Q):
    return np.array([np.mean(Q[map_Q[k]]) for k in map_Q.keys()])

class Agent():
    def __init__(self,env,k, 
                 log_actions=True,log_rewards=True,log_Qs=True):
        self.env = env
        self.n_actions = k
        self.action_counter = np.zeros(self.n_actions)
        self.reward_sum = np.zeros(self.n_actions)
        
        self.log_actions = log_actions; self.log_rewards = log_rewards; self.log_Qs = log_Qs
        self.history_actions = []; self.history_rewards = []; self.history_Qs = []
        
    
    def initialize_Q(self,Q):
        self.Q = Q
        
    def initialization_round(self):
        for i in range(self.n_actions):
            self.step(i)
    
    # for (t:T) \sum_t R_t
    def get_cumulative_reward(self):
        return np.cumsum(np.array(self.history_rewards))
    
    # \sum_T R_t
    def get_total_reward(self):
        return np.sum(np.array(self.history_rewards))
    
    # for (t:T) t*E[R(a*)]
    def get_max_cumulative_reward(self):
        totalsteps = len(self.history_actions)
        return (np.arange(totalsteps)+1) * self.env.optim_expected_reward
    
    # T*E[R(a*)]
    def get_max_total_reward(self):
        totalsteps = len(self.history_actions)
        return totalsteps * self.env.optim_expected_reward
    
    # for (t:T) t*E[R(a*)] - \sum_t R_t 
    def get_cumulative_regret(self):
        return self.get_max_cumulative_reward() - self.get_cumulative_reward()
    
    # T*E[R(a*)] - \sum_T R_t
    def get_total_regret(self):
        return self.get_max_total_reward() - self.get_total_reward()
    
    # for (t:T) \sum_t 1[a_t==a*]
    def get_optimality(self):
        return np.cumsum(np.array(self.history_actions)==self.env.optim_action)
    
    # \sum_T 1[a_t==a*]
    def get_total_optimality(self):
        return np.sum(np.array(self.history_actions)==self.env.optim_action)
    
    # \sum_a p(a)*Delta(a)
    def get_simple_regret(self):
        return -np.sum(self._action_distribution() * self.env.optim_gaps)
    
    # 1[argmax p(a) == a*]
    def has_learned_optimal_action(self):
        return np.argmax(self._action_distribution()) == self.env.optim_action
        
    def run(self,n_steps=1):
        for _ in range(n_steps):
            action = self._select_action()
            self.step(action)
    
    def imitate(self,D_actions):
        for action in D_actions:
            self.step(action)
            
    def replay(self,D_actions,D_rewards):
        for action,reward in zip(D_actions,D_rewards):
            self.step(action,fixedreward=reward)
        
    def step(self,action,fixedreward=None):
        if fixedreward is not None: reward = fixedreward
        else: _,reward,_,_ = self.env.step(action)
                
        if self.log_actions: self.history_actions.append(action)
        if self.log_rewards: self.history_rewards.append(reward)
        if self.log_Qs: self.history_Qs.append(self.Q.copy())
        self.reward_sum[action] = self.reward_sum[action] + reward
        self.action_counter[action] = self.action_counter[action]+1
        
        self._update_agent(action,reward)
        return reward            
    
class Agent_epsilon(Agent):
    
    def __init__(self,env,n_actions,eps,alpha=1):
        super().__init__(env,n_actions)
        self.eps = eps
        self.alpha = alpha
    
    def _action_distribution(self):
        p = np.ones(self.n_actions)*self.eps
        p[np.argmax(self.Q)] = 1-self.eps
        return p
    
    def _select_action(self):
        if(np.random.random() < self.eps):
            self.eps = self.eps*self.alpha
            return np.random.randint(0,self.n_actions)
        else:
            self.eps = self.eps*self.alpha
            return np.argmax(self.Q)
    
    def _update_agent(self,action,reward):
        self.Q[action] = self.Q[action] + (1./self.action_counter[action])*(reward-self.Q[action])
        

class Agent_UCB(Agent):  
    
    def __init__(self,env,n_actions,c):
        super().__init__(env,n_actions)
        self.c = c
        
    def _action_distribution(self):
        p = np.zeros(self.n_actions)
        p[self._select_action()] = 1
        return p
           
    def _select_action(self):        
        U = self.c * np.sqrt( np.log(np.sum(self.action_counter)) / self.action_counter  )
        a = self.Q + U
        return np.argmax(a)
        
    def _update_agent(self,action,reward):
        self.Q[action] = self.Q[action] + (1./self.action_counter[action])*(reward-self.Q[action])

        
class Agent_B_UCB(Agent_UCB):
    
    def __init__(self,env,n_actions,c,bounds):
        super().__init__(env,n_actions,c)
        self.bounds = bounds
        self.lbounds = np.array(bounds[0])
        self.hbounds = np.array(bounds[1])
        self._check_bounds()

        self.initialize_Q((np.array(self.lbounds) + np.array(self.hbounds))/2)
        
    def _check_bounds(self):
        lmax = np.max(self.lbounds)
        indexes = np.where(self.hbounds < lmax)[0]
        if len(indexes)>0:
            new_actions = [self.env.actions[i] for i in range(len(self.env.actions)) if i not in indexes]
            new_lbounds = [self.lbounds[i] for i in range(len(self.lbounds)) if i not in indexes]
            new_hbounds = [self.hbounds[i] for i in range(len(self.hbounds)) if i not in indexes]
            new_env = be.SCMEnv(self.env.scm, new_actions, self.env.target, self.env.ydomain)
            self.env = new_env
            self.n_actions = len(new_actions)
            self.action_counter = np.zeros(len(new_actions))
            self.reward_sum = np.zeros(len(new_actions))
            self.lbounds = new_lbounds
            self.hbounds = new_hbounds
            self.bounds = [new_lbounds,new_hbounds]
           
    def _select_action(self):        
        U = self.Q + self.c * np.sqrt( np.log(np.sum(self.action_counter)) / self.action_counter )
        U = [np.min([U[i],self.hbounds[i]]) for i in range(self.n_actions)]
        a = self.Q + U
        return np.argmax(a)


class Agent_UCB_alphaE(Agent_UCB):

    def __init__(self, env, n_actions, c,
                 basemodel,map_ydomains,map_actions,transferstep,delta,abserr):
        super().__init__(env, n_actions, c)

        Qbase = basemodel.history_Qs[transferstep]
        Qlinear, errors = self._eval_Qlinear(Qbase, map_ydomains, map_actions, len(env.actions))
        self.initialize_Q(Qlinear)

        self._transfer_counters(basemodel,map_actions)

        self.lbounds,self.hbounds = self._build_bounds(delta,abserr,errors)
        self.bounds = [self.lbounds,self.hbounds]
        self._check_bounds()


    def _eval_Qlinear(self, Qs, map_ydomains, map_actions, n_actions):
        Xtr = np.array(list(map_ydomains.keys()))
        Ytr = np.array(list(map_ydomains.values()))
        model = lm.LinearRegression().fit(Xtr.reshape(-1, 1), Ytr)

        Qs_linear = []
        for a in range(n_actions):
            fQs = model.predict(Qs[ut.inverse_fx(map_actions, a)].reshape(-1, 1))
            Qs_linear.append(np.max(fQs))
        errors = Ytr - model.predict(Xtr.reshape(-1, 1))

        return Qs_linear, errors

    def _transfer_counters(self, basemodel, map_actions,):
        counter0 = basemodel.action_counter
        for i in range(len(counter0)):
            self.action_counter[map_actions[i]] += counter0[i]

    def _build_bounds(self, delta, abserr, errors):
        hbounds = []
        lbounds = []
        for i in range(len(self.Q)):
            Q = self.Q[i]
            bound = np.sqrt(2 * np.log(2 / delta) / self.action_counter[i]) + abserr + np.abs(np.mean(errors))
            hbounds.append(Q + bound)
            lbounds.append(Q - bound)
        return lbounds,hbounds

    def _check_bounds(self):
        lmax = np.max(self.lbounds)
        indexes = np.where(self.hbounds < lmax)[0]
        if len(indexes) > 0:
            new_actions = [self.env.actions[i] for i in range(len(self.env.actions)) if i not in indexes]
            new_lbounds = [self.lbounds[i] for i in range(len(self.lbounds)) if i not in indexes]
            new_hbounds = [self.hbounds[i] for i in range(len(self.hbounds)) if i not in indexes]
            new_env = be.SCMEnv(self.env.scm, new_actions, self.env.target, self.env.ydomain)
            self.env = new_env
            self.n_actions = len(new_actions)
            self.action_counter = np.zeros(len(new_actions))
            self.reward_sum = np.zeros(len(new_actions))
            self.lbounds = new_lbounds
            self.hbounds = new_hbounds
            self.bounds = [new_lbounds, new_hbounds]

        
