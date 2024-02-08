import numpy as np

from src import utils as ut
import src.bandit_agents as BA
import sklearn.linear_model as lm


def run_trial(env,n_steps,alg,params,
              map_actions,map_rewards,map_ydomains,base_bandit=None):
    
    env.reset()
    n_actions = len(env.actions)
    
    if alg=='epsilongreedy':
        Ag = BA.Agent_epsilon(env,n_actions,params['epsilon'],params['alpha'])
        Ag.initialize_Q(params['Qinit'](n_actions))
        Ag.run(n_steps)
    elif alg=='ucb':
        Ag = BA.Agent_UCB(env,n_actions,params['c'])
        Ag.initialize_Q(params['Qinit'](n_actions))
        Ag.run(n_steps)
    elif alg=='transfer-optimum':
        Ag = BA.Agent_UCB(env,n_actions,params['c'])
        maxQ = np.zeros(n_actions)
        maxQ[map_actions[np.argmax(base_bandit.Q)]] = 1
        Ag.initialize_Q(maxQ)
        Ag._update_agent = lambda *args: None 
        Ag.run(n_steps)
    elif alg=='imitation':
        Ag = BA.Agent_UCB(env,n_actions,params['c'])
        Ag.initialize_Q(params['Qinit'](n_actions))
        D_actions = [map_actions[a] for a in base_bandit.history_actions]
        Ag.imitate(D_actions)
    elif alg=='replay':
        Ag = BA.Agent_UCB(env,len(env.actions),params['c'])
        Ag.initialize_Q(params['Qinit'](n_actions))
        D_actions = [map_actions[a] for a in base_bandit.history_actions]
        D_rewards = [map_ydomains[r] for r in base_bandit.history_rewards]
        Ag.replay(D_actions,D_rewards)
    elif alg=='ucb_bounds':
        Ag = BA.Agent_B_UCB(env,len(env.actions),params['c'],params['bounds'])
        Ag.run(n_steps)  
    elif alg=='transfer-expval':
        Ag = BA.Agent_UCB_alphaE(env, n_actions, params['c'],
                 base_bandit,map_ydomains,map_actions,params['Qtransferstep'],params['delta'],params['abserr'])
        Ag.run(n_steps)
            
    return Ag
    
    