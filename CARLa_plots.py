import numpy as np
import matplotlib.pyplot as plt

def get_rows(N,ncols):
    nrows = N//ncols
    if N%ncols > 0:
        nrows +=1
    return nrows

def get_current_row_col(i,nrows,ncols):
    r = i // ncols
    c = i - (r*ncols)
    return r,c

def print_distr(M,actions,outcome,outcome_vals=None):
    for a in actions:
        if(len(a)==0):
            pdf = M.compute_observational_marginal(outcome)
            intvars = ''; intvals = ''
        else:
            pdf = M.compute_interventional_marginal(outcome,list(a.keys()),list(a.values()))
            intvars = list(a.keys()); intvals = list(a.values())

        if outcome_vals is None:
            print('P({0}|do({1}={2})) \t= {3}'.format(outcome, intvars, intvals, pdf.values))
        else:
            print('P({0}|do({1}={2})) \t= {3} \t\t Q={4}'.format(outcome, intvars, intvals, pdf.values,pdf.values@outcome_vals))

def plot_stem_actions_taken_in_trial(agents,trial,labels,figsize):
    print('Trial {0}'.format(trial))
    for i,a in enumerate(agents):
        plt.figure(figsize=figsize)
        plt.stem(range(len(a[trial].history_actions)),a[trial].history_actions)
        plt.xlabel('Steps')
        plt.ylabel('Action')
        plt.title(labels[i])
        
def plot_stem_rewards_collected_in_trial(agents,trial,labels,figsize):
    print('Trial {0}'.format(trial))
    for i,a in enumerate(agents):
        plt.figure(figsize=figsize)
        plt.stem(range(len(a[trial].history_rewards)),a[trial].history_rewards)
        plt.xlabel('Steps')
        plt.ylabel('Rewards')
        plt.title(labels[i])
        
def plot_bar_rewards_choices_Qs(agents,eps,truerewards,trial,labels,figsize):
    print('Trial {0}'.format(trial))
    fig, ax = plt.subplots(len(agents),6,figsize=figsize)

    for i,a in enumerate(agents):

        ax[i,0].bar(range(len(truerewards[i])),truerewards[i])
        ax[i,0].set_xlabel('Action')
        ax[i,0].set_xticks(range(len(truerewards[i])))
        ax[i,0].set_ylabel('P(reward)')
        
        delta_A = np.max(truerewards[i]) - truerewards[i]
        ax[i,1].bar(range(len(agents[i][trial].Q)), delta_A, color='C1')
        ax[i,1].set_xlabel('Action')
        ax[i,1].set_xticks(range(len(agents[i][trial].Q)))
        ax[i,1].set_ylabel('Delta(a)')

        ax[i,2].bar(*np.unique(agents[i][trial].history_actions, return_counts=True),color='C9')
        ax[i,2].set_xlabel('Action')
        ax[i,2].set_xticks(range(len(truerewards[i])))
        ax[i,2].set_ylabel('Number of actions taken')
        
        ax[i,3].bar(range(len(agents[i][trial].Q)),agents[i][trial].Q,color='C7')
        ax[i,3].set_xlabel('Action')
        ax[i,3].set_xticks(range(len(agents[i][trial].Q)))
        ax[i,3].set_ylabel('Q')
        
        renorm_Q = (agents[i][trial].Q * (1-eps[i])) / np.sum(agents[i][trial].Q)
        uniform_V = np.ones(len(agents[i][trial].Q)) * eps[i] / len(agents[i][trial].Q)
        actual_P = renorm_Q + uniform_V      
        ax[i,4].bar(range(len(agents[i][trial].Q)), actual_P, color='C7')
        ax[i,4].set_xlabel('Action')
        ax[i,4].set_xticks(range(len(agents[i][trial].Q)))
        ax[i,4].set_ylabel('P(a)')
        
        ax[i,5].bar(range(len(agents[i][trial].Q)), delta_A*actual_P, color='C7')
        ax[i,5].set_xlabel('Action')
        ax[i,5].set_xticks(range(len(agents[i][trial].Q)))
        ax[i,5].set_ylabel('Delta(a) * P(a)')
        ax[i,5].set_title('sum = {0}'.format(np.sum(delta_A*actual_P)))

    fig.tight_layout()
    
def plot_cumreward_optimality_all_trials(cumrewards,optimalities,n_trials,n_steps,labels,figsize):
    fig, ax = plt.subplots(n_trials,2,figsize=figsize)

    for t in range(n_trials):
        for i in range(len(labels)):
            ax[t,0].plot(np.arange(n_steps),cumrewards[i,t],label=labels[i])
        #ax[t,0].plot(np.arange(n_steps),np.arange(n_steps),'k')
        ax[t,0].set_xlabel('Steps')
        ax[t,0].set_ylabel('Cumulative reward')
        ax[t,0].legend()

        for i in range(len(labels)):
            ax[t,1].plot(np.arange(n_steps),optimalities[i,t],label=labels[i])
        #ax[t,1].plot(np.arange(n_steps),np.arange(n_steps),'k')
        ax[t,1].set_xlabel('Steps')
        ax[t,1].set_ylabel('Optimal choices')
        ax[t,1].legend()

    plt.tight_layout()
       
def plot_cumreward_optimality_avg(cumrewards,optimalities,n_trials,n_steps,labels,figsize):
    fig, ax = plt.subplots(1,2,figsize=figsize)

    for i in range(len(labels)):
        ax[0].plot(np.arange(n_steps),np.mean(cumrewards[i],axis=0),label=labels[i])
    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel('Cumulative reward')
    ax[0].legend()

    for i in range(len(labels)):
        ax[1].plot(np.arange(n_steps),np.mean(optimalities[i],axis=0),label=labels[i])
    ax[1].plot(np.arange(n_steps),np.arange(n_steps),'k')
    ax[1].set_xlabel('Steps')
    ax[1].set_ylabel('Optimal choices')
    ax[1].legend()
    
def plot_cumreward_optimality_avg_by_agent(cumrewards,optimalities,truerewards,n_trials,n_steps,labels,figsize):
    fig, ax = plt.subplots(len(labels),2,figsize=figsize)

    for i in range(len(labels)):
        avg = np.mean(cumrewards[i],axis=0)
        std = np.std(cumrewards[i],axis=0)
        ax[i,0].plot(np.arange(n_steps),avg,label=labels[i],color="C{}".format(i))
        ax[i,0].fill_between(np.arange(n_steps),avg+std,avg-std, color='C{}'.format(i), alpha=0.3)
        ax[i,0].set_xlabel('Steps')
        ax[i,0].set_ylabel('Cumulative reward')
        ax[i,0].set_title(labels[i])
        ax[i,0].plot(np.arange(n_steps),np.max(truerewards[i])*np.arange(n_steps),'k')

    for i in range(len(labels)):
        avg = np.mean(optimalities[i],axis=0)
        std = np.std(optimalities[i],axis=0)
        ax[i,1].plot(np.arange(n_steps),avg,label=labels[i],color="C{}".format(i))
        ax[i,1].fill_between(np.arange(n_steps),avg+std,avg-std, color='C{}'.format(i), alpha=0.3)
        ax[i,1].set_xlabel('Steps')
        ax[i,1].set_ylabel('Optimal choices')
        ax[i,1].set_title(labels[i])
        ax[i,1].plot(np.arange(n_steps),np.arange(n_steps),'k')

    fig.tight_layout()

def plot_matshow_rewards_in_trial(agents,trial,figsize):
    print('Trial {0}'.format(trial))
    plt.matshow(np.array([agents[i][trial].history_rewards for i in range(len(agents))]), cmap=plt.cm.gray_r)
    plt.xlabel('Steps')
    plt.ylabel('Agents')
    plt.colorbar()
    
def plot_matshow_actions_in_trial(agents,trial,figsize):
    print('Trial {0}'.format(trial))
    plt.matshow(np.array([agents[i][trial].history_actions for i in range(len(agents))]), cmap=plt.cm.gray_r)
    plt.xlabel('Steps')
    plt.ylabel('Agents')
    plt.colorbar()
    
def plot_matshow_optimality_in_trial(agents,banditoptimalactions,trial,figsize):
    print('Trial {0}'.format(trial))
    plt.matshow(np.array([np.array(agents[i][trial].history_actions)==banditoptimalactions[i] for i in range(len(agents))]), cmap=plt.cm.gray_r)
    plt.xlabel('Steps')
    plt.ylabel('Agents')
    plt.colorbar()
    
def plot_bar_regrets(agents,eps,truerewards,labels,figsize,ncols=4):
    nrows = get_rows(len(agents),ncols)   
    fig, ax = plt.subplots(nrows,ncols,figsize=figsize,sharey=True,squeeze=False)

    for i,a in enumerate(agents):
        r,c = get_current_row_col(i,nrows,ncols)
        
        delta_A = np.max(truerewards[i]) - truerewards[i]
        
        ps = []
        for t in range(len(agents[i])):
            if (np.sum(agents[i][t].Q)>0): renorm_Q = (agents[i][t].Q * (1-eps[i])) / np.sum(agents[i][t].Q)
            else: renorm_Q = agents[i][t].Q
            uniform_V = np.ones(len(agents[i][t].Q)) * eps[i] / len(agents[i][t].Q)
            actual_P = renorm_Q + uniform_V
            ps.append(actual_P) 
        
        regrets = np.sum(delta_A*ps,axis=1)     
        ax[r,c].bar(range(len(agents[i])), regrets)
        ax[r,c].set_xlabel('Trial')
        ax[r,c].set_xticks(range(len(agents[i])))
        ax[r,c].hlines(np.mean(regrets),0,len(agents[i])-1,'r')
        ax[r,c].fill_between(np.arange(0,len(agents[i])-1,0.01), 
                           np.mean(regrets)+np.std(regrets),np.mean(regrets)-np.std(regrets), color='C3', alpha=0.3)
        ax[r,c].set_ylabel('Regret')
        ax[r,c].set_title('{0}: {1:.4f}'.format(labels[i],np.mean(regrets)))

    fig.tight_layout()
    
def plot_bar_Qs(agents,eps,truerewards,labels,figsize,ncols=4,gaps=True):
    nrows = get_rows(len(agents),ncols)   
    fig, ax = plt.subplots(nrows,ncols,figsize=figsize,sharey=True,squeeze=False)

    for i,a in enumerate(agents):
        r,c = get_current_row_col(i,nrows,ncols)
        
        Q_A = truerewards[i]
        delta_A = np.max(truerewards[i]) - truerewards[i]
        
        hatQs_A = np.array([a[t].Q for t in range(len(a))])
        mean_Qs_A = np.mean(hatQs_A,axis=0)
        std_Qs_A = np.std(hatQs_A,axis=0)
        
        ax[r,c].bar(np.arange(len(Q_A))-0.2, Q_A, width=0.2, color='black', label='true Q')
        ax[r,c].bar(np.arange(len(Q_A))+0.3, mean_Qs_A, yerr=std_Qs_A, width=0.5, color='C0', label='true Q')
        if gaps: ax[r,c].bar(np.arange(len(Q_A)), -delta_A, width=0.2, color='C7', label='gap')
        ax[r,c].set_xlabel('Actions')
        ax[r,c].set_ylabel('Return')
        
        ax[r,c].xaxis.set_ticks_position('none')
        
        ax[r,c].set_title('{0}'.format(labels[i]))

    fig.tight_layout()