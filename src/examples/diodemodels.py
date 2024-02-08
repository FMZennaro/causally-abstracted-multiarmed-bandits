
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd


class simpleShockley:
    def __init__(self,Is=10**-12,Vt = 26*10**-3):
        self.Is = Is
        self.Vt = Vt
    
    def sample_Vd(self):
        return stats.uniform(loc=0.85,scale=0.1).rvs(1)
    
    def sample_n(self):
        return stats.norm(loc=1.5,scale=0.01).rvs(1)
    
    def sample_Id(self):
        Vd,n,Id = self.sample()
        return Id
    
    def sample(self):
        Vd = self.sample_Vd()
        n = self.sample_n()
        Id = self.Is * np.exp(Vd / (n*self.Vt))
        return Vd,n,Id
    
    def generate_dataset(self,n_samples=1000):
        self.D = np.array([self.sample() for _ in range(n_samples)]).reshape((n_samples,3))
        self.DF = pd.DataFrame(self.D,columns=['Vd','n','Id'])
    
    def plot_f(self):
        plt.tricontourf(self.D[:,0],self.D[:,1],self.D[:,2])
        plt.colorbar()
        plt.xlabel('Vd')
        plt.ylabel('n')
        plt.title('Id')
        
    def plot_p(self):
        g = sns.PairGrid(self.DF)
        g.map(sns.histplot)

        
class simpleShockley_doVd(simpleShockley):
    def __init__(self,Is=10**-12,Vt = 26*10**-3,doVd=.9):
        super().__init__(Is,Vt)
        self.doVd = doVd
        
    def sample_Vd(self):
        return np.array([self.doVd])
    
    def plot_f(self):
        g = sns.scatterplot(self.D[:,1],self.D[:,2])
        g.set(xlabel='n', ylabel='Id')
        
    def plot_p(self):
        g = sns.jointplot(self.D[:,1],self.D[:,2])
        g.set_axis_labels('n', 'Id', fontsize=16)
        
class linearizedShockley:
    def __init__(self,Is=10**-12,Vt = 26*10**-3,x0=0.9):
        self.Is = Is
        self.Vt = Vt
        self.x0 = x0
    
    def sample_Vd(self):
        return stats.uniform(loc=0.85,scale=0.1).rvs(1)
    
    def sample_n(self):
        return stats.norm(loc=1.5,scale=0.01).rvs(1)
    
    def sample_Id(self):
        Vd,n,Id = self.sample()
        return Id
    
    def sample(self):
        Vd = self.sample_Vd()
        n = self.sample_n()
        
        f0 = self.Is * np.exp(self.x0 / (n*self.Vt))
        df0 = self.Is * np.exp(self.x0 / (n*self.Vt)) * (n / self.Vt)
        
        Id = f0 + df0 * (Vd - self.x0)
        return Vd,n,Id
    
    def generate_dataset(self,n_samples=1000):
        self.D = np.array([self.sample() for _ in range(n_samples)]).reshape((n_samples,3))
        self.DF = pd.DataFrame(self.D,columns=['Vd','n','Id'])
    
    def plot_f(self):
        plt.tricontourf(self.D[:,0],self.D[:,1],self.D[:,2])
        plt.colorbar()
        plt.xlabel('Vd')
        plt.ylabel('n')
        
    def plot_p(self):
        g = sns.PairGrid(self.DF)
        g.map(sns.histplot)
        
class linearizedShockley_doVd(linearizedShockley):
    def __init__(self,Is=10**-12,Vt = 26*10**-3,doVd=0.9):
        super().__init__(Is,Vt)
        self.doVd = doVd
        
    def sample_Vd(self):
        return np.array([self.doVd])
    
    def plot_f(self):
        g = sns.scatterplot(self.D[:,1],self.D[:,2])
        g.set(xlabel='n', ylabel='Id')
        
    def plot_p(self):
        g = sns.jointplot(self.D[:,1],self.D[:,2])
        g.set_axis_labels('n', 'Id', fontsize=16)