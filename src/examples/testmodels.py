
import numpy as np

from src.examples import randomgenerators as rg
from src.CBN import CausalBayesianNetwork as CBN
from pgmpy.factors.discrete import TabularCPD as cpd

def Mt0_0():
    M = CBN([('A','C'), ('E','B'), ('E','C'), ('B','C'), ('C','D'), ('C','F')])

    cpdA = cpd(variable='A',
              variable_card=3,
              values=rg.generate_random_mechanism(3,1),
              evidence=None,
              evidence_card=None)

    cpdE = cpd(variable='E',
              variable_card=2,
              values=rg.generate_random_mechanism(2,1),
              evidence=None,
              evidence_card=None)

    cpdB = cpd(variable='B',
              variable_card=4,
              values=rg.generate_random_mechanism(4,2),
              evidence=['E'],
              evidence_card=[2])

    cpdC = cpd(variable='C',
              variable_card=6,
              values=rg.generate_random_mechanism(6,24),
              evidence=['A','B','E'],
              evidence_card=[3,4,2])

    cpdD = cpd(variable='D',
              variable_card=3,
              values=rg.generate_random_mechanism(3,6),
              evidence=['C'],
              evidence_card=[6])

    cpdF = cpd(variable='F',
              variable_card=2,
              values=rg.generate_random_mechanism(2,6),
              evidence=['C'],
              evidence_card=[6])

    M.add_cpds(cpdA,cpdB,cpdC,cpdD,cpdE,cpdF)
    if M.check_model(): return M
    
def Mt0_1():
    M = CBN([('X','Y'), ('Y','Z'), ('Y','W'), ('X','Z')])

    cpdX = cpd(variable='X',
              variable_card=3,
              values=rg.generate_random_mechanism(3,1),
              evidence=None,
              evidence_card=None)

    cpdY = cpd(variable='Y',
              variable_card=4,
              values=rg.generate_random_mechanism(4,3),
              evidence=['X'],
              evidence_card=[3])

    cpdZ = cpd(variable='Z',
              variable_card=2,
              values=rg.generate_random_mechanism(2,12),
              evidence=['Y','X'],
              evidence_card=[4,3])

    cpdW = cpd(variable='W',
              variable_card=2,
              values=rg.generate_random_mechanism(2,4),
              evidence=['Y'],
              evidence_card=[4])

    M.add_cpds(cpdX,cpdY,cpdZ,cpdW)
    if M.check_model(): return M

def get_A_Mt0_1_Mt0_1():
    R = ['A','B', 'C', 'D', 'F']
    a = {'A': 'X',
         'B': 'X',
         'C': 'Y',
         'D': 'Z',
         'F': 'W'}
    alphas = {'X': rg.generate_random_mechanism(3,12),
             'Y': rg.generate_random_mechanism(4,6),
             'Z': rg.generate_random_mechanism(2,3),
             'W': rg.generate_random_mechanism(2,2)}
    return Mt0_0(),Mt0_1(),R,a,alphas

def Mt1_0():
    M = CBN([('A','C'), ('A','D'), ('B','D'), ('B','E'), ('B','F'), ('F','I'), ('D','H'), ('G','E'), ('G','F')])

    cpdA = cpd(variable='A',
              variable_card=3,
              values=rg.generate_random_mechanism(3,1),
              evidence=None,
              evidence_card=None)

    cpdB = cpd(variable='B',
              variable_card=4,
              values=rg.generate_random_mechanism(4,1),
              evidence=None,
              evidence_card=None)

    cpdC = cpd(variable='C',
              variable_card=7,
              values=rg.generate_random_mechanism(7,3),
              evidence=['A'],
              evidence_card=[3])

    cpdD = cpd(variable='D',
              variable_card=4,
              values=rg.generate_random_mechanism(4,12),
              evidence=['A','B'],
              evidence_card=[3,4])

    cpdE = cpd(variable='E',
              variable_card=3,
              values=rg.generate_random_mechanism(3,8),
              evidence=['B','G'],
              evidence_card=[4,2])

    cpdF = cpd(variable='F',
              variable_card=3,
              values=rg.generate_random_mechanism(3,8),
              evidence=['B','G'],
              evidence_card=[4,2])

    cpdG = cpd(variable='G',
              variable_card=2,
              values=rg.generate_random_mechanism(2,1),
              evidence=None,
              evidence_card=None)

    cpdH = cpd(variable='H',
              variable_card=6,
              values=rg.generate_random_mechanism(6,4),
              evidence=['D'],
              evidence_card=[4])

    cpdI = cpd(variable='I',
              variable_card=5,
              values=rg.generate_random_mechanism(5,3),
              evidence=['F'],
              evidence_card=[3])

    M.add_cpds(cpdA,cpdB,cpdC,cpdD,cpdE,cpdF,cpdG,cpdH,cpdI)
    if M.check_model(): return M
    
def Mt1_1():
    M = CBN([('X','Z'), ('Y','W'), ('Y','V')])
    M.add_node('U')

    cpdX = cpd(variable='X',
              variable_card=3,
              values=rg.generate_random_mechanism(3,1),
              evidence=None,
              evidence_card=None)

    cpdY = cpd(variable='Y',
              variable_card=3,
              values=rg.generate_random_mechanism(3,1),
              evidence=None,
              evidence_card=None)

    cpdZ = cpd(variable='Z',
              variable_card=24,
              values=rg.generate_random_mechanism(24,3),
              evidence=['X'],
              evidence_card=[3])

    cpdW = cpd(variable='W',
              variable_card=2,
              values=rg.generate_random_mechanism(2,3),
              evidence=['Y'],
              evidence_card=[3])

    cpdV = cpd(variable='V',
              variable_card=8,
              values=rg.generate_random_mechanism(8,3),
              evidence=['Y'],
              evidence_card=[3])

    cpdU = cpd(variable='U',
              variable_card=2,
              values=rg.generate_random_mechanism(2,1),
              evidence=None,
              evidence_card=None)

    M.add_cpds(cpdX,cpdY,cpdZ,cpdW,cpdV,cpdU)
    if M.check_model(): return M
    
def get_A_Mt1_1_Mt1_1():
    R = ['A','B', 'D', 'E', 'F', 'G', 'H', 'I']
    a = {'A': 'X',
         'B': 'Y',
         'D': 'Z',
         'E': 'W',
         'F': 'V',
         'G': 'U',
         'H': 'Z',
         'I': 'V'}
    alphas = {'X': rg.generate_random_mechanism(3,3),
             'Y': rg.generate_random_mechanism(3,4),
             'Z': rg.generate_random_mechanism(24,24),
             'W': rg.generate_random_mechanism(2,3),
             'V': rg.generate_random_mechanism(8,15),
             'U': rg.generate_random_mechanism(2,2),
             }
    return Mt1_0(),Mt1_1(),R,a,alphas

def instantiate_chain_models(mA,mBA,mX,mYX):
    M0 = CBN([('A','B')])
    cpdA = cpd(variable='A',
              variable_card=mA.shape[0],
              values=mA,
              evidence=None,
              evidence_card=None)
    cpdB = cpd(variable='B',
              variable_card=mBA.shape[0],
              values=mBA,
              evidence=['A'],
              evidence_card=[mA.shape[0]])
    M0.add_cpds(cpdA,cpdB)
    M0.check_model()
    
    M1 = CBN([('A','B')])
    cpdX = cpd(variable='X',
              variable_card=mX.shape[0],
              values=mX,
              evidence=None,
              evidence_card=None)
    cpdY = cpd(variable='Y',
              variable_card=mYX.shape[0],
              values=mYX,
              evidence=['X'],
              evidence_card=[mX.shape[0]])
    M1.add_cpds(cpdX,cpdY)
    M1.check_model()

    return M0,M1

def M0_simpletest(mech=None):
    if mech is None: mech = np.array([[0.2,0.1,],[0.3,0.6],[0.5,0.3]], dtype=np.float32)
    
    M0 = CBN([('a','b')])

    cpdA = cpd(variable='a',
              variable_card=2,
              values=[[0.5],[0.5]],
              evidence=None,
              evidence_card=None)
    cpdB = cpd(variable='b',
              variable_card=3,
              values=mech,
              evidence=['a'],
              evidence_card=[2])

    M0.add_cpds(cpdA,cpdB)
    M0.check_model()    
    return M0

def M1_simpletest(mech=None):
    if mech is None: mech = np.array([[0.7,0.4],[0.3,0.6]], dtype=np.float32)
    
    M1 = CBN([('x','y')])

    cpdX = cpd(variable='x',
              variable_card=2,
              values=[[0.5],[0.5]],
              evidence=None,
              evidence_card=None)
    cpdY = cpd(variable='y',
              variable_card=2,
              values=mech,
              evidence=['x'],
              evidence_card=[2])


    M1.add_cpds(cpdX,cpdY)
    M1.check_model()   
    return M1

def A_simpletest(mech_low=None,mech_high=None,alphas=None):
    M0 = M0_simpletest(mech_low)
    M1 = M1_simpletest(mech_high)
    R = ['a','b']
    a = {'a': 'x',
         'b': 'y'}
    if alphas is None:
        alphas = {"x": np.array([[1,0],[0,1]]),
                 "y": np.array([[1,0,1],[0,1,0]])}
    return M0,M1,R,a,alphas

def M0_synthetictest():
    M0 = CBN([('A','B'), ('A','C'), ('B','C'), ('C','D'), ('C','E')])

    cpdA = cpd(variable='A',
              variable_card=2,
              values=[[0.3],[0.7]],
              evidence=None,
              evidence_card=None)
    cpdB = cpd(variable='B',
              variable_card=2,
              values=[[.2,.7],[.8,.3]],
              evidence=['A'],
              evidence_card=[2])
    cpdC = cpd(variable='C',
              variable_card=2,
              values=[[.15,.85,.65,.75],[.85,.15,.35,.25]],
              evidence=['A','B'],
              evidence_card=[2,2])
    cpdD = cpd(variable='D',
              variable_card=2,
              values=[[.2,.0],[.8,1]],
              evidence=['C'],
              evidence_card=[2])
    cpdE = cpd(variable='E',
              variable_card=3,
              values=[[.3,.8],[.6,.1],[.1,.1]],
              evidence=['C'],
              evidence_card=[2])


    M0.add_cpds(cpdA,cpdB,cpdC,cpdD,cpdE)
    M0.check_model()
    return M0
    
def M1_synthetictest():
    M1 = CBN([('X','Y'), ('Y','Z')])

    cpdX = cpd(variable='X',
              variable_card=2,
              values=[[0.5],[0.5]],
              evidence=None,
              evidence_card=None)
    cpdY = cpd(variable='Y',
              variable_card=2,
              values=[[.15,.75],[.85,.25]],
              evidence=['X'],
              evidence_card=[2])
    cpdZ = cpd(variable='Z',
              variable_card=2,
              values=[[.8,.2],[.2,.8]],
              evidence=['Y'],
              evidence_card=[2])


    M1.add_cpds(cpdX,cpdY,cpdZ)
    M1.check_model()
    return M1
    
def A_synthetictest():
    M0 = M0_synthetictest()
    M1 = M1_synthetictest()
    R = ['A','B','C','E']
    a = {'A': 'X',
         'B': 'X',
         'C': 'Y',
         'E': 'Z'}
    alphas = {'X': np.array([[1,1,1,.0],[0,0,0,1.]]),
              'Y': np.eye(2),
              'Z': np.array([[1,1,.0],[0,0,1.]])}
    return M0,M1,R,a,alphas
synthetic = A_synthetictest