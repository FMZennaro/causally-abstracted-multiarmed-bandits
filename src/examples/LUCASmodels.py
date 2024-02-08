
import numpy as np
from src.CBN import CausalBayesianNetwork as CBN
from pgmpy.factors.discrete import TabularCPD as CPD

Anx = 'Anxiety'
PP = 'Peer Pressure'
Sm = 'Smoking'
YF = 'Yellow Fingers'
BED = 'Born an Even Day'
Gen = 'Genetics'
LC = 'Lung Cancer'
All = 'Allergy'
AD = 'Attention Disorder'
Cou = 'Coughing'
Fat = 'Fatigue'
CA = 'Car Accident'

def lucas0():
    M = CBN([(Anx,Sm),
           (PP,Sm),
           (Sm,YF),
           (Sm,LC),
           (Gen,LC),
           (Gen,AD),
           (All,Cou),
           (LC,Cou),
           (LC,Fat),
           (Cou,Fat),
           (Fat,CA),
           (AD,CA)])
    M.add_node(BED)

    ps = np.array([0.64277])
    cpd = CPD(variable=Anx,
             variable_card = 2,
             values = [1-ps,ps],
             evidence = None,
             evidence_card = None)
    M.add_cpds(cpd)

    ps = np.array([0.32997])
    cpd = CPD(variable=PP,
             variable_card = 2,
             values = [1-ps,ps],
             evidence = None,
             evidence_card = None)
    M.add_cpds(cpd)

    ps = np.array([0.43118,0.74591,0.8686,0.91576])
    cpd = CPD(variable=Sm,
             variable_card = 2,
             values = [1-ps,ps],
             evidence = [Anx,PP],
             evidence_card = [2,2])
    M.add_cpds(cpd)

    ps = np.array([0.23119,0.95372])
    cpd = CPD(variable=YF,
             variable_card = 2,
             values = [1-ps,ps],
             evidence = [Sm],
             evidence_card = [2])
    M.add_cpds(cpd)

    ps = np.array([0.15953])
    cpd = CPD(variable=Gen,
             variable_card = 2,
             values = [1-ps,ps],
             evidence = None,
             evidence_card = None)
    M.add_cpds(cpd)

    ps = np.array([0.5])
    cpd = CPD(variable=BED,
             variable_card = 2,
             values = [1-ps,ps],
             evidence = None,
             evidence_card = None)
    M.add_cpds(cpd)

    ps = np.array([0.23146,0.86996,0.83934,0.99351])
    cpd = CPD(variable=LC,
             variable_card = 2,
             values = [1-ps,ps],
             evidence = [Sm,Gen],
             evidence_card = [2,2])
    M.add_cpds(cpd)

    ps = np.array([0.28956,0.68706])
    cpd = CPD(variable=AD,
             variable_card = 2,
             values = [1-ps,ps],
             evidence = [Gen],
             evidence_card = [2])
    M.add_cpds(cpd)

    ps = np.array([0.32841])
    cpd = CPD(variable=All,
             variable_card = 2,
             values = [1-ps,ps],
             evidence = None,
             evidence_card = None)
    M.add_cpds(cpd)

    ps = np.array([0.1347,0.64592,0.7664,0.99947])
    cpd = CPD(variable=Cou,
             variable_card = 2,
             values = [1-ps,ps],
             evidence = [LC,All],
             evidence_card = [2,2])
    M.add_cpds(cpd)

    ps = np.array([0.35212,0.56514,0.80016,0.89589])
    cpd = CPD(variable=Fat,
             variable_card = 2,
             values = [1-ps,ps],
             evidence = [Cou,LC],
             evidence_card = [2,2])
    M.add_cpds(cpd)

    ps = np.array([0.2274,0.779,0.78861,0.97169])
    cpd = CPD(variable=CA,
             variable_card = 2,
             values = [1-ps,ps],
             evidence = [Fat,AD],
             evidence_card = [2,2])
    M.add_cpds(cpd)

    if M.check_model(): return M