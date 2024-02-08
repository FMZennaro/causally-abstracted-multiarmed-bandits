import numpy as np
from src.CBN import CausalBayesianNetwork as CBN
from pgmpy.factors.discrete import TabularCPD as cpd

from src.SCMMappings import Abstraction
from src.measuring import ICEvaluator
import src.evaluationsets as esets

M0 = CBN([('Smoking','Tar'),('Tar','Cancer')])

cpdS = cpd(variable='Smoking',
          variable_card=2,
          values=[[.8],[.2]],
          evidence=None,
          evidence_card=None)
cpdT = cpd(variable='Tar',
          variable_card=2,
          values=[[1,.2],[0.,.8]],
          evidence=['Smoking'],
          evidence_card=[2])
cpdC = cpd(variable='Cancer',
          variable_card=2,
          values=[[.9,.6],[.1,.4]],
          evidence=['Tar'],
          evidence_card=[2])

M0.add_cpds(cpdS,cpdT,cpdC)
M0.check_model()

M1 = CBN([('Smoking_','Cancer_')])

cpdS = cpd(variable='Smoking_',
          variable_card=2,
          values=[[.8],[.2]],
          evidence=None,
          evidence_card=None)
cpdC = cpd(variable='Cancer_',
          variable_card=2,
          values=[[.9,.66],[.1,.34]],
          evidence=['Smoking_'],
          evidence_card=[2])

M1.add_cpds(cpdS,cpdC)
M1.check_model()

R = ['Smoking','Cancer']

a = {'Smoking': 'Smoking_',
    'Cancer': 'Cancer_'}

alphas = {'Smoking_': np.eye(2),
         'Cancer_': np.eye(2)}

A = Abstraction(M0,M1,R,a,alphas)
Ae = ICEvaluator(A)

J = esets.get_pairs_in_M1_with_directed_path_in_M1_or_M0(M0,M1,a)
print(J)

Ae.compute_overall_error(J)