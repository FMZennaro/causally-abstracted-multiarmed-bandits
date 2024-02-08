import numpy as np
import networkx as nx
import scipy.sparse as sparse

from src.CBN import CausalBayesianNetwork as CBN
from pgmpy.factors.discrete import TabularCPD as cpd

import src.utils as ut

def set_seed(seed):
     np.random.seed(seed)

def generate_random_mechanism(codom,dom):
    val = np.random.rand(codom,dom)
    return val / np.sum(val,axis=0)

def generate_random_alpha(dom,codom):
    surjective_values = np.arange(0,codom,dtype=int)
    non_surjective_values = np.random.randint(0,codom,(dom-codom),dtype=int)
    random_values = np.concatenate((surjective_values,non_surjective_values))
    np.random.shuffle(random_values)
    return (ut.map_vect2matrix(random_values)).copy()

def generate_random_alphas(M0,M1,a):
    M1names = list(M1.nodes)

    alphas = {}
    for m in M1names:
        dom,codom = ut.get_cardinalities_Fx(a,m,M0.get_cardinality(),M1.get_cardinality())
        alphas[m] = generate_random_alpha(dom,codom)

    return alphas

def get_random_DAG_adjacency(n, density):
    A = sparse.random(n, n, density=density).todense()
    A[A > 0] = 1
    A = np.tril(A, -1)

    for i in range(n):
        if np.sum(A[i, :]) == 0 and np.sum(A[:, i]) == 0:
            if i == (n - 1) or (i >= 2 and np.random.uniform() < .5):
                A[i, np.random.randint(0, i)] = 1
            else:
                A[np.random.randint(i + 1, n), i] = 1

    return A

def _compute_abstractions(alphas, alphakeys):
    alphas = [alphas[i] for i in alphakeys]
    return ut.tensorize_list(None, alphas)

def random_models_for_abstraction_0(n_vars_base, n_vars_abst, cardinalities_base, cardinalities_abst):
    # Create abstracted vars
    vars_abst = [str(i) for i in range(n_vars_abst)]

    # Create base variables
    surplus = n_vars_base - n_vars_abst

    abst_counterimage_sizes = []
    for i in range(n_vars_abst - 1):
        counterimage_size = np.random.choice(np.arange(surplus + 1), size=1)[0]
        abst_counterimage_sizes.append(counterimage_size + 1)
        surplus = surplus - counterimage_size
    abst_counterimage_sizes.append(surplus + 1)

    vars_base = [str(i) + '_' + str(j) for i in range(n_vars_abst) for j in range(abst_counterimage_sizes[i])]

    inverse_abst_nodes = [[j for j in vars_base if j[0] == str(i)] for i in range(n_vars_abst)]

    # Instantiate abstraction
    R = vars_base

    a = {}
    for i in range(n_vars_base):
        a[vars_base[i]] = vars_base[i][0]

    alphas = {}
    for i in range(n_vars_abst):
        dom = inverse_abst_nodes[i]
        dom_idx = np.searchsorted(vars_base, dom)
        card_dom = np.prod(cardinalities_base[dom_idx])
        card_codom = cardinalities_abst[i]
        alphas[vars_abst[i]] = generate_random_alpha(card_dom, card_codom)

    # Generate M1 DAG
    M1 = CBN()
    M1.add_nodes_from(vars_abst)

    A_abst = get_random_DAG_adjacency(n_vars_abst, 0.5)
    G = nx.from_numpy_matrix(A_abst.T, create_using=nx.DiGraph)

    M1.add_edges_from([(vars_abst[i], vars_abst[j]) for i, j in G.edges()])

    # Generate M0 DAG and model
    M0 = CBN()
    M0.add_nodes_from(vars_base)

    A_base = get_random_DAG_adjacency(n_vars_base, 0.5)
    G = nx.from_numpy_matrix(A_base.T, create_using=nx.DiGraph)

    M0.add_edges_from([(vars_base[i], vars_base[j]) for i, j in G.edges()])

    for i in range(n_vars_base):

        sources = np.where(A_base[i, :] > 0)[0]

        if sources.size == 0:
            evidence = None
        else:
            evidence = [vars_base[i] for i in sources]

        if sources.size == 0:
            evidence_card = None
        else:
            evidence_card = cardinalities_base[sources]
        total_card = np.prod(evidence_card)

        if sources.size == 0:
            phi = np.random.uniform(0, 1, size=(cardinalities_base[i], 1))
        else:
            phi = np.random.uniform(0, 1, size=(cardinalities_base[i], total_card))
        phi = phi / np.sum(phi, axis=0)

        cpd_i = cpd(variable=vars_base[i],
                    variable_card=cardinalities_base[i],
                    values=phi,
                    evidence=evidence,
                    evidence_card=evidence_card)

        M0.add_cpds(cpd_i)
    M0.check_model()

    # Generate M1 model
    for i in range(n_vars_abst):
        sources = np.where(A_abst[i, :] > 0)[0]

        if sources.size == 0:
            evidence = None
        else:
            evidence = [vars_abst[i] for i in sources]

        if sources.size == 0:
            evidence_card = None
        else:
            evidence_card = cardinalities_abst[sources]
        total_card = np.prod(evidence_card)

        if sources.size == 0:
            phi = np.random.uniform(0, 1, size=(cardinalities_abst[i], 1))
        else:
            phi = np.random.uniform(0, 1, size=(cardinalities_abst[i], total_card))
        phi = phi / np.sum(phi, axis=0)

        cpd_i = cpd(variable=vars_abst[i],
                    variable_card=cardinalities_abst[i],
                    values=phi,
                    evidence=evidence,
                    evidence_card=evidence_card)

        M1.add_cpds(cpd_i)

    M1.check_model()

    # Return data for abstraction
    return M0, M1, R, a, alphas


def random_models_for_abstraction_1(n_vars_base, n_vars_abst, cardinalities_base, cardinalities_abst):
    # Create abstracted vars
    vars_abst = [str(i) for i in range(n_vars_abst)]

    # Create base variables
    surplus = n_vars_base - n_vars_abst

    abst_counterimage_sizes = []
    for i in range(n_vars_abst - 1):
        counterimage_size = np.random.choice(np.arange(surplus + 1), size=1)[0]
        abst_counterimage_sizes.append(counterimage_size + 1)
        surplus = surplus - counterimage_size
    abst_counterimage_sizes.append(surplus + 1)

    vars_base = [str(i) + '_' + str(j) for i in range(n_vars_abst) for j in range(abst_counterimage_sizes[i])]

    inverse_abst_nodes = [[j for j in vars_base if j[0] == str(i)] for i in range(n_vars_abst)]

    # Instantiate abstraction
    R = vars_base

    a = {}
    for i in range(n_vars_base):
        a[vars_base[i]] = vars_base[i][0]

    alphas = {}
    for i in range(n_vars_abst):
        dom = inverse_abst_nodes[i]
        dom_idx = np.searchsorted(vars_base, dom)
        card_dom = np.prod(cardinalities_base[dom_idx])
        card_codom = cardinalities_abst[i]
        alphas[vars_abst[i]] = generate_random_alpha(card_dom, card_codom)

    # Generate M1 DAG
    M1 = CBN()
    M1.add_nodes_from(vars_abst)

    A_abst = get_random_DAG_adjacency(n_vars_abst, 0.5)
    G = nx.from_numpy_matrix(A_abst.T, create_using=nx.DiGraph)

    M1.add_edges_from([(vars_abst[i], vars_abst[j]) for i, j in G.edges()])

    # Generate M0 DAG and model
    M0 = CBN()
    M0.add_nodes_from(vars_base)

    A_base = np.zeros((n_vars_base, n_vars_base))

    for i in range(n_vars_abst):
        sources = np.where(A_abst[i, :])[0]
        if sources.size > 0:
            for s in sources:
                base_source = np.random.choice(inverse_abst_nodes[s])
                base_target = np.random.choice(inverse_abst_nodes[i])
                base_source_idx = np.where(np.array(vars_base) == base_source)[0][0]
                base_target_idx = np.where(np.array(vars_base) == base_target)[0][0]
                A_base[base_target_idx, base_source_idx] = 1

        internals = inverse_abst_nodes[i]
        if len(internals) > 1:
            A_internals = get_random_DAG_adjacency(len(internals), 0.5)
            internals_idxs = np.searchsorted(vars_base, internals)
            A_base[np.ix_(internals_idxs, internals_idxs)] = A_internals.T

    G = nx.from_numpy_matrix(A_base.T, create_using=nx.DiGraph)
    M0.add_edges_from([(vars_base[i], vars_base[j]) for i, j in G.edges()])

    for i in range(n_vars_base):

        sources = np.where(A_base[i, :] > 0)[0]

        if sources.size == 0:
            evidence = None
        else:
            evidence = [vars_base[i] for i in sources]

        if sources.size == 0:
            evidence_card = None
        else:
            evidence_card = cardinalities_base[sources]
        total_card = np.prod(evidence_card)

        if sources.size == 0:
            phi = np.random.uniform(0, 1, size=(cardinalities_base[i], 1))
        else:
            phi = np.random.uniform(0, 1, size=(cardinalities_base[i], total_card))
        phi = phi / np.sum(phi, axis=0)

        cpd_i = cpd(variable=vars_base[i],
                    variable_card=cardinalities_base[i],
                    values=phi,
                    evidence=evidence,
                    evidence_card=evidence_card)

        M0.add_cpds(cpd_i)
    M0.check_model()

    # Generate M1 model
    for i in range(n_vars_abst):
        abst_target = vars_abst[i]
        abst_sources = np.where(A_abst[i, :] > 0)[0]

        if abst_sources.size > 0:
            base_target = inverse_abst_nodes[i]
            base_sources = []
            for t in abst_sources:
                base_sources = base_sources + inverse_abst_nodes[t]

            alpha_sources = _compute_abstractions(alphas, [str(k) for k in abst_sources])
            alpha_target = alphas[abst_target]
            mu = M0.compute_mechanisms(base_sources, base_target)
            phi = np.dot(np.dot(alpha_target, mu), np.linalg.pinv(alpha_sources))

            cpd_i = cpd(variable=str(abst_target),
                        variable_card=cardinalities_abst[i],
                        values=phi,
                        evidence=[str(k) for k in abst_sources],
                        evidence_card=cardinalities_abst[abst_sources])

            M1.add_cpds(cpd_i)

        else:
            phi = np.random.uniform(0, 1, size=(cardinalities_abst[i], 1))
            phi = phi / np.sum(phi, axis=0)

            cpd_i = cpd(variable=str(abst_target),
                        variable_card=cardinalities_abst[i],
                        values=phi,
                        evidence=None,
                        evidence_card=None)

            M1.add_cpds(cpd_i)

    M1.check_model()

    # Return data for abstraction
    return M0, M1, R, a, alphas