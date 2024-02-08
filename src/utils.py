
import numpy as np
from scipy.sparse import coo_matrix
from scipy.special import kl_div
from ot import wasserstein_1d

import itertools
from functools import reduce

def inverse_fx(f,x):
    """
    Given a function f as a dictionary, it computes f^-1(x).

    Args:
        f: a function encoded in a dictionary
        x: a value encoded as a dictionary value

    Returns:
        List of all the keys f^-1(x)
        
    Example: 
        f = {'a': X, 'b': Y, 'c': X};
        x = X;
        res = ['a','c'].
    """
    return list(np.array(list(f.keys()))[np.where(np.in1d(np.array(list(f.values())),x))[0]])

def get_cardinalities_Fx(F,x,cards_domain,cards_codomain):
    """
    It computes the cardinality of domain and codomain, of a function f out of a dictionary of functions F.

    Args:
        F: a dictionary of set functions encoded as a dictionary
        alpha: a value encoded as a dictionary value
        cards_domain: dictionary of set cardinalities
        cards_codomain: dictionary of set cardinalities

    Returns:
        cards_domain: the cardinality of the domain of F_alpha;
        
        cards_codomain: the cardinality of the codomain of F_alpha.
        
    Example:
        F = {'A': X, 'B': Y, 'C': X};
        x = X;
        cards_domain = {'A': 5, 'B': 4, 'C': 2};
        cards_codomain = {'X': 3, 'Y': 4};
        res = (10,3).
    """
    domain = inverse_fx(F,x)
    card_domain = 1
    for d in domain:
        card_domain = card_domain * cards_domain[d]
    card_codomain = cards_codomain[x]
    return card_domain,card_codomain

def is_list_contained_in_list(subset,superset):
    """
    Returns whether the list subset is set-contained in the list superset.

    Args:
        subset: list
        superset: list

    Returns:
        True if subset is contained in superset
        
    Example:
        subset = [0,1];
        superset = [1,2,0];
        res = True.
    
    """ 
    return np.all(np.in1d(np.array(subset),np.array(superset)))

def is_list_contents_unique(ulist):
    """
    Returns whether the list contains duplicates.

    Args:
        ulist: list

    Returns:
        True if the list does not contain duplicates
        
    Example:
        ulist = [0,1,0];
        res = True.
    """ 
    return len(set(ulist)) == len(ulist)

def is_surjective(frange,fcodomain):
    """
    Check the surjectivity conditions by evaluating whether codomain and range are equal

    Args:
        frange: list
        fcodomain: list

    Returns:
        True if codomain and range are set equal 
        
    Example:
        range = [0,1,0];
        codomain = [0,1,2];
        res = False.
    """ 
    return set(fcodomain)==set(frange)

def is_matrix_surjective(M):
    """
    Check whether a binary matrix may encode a surjective function

    Args:
        M: 2D binary numpy array

    Returns:
        True if every row contains at least a 1
        
    Example:
        M = [[0,1,0],
             [0,0,1],
             [1,0,0]].
        res = True.
    """ 
    return np.all(np.sum(M,axis=1)>=1)

def flat_tensor_product(x,y):
    """
    Compute the tensor product of two 2D matrix and flatten it into a new 2D matrix
    
    Args:
        x: 2D numpy array
        y: 2D numpy array

    Returns:
        2D numpy array of the tensor product of x and y
        
    Example:
        x.shape = (3,7);
        y.shape = (5,4);
        res.shape = (15,28).
    """
    tensor = np.einsum('ij,kl->ikjl',x,y)
    return tensor.reshape((tensor.shape[0]*tensor.shape[1],tensor.shape[2]*tensor.shape[3]))

def tensorize_list(tensor,l):
    """
    Compute the tensor product of two 2D matrix and flatten it into a new 2D matrix

    Args:
        tensor: 2D numpy array
        l: list of 2D numpy arrays

    Returns:
        2D numpy array of the tensor product of tensor and the elements in the list
        
    Example:
        tensor = None;
        l = [x,y,z];
            x.shape = (3,7);
            y.shape = (5,4);
            z.shape = (3,2);
        res.shape = (45,56).
    """ 
    if tensor is None:
        if len(l)>1:
            tensor = flat_tensor_product(l[0],l[1])
            return tensorize_list(tensor,l[2:])
        else:
            return l[0]
    else:
        if len(l)>0:
            tensor = flat_tensor_product(tensor,l[0])
            return tensorize_list(tensor,l[1:])
        else:
            return tensor

def invert_matrix_max_entropy(A):
    """
    Compute the inverse of matrix A by transposting and normalizing the column

    Args:
        A: 2D numpy array

    Returns:
        invA: 2D numpy array
        
    Example:
        A.shape = (3,7);
        res.shape = (7,3).
    """
    invA = np.transpose(A)
    invA = invA / np.sum(invA,axis=0)
    return invA
    
def invert_matrix_pinv(A):
    """
    Compute the pseudo-inverse of matrix A

    Args:
        A: 2D numpy array

    Returns:
        invA: 2D numpy array
        
    Example:
        A.shape = (3,7);
        res.shape = (7,3).
    """
    invA = np.linalg.pinv(A)
    return invA

def map_vect2matrix(v,rows=None):
    """
    Covert an integer vector into a binary matrix

    Args:
        v: integer vector
        rows: integer denoting the number of rows (if the vectors encode a surjective map this param is optional)

    Returns:
        M: 2D numpy array
        
    Example:
        v = [2,0,1];
        M = [[0,1,0],
             [0,0,1],
             [1,0,0]].
    """
    dom = len(v)
    if rows==None: 
        codom = np.max(v)+1
    else:
        codom = rows
    M = coo_matrix((np.ones(dom), (v,np.arange(dom))), shape=(codom,dom), dtype=np.int32).toarray()
    return M

def map_matrix2vect(M):
    """
    Covert a binary matrix into an integer vector 

    Args:
        M: 2D numpy array

    Returns:
        v: integer vector
        
    Example:
        M = [[0,1,0],
             [0,0,1],
             [1,0,0]];
        v = [2,0,1].
    """
    matrix = coo_matrix(M)
    rows = matrix.row
    cols = matrix.col
    idxs = np.argsort(cols)
    v = matrix.row[idxs]
    return v

def powerset(iterable):
    """
    It computes the power set of a set.

    Args:
        iterable: an iterable

    Returns:
        The power set of iterable
    """
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def jsd(p,q):
    """
    A robust method to compute Jensen-Shannon distance between the columns of two matrices

    Args:
        p: 2D numpy array of shape MxN
        q: 2D numpy array of shape MxN

    Returns:
        jsd: 1D numpy array of length N
        
    Example:
        p = [[.9,.7],
            [.1,.3]];
        q = [[.9,.2],
            [.1,.8]];
        jsd = [0.,0.364].
    """
    if np.allclose(p,q, rtol=0, atol=1e-08): return 0
    jsd = np.sqrt(.5*np.sum(kl_div(p,(p+q)/2),axis=0) + .5*np.sum(kl_div(q,(p+q)/2),axis=0))
    return jsd

def wasserstein_2(p,q):
    """
    A method to compute Wasserstein_2 distance between the columns of two matrices

    Args:
        p: 2D numpy array of shape MxN
        q: 2D numpy array of shape MxN

    Returns:
        w: 1D numpy array of length N
        
    Example:
        p = [[.9,.7],
            [.1,.3]];
        q = [[.9,.2],
            [.1,.8]];
        w = [0.,0.01].
    """
    return wasserstein_1d(p,q,p=2)

def convert_binary_to_decimal(n):
    return reduce(lambda x,y: 2*x+y, n)

def convert_decimal_to_binary(n):
    return [int(i) for i in bin(n)[2:]]

def unique(l):
    """
        It removes the duplicates from a list.

        Args:
            l: a list

        Returns:
            The list without duplicates
        """
    #return list(set(l)) #Does not preserve order

    return sorted(set(l),key=l.index)