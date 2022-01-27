# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 20:12:55 2022

@author: Wuestney

Some variable naming is chosen to be in line with the notation used in the papers by Pincus (1991)
and Richman & Moorman (2000).
"""
import numpy as np

def discrete_renyi(n, alpha=2):
    """
    Given a frequency vector, this function calculates the discrete renyi entropy as 1/(1-alpha)*log2(sum(p**alpha))

    Parameters
    ----------
    n : LIST LIKE OBJECT OF L-TUPLE FREQUENCY COUNTS
    alpha : PARAMETER FOR THE RENYI OPERATION, DEFINES RENYI ORDER

    Returns
    -------
    r, rmax, rmin : r is the discrete renyi(alpha) entropy of the sequence;
            rmax and rmin are the max possible entropy for alphabet len(n) and the minimum entropy as alpha approaches infinity

    """

    p = np.array(n)/np.array(n).sum()

    #define special case for Shannon's entropy alpha=1
    if alpha==1:
        p = np.where(p==0, 1, p)
        r = -sum(p*np.log2(p))
        print("Shannon's entropy is:", r)

    else:
        r= 1/(1-alpha)*np.log2(np.sum(p**alpha))
    rmax = np.log2(len(n)) #the case of the uniform distribution or if alpha=0
    rmin = -np.log2(np.max(p)) #the case as alpha -> infinity

    return r, rmax, rmin

def vector_matches(data, m, r=0, enttype=None):
    """
    Calculates the number of sequences matching the template within the margin r.

    Parameters
    ----------
    data : ITERABLE REPRESENTING THE FULL DATA SEQUENCE
    m : TEMPLATE LENGTH
    r : FILTER SIZE or MAX DISTANCE BETWEEN TWO VECTORS FOR THE VECTORS TO BE CONSIDERED MATCHING
    enttype : NONE OR STRING
        IF None WILL RETURN A SINGLE INTEGER SUM OVER ALL XI/XJ PAIRS. IF 'sampen'
        WILL GIVE SUM OVER ALL XI/XJ PAIRS, SUBTRACTING len(data) TO REMOVE SELF
        MATCHES. IF 'apen' WILL RETURN AN ARRAY OF THE NUMBER OF XI/XJ MATCHES
        FOR EACH XI.

    Returns
    -------
    matches

    matches : integer representing the number of matching vectors length(m) within a distance r contained within the full sequence

    """
    xmi = np.array([data[i:i+m] for i in range(len(data)-m+1)])

    #z is the number of m-vectors in xmi
    z = len(xmi)

    #create 3d array of xmi vectors where each xi in xmi is its own 1xm subarray
    xi_matrix = np.stack([xmi], axis=2).reshape((z,1,m))

    #dif is a 3D array containing the pairwise kronecker delta between xi and xmi for all xi.
    dif = np.invert(xi_matrix==xmi).astype(int)
    #dif.sum(axis=2) evaluates to 0 for xi that fully matched and >0 for xi that did not fully match
    sim_dist = dif.sum(axis=2)

    if enttype == 'sampen':
        #subtract len(xmi) to remove self matches from the total
        matches = np.sum(sim_dist==r) - len(xmi)
    elif enttype == 'apen':
        matches= np.sum(sim_dist==r, axis=1)
    elif enttype == None:
        matches = np.sum(sim_dist==r)
    return matches

def sampen(data, m):
    """
    This is a function to measure the sample entropy of a given string or
    discrete-valued (categorical) time series. For this implementation, we follow
    the same terminology and notation used by Richman & Moorman (2000). The
    data sequence, X={u(1), u(2), u(i), ..., u(n-m)}, is used to form a sequence, xmi, of
    m-length vectors where each xi in xmi is the vector of m data points from
    u(i) to u(i+m-1). For each m-length vector, xm(i) for 1 <= i <= N-m, in X,
    let Bi be the number of vectors, xm(j) for 1<=j<= N-m and j!= i, in X that match xm(i).
    The xm(i) is called the template vector, and an xm(j) which matches
    with xm(i) is called the template match. For each m+1-length vector, xm1(i) for
    1 <= i <= N-m, let Ai be the number of vectors, xm1(j) for 1<=j<= N-m and j!= i,
    in X that match xm1(i). Then let B be the sum of Bi over all i and A be the sum
    of Ai over all i.

    with the notable
    exception that the parameter, r, is omitted. This is due to the fact that for
    integer coded discrete-valued sequences, the tolerance value is set to r<1 and thus
    the impact of r is suppressed because r < min(|x-y|, x != y, x and y state space values).
    Therefore only template

    Parameters
    ----------
    data : STRING OR LIST-LIKE ITERABLE OF INTEGERS
        INTEGER CODED DISCRETE-VALUED DATA SEQUENCE.

    m : INTEGER
        THE LENGTH OF THE TEMPLATE SEQUENCE.

    Returns
    -------
    sampen, B, A

    sampen : floating point representing the Sample Entropy of the sequence

    B : integer representing number of possible matches, length m

    A : integer representing number of matches, length m+1
    """

    #get number of m-length matches, store in variable 'B'
    #use data sequence with ultimate value removed so that number of m-length vectors equals the number of m+1-length vectors.

    B = vector_matches(data[:-1], m, enttype='sampen')
    #print(B)

    #get number of matches, store in variable 'A'
    k = m+1
    A = vector_matches(data, k, enttype='sampen')
    #print(A)
    if A == 0:
        print("The data set is unique, there were no m+1-length matches.")
        sampen = 'Undefined'
        print(sampen, B, A)
        return sampen, B, A
    if B == 0:
        print("The data set is unique, there were no m-length matches.")
        sampen = 'Undefined'
        print(sampen, B, A)
        return sampen, B, A
    else:
        ratio = A/B
        sampen = np.negative(np.log(ratio))
        return sampen, B, A

def apen(data, m):
    N = len(data)
    Bi = vector_matches(data[:-1], m, enttype='apen')
    k=m+1
    Ai = vector_matches(data, k, enttype='apen')
    #print('N-m', N-m)
    #print('Bi', len(Bi), 'Ai', len(Ai))
    #print(Bi.shape == Ai.shape)
    apen = np.sum(np.negative(np.log(Ai/Bi)))/(N-m)
    return apen, Bi, Ai
