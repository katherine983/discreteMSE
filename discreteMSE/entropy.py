# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 20:12:55 2022

@author: Wuestney

Some variable naming is chosen to be in line with the notation used in the papers by Pincus (1991)
and Richman & Moorman (2000).
"""
import numpy as np
#from memory_profiler import profile

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

#@profile
def vector_matches(data, m, r=0):
    """
    Calculates the number of sequences matching the template within the margin r.

    Parameters
    ----------
    data : ITERABLE REPRESENTING THE FULL DATA SEQUENCE
    m : TEMPLATE LENGTH
    r : FILTER SIZE or MAX DISTANCE BETWEEN TWO VECTORS FOR THE VECTORS TO BE CONSIDERED MATCHING

    Returns
    -------
    matches

    """
    xmi = np.array([data[i:i+m] for i in range(len(data)-m+1)])

    #z is the number of m-vectors in xmi
    z = len(xmi)

    #create 3d array of xmi vectors where each xi in xmi is its own 1xm subarray
    xi_matrix = np.stack([xmi], axis=2).reshape((z,1,m))

    #dif is a 3D array containing the pairwise inverse kronecker delta between xi and xmi for all xi.
    dif = np.invert(xi_matrix==xmi).astype(np.uint8)
    #dif.sum(axis=2) evaluates to 0 for xi that fully matched and >0 for xi that did not fully match
    sim_dist = dif.sum(axis=2)
    matches= np.sum(sim_dist==r, axis=1)
    """
    if enttype == 'sampen':
        #subtract len(xmi) to remove self matches from the total
        matches = np.sum(sim_dist==r) - len(xmi)
    elif enttype == 'apen':
        matches= np.sum(sim_dist==r, axis=1)
    elif enttype == None:
        matches = np.sum(sim_dist==r)
        """
    return matches

#@profile
def sampen(data, m, refseq=None):
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

    refseq : STR, NONE
        USE TO SUPPLY A NAME OF THE DATA SEQUENCE IN STRING FORMAT.

    Returns
    -------
    sampen, B, A

    sampen : floating point representing the Sample Entropy of the sequence

    B : integer representing number of possible matches, length m

    A : integer representing number of matches, length m+1
    """

    #get number of m-length matches, store in variable 'B'
    #use data sequence with ultimate value removed so that number of
    #m-length vectors equals the number of m+1-length vectors.
    Bi = vector_matches(data[:-1], m)
    B = np.sum(Bi) - (len(data)-m)
    #print(B)

    #get number of m+1-length matches, store in variable 'A'
    k = m+1
    Ai = vector_matches(data, k)
    A = np.sum(Ai) - (len(data)-m)
    #print(A)
    if A == 0:
        print(f"The sequence {refseq} is unique, there were no m+1-length matches.")
        sampen = 'Undefined'
        print(sampen, B, A)
        return sampen, B, A
    if B == 0:
        print("The sequence {refseq} is unique, there were no m-length matches.")
        sampen = 'Undefined'
        print(sampen, B, A)
        return sampen, B, A
    else:
        sampen = np.negative(np.log(A/B))
        return sampen, B, A

#@profile
def apen(data, m, version='approx'):
    """


    Parameters
    ----------
    data : STRING OR LIST-LIKE ITERABLE OF INTEGERS
        INTEGER CODED DISCRETE-VALUED DATA SEQUENCE.
    m : INTEGER
        THE LENGTH OF THE TEMPLATE SEQUENCE.
    version : STRING, optional
        SPECIFIES WHICH TYPE OF APEN ALGORITHM TO USE TO CALCULATE APEN. The default is 'approx'.
        The value 'approx' will use the approximation of the ApEn statistic,
        ApEn ~ (N-m)^-1 * SUM(i=0, N-m-1)ln(A_i/B_i).
        The value 'phi' will calculate ApEn as the difference Phi_m(r) - Phi_m+1(r),
        which is equivalent to
        ((N-m+1)**(-1) * SUM(i=0, N-m) ln(B_i/(N-m+1))) - ((N-m)**(-1) * SUM(i=0, N-m-1) ln(A_i/(N-m)))

    Returns
    -------
    apen : NUMERIC, FLOAT
        THE VALUE OF APPROXIMATE ENTROPY PER THE METHOD SPECIFIED BY version.
    Bi : ARRAY
        NUMPY ARRAY CONTAINING THE VALUES OF B_i FOR EACH i.
    Ai : ARRAY
        NUMPY ARRAY CONTAINING THE VALUES OF A_i FOR EACH i.

    """
    if version == 'approx':
        N = len(data)
        Bi = vector_matches(data[:-1], m)
        k=m+1
        Ai = vector_matches(data, k)
        #print('N-m', N-m)
        #print('Bi', len(Bi), 'Ai', len(Ai))
        #print(Bi.shape == Ai.shape)
        apen = np.sum(np.negative(np.log(Ai/Bi)))/(N-m)

    if version == 'phi':
        N = len(data)
        Bi = vector_matches(data, m)
        k=m+1
        Ai = vector_matches(data, k)
        Cmi = Bi/(N-m+1)
        Cm1i = Ai/(N-m)
        phim = (np.log(Cmi).sum())/(N-m+1)
        phim1 = (np.log(Cm1i).sum())/(N-m)
        apen = phim - phim1

    return apen, Bi, Ai

if __name__ == "__main__":
    nobs = [1000, 5000, 10000, 25000, 50000]
    for n in nobs:
        data = np.random.randint(1, 27, size=n, dtype=np.uint8)
        ApEn, Bi, Ai = apen(data, 2)
        SampEn, B, A = sampen(data, 2)
