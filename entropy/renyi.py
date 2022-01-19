# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 17:03:53 2022

@author: Wuestney
"""
import numpy as np

def discrete(n, alpha=2):
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