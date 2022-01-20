# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:17:10 2020

@author: wuest
"""
import numpy as np
import copy
from more_itertools import windowed, collapse, sliced
from itertools import product
from collections import Counter
from math import log
import pandas as pd


def sim_distance(xmi, m, xixjprob_df=None):
    """
    Calculates the number of vectors in xmi whose max elementwise similarity distance from the template vector, xi, is <=r.

    If xixjprob_df provided:
        The similarity distance between two vector elements is the reverse Kronecker delta
        times the complement of the conditional probability of the elementwise pair.
            Reverse Kronecker delta, d(x,y):
                if x==y, then d(x,y) = 0;
                if x!=y, then d(x,y) = 1
            The values of the delta function are then multiplied by the complement of the conditional probability of the comparison vector element,
            xj(l), given the template vector element, xi(k), based on a 1st order Markov model estimated on the full length of the original data sequence.
            This ensures that elementwise pairs which are unlikely to occur together are weighted to have a larger sim_distance
            and pairs which are likely to occur together are weighted to have a smaller sim_distance.

            .max(axis=2) takes the max sim_distance for each xi-xj comparison array and relates to the degree of mismatch between the template vector and the xmi vectors.

        The final sum produces the number of vectors in xmi whose max sim_distance from xi is less than r.

    Otherwise:
         The similarity distance between two vector elements is the reverse Kronecker delta
        times the complement of the conditional probability of the elementwise pair.
            Reverse Kronecker delta, d(x,y):
                if x==y, then d(x,y) = 0;
                if x!=y, then d(x,y) = 1


    Parameters
    ----------
    xmi : LIST CONTAINING ALL VECTORS OF LENGTH m DERIVED FROM THE DATA SEQUENCE
    xi_xj_prob : DICT OF NORMALIZED FREQUENCIES OF 2-TUPLE VECTORS FROM ORIGINAL DATA SEQUENCE
    m : LENGTH OF VECTORS IN xmi


    Returns
    -------
    TOTAL MATCHES WHOSE MAX SIMILARITY DISTANCE IS <= r FOR ALL xi IN xmi

    """
    if xixjprob_df:

        #z is the number of vectors in xmi
        z = len(xmi)

        #create 3d array of xmi vectors where each xi in xmi is its own 1xm subarray
        xi_matrix = np.stack([xmi], axis=2).reshape((z,1,m))

        #dif is a 3D array containing the pairwise reverse kronecker delta between xi and xmi for all xi.
        dif = np.invert(xi_matrix==xmi).astype(int)

        #swap axes so that z by z arrays in the right orientation are made from broadcasting the arrays
        xi_mat_vert = np.swapaxes(xi_matrix, 1,2)
        xmi_ar_vert = np.swapaxes(xmi, 0,1)

        xmi_bc = np.broadcast_arrays(xi_mat_vert, xmi_ar_vert)

        #create a vertical (z,) array of all xi(k) in order of (xi-xj) comparisons
        xik_arrlist = np.split(xmi_bc[1], z, axis=2)
        xik_arr = np.vstack(xik_arrlist)

        #create vertical array of all xj(l) in order of (xi-xj) comparisons
        xjl_arrlist = np.split(xmi_bc[0], z, axis=2)
        xjl_arr = np.vstack(xjl_arrlist)

        #pair each xi(k) and xi(l) together in sequence to produce (z,2) array
        x_pairs = np.concatenate((xik_arr, xjl_arr), axis=2).reshape(z**2*m, 2)
        x_pairs_df = pd.DataFrame(x_pairs, columns=['xi_k', 'xj_l'])

        #merge the x_pairs dataframe with the pairwise conditional probabilities dataframe so that each xi(k),xj(l) pair is matched with its associated probability
        p_x_pairs_df = x_pairs_df.merge(xixjprob_df, left_on=['xi_k','xj_l'], right_on=['xi_k','xj_l'], how='left')

        #get shape of difference matrix
        nd=dif.shape

        #Transform series of P(xj(l)|xi(k)) into numpy array with same shape as "dif"
        p_x_pairs = p_x_pairs_df['trans_p'].to_numpy()
        p_x_pairs = p_x_pairs.reshape(nd)

        #take the complement of p_x_pairs so that xixj transitions with low probability have a greater distance and
        #transitions with high probability have a lesser distance
        p_x_pairs = 1-p_x_pairs

        #get the max normalized similarity distance of all xi from all xmi
        #sim_dist = np.multiply(dif, p_x_pairs).max(axis=2)
        #return np.sum(sim_dist <= r)

        #multiple dif and p_x_pairs to get the weighted similarity distance for each xi,xj pair.
        #sim_dist is a (z,1,m) matrix
        sim_dist= np.multiply(dif, p_x_pairs).max(axis=2)
        return sim_dist
    else:
        #z is the number of m-vectors in xmi
        z = len(xmi)

        #create 3d array of xmi vectors where each xi in xmi is its own 1xm subarray
        xi_matrix = np.stack([xmi], axis=2).reshape((z,1,m))

        #dif is a 3D array containing the pairwise inverse kronecker delta between xi and xmi for all xi.
        dif = np.invert(xi_matrix==xmi).astype(int)

        #get the similarity difference as the number of xi(k),xj(l) pairs that do not match for each xi,xj pair
        sim_dist = dif.sum(axis=2)
        return sim_dist


def transition_prob(data):
    """
    Generates a pandas dataframe containing the conditional transition probabilities of x given x-1.

    Parameters
    ----------
    data : INPUT DATA SEQUENCE

    Returns
    -------
    xixjprob_df : pandas dataframe of xi(k),xj(l) probabilities; column[0]= xi_k value, column[1]= xj_l value, column[2]= P(xj(l)|xi(k)).

    """

    xi_xj = list(windowed(data, 2))
    xi, xn = np.unique(data[:-1], return_counts=True)
    xi_counts = dict(zip(xi, xn))
    tuples = list(product(xi, repeat=2))
    xi_xj_counts = dict.fromkeys(tuples, 0)
    #print(xi_xj_counts)
    for i in xi_xj:
        xi_xj_counts[i] = xi_xj_counts.get(i, 0) +1

    #Dict of the normalized frequency ie proportion of each 2-tuple vector
    xi_xj_prob = {}
    for k, v in xi_xj_counts.items():
        x_sum = xi_counts[k[0]]
        xi_xj_prob[k] = v/x_sum

    #turn xi_xj_prob dictionary into a list of tuples
    xi_xj_probtups = list(sliced(list(collapse(xi_xj_prob.items())), 3))
    #print(xi_xj_probtups)

    #create DataFrame of xi_xj probabilities from xi_xj_probtups
    xixjprob_df = pd.DataFrame(xi_xj_probtups, columns=['xi_k', 'xj_l', 'trans_p'])

    return xixjprob_df

def vector_matches(data, m, r, xixjprob_df=None, sampen=False, apen=False):
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

    matches : integer representing the number of matching vectors length(m) within a distance r contained within the full sequence

    """
    xmi = np.array([data[i:i+m] for i in range(len(data)-m+1)])
    if xixjprob_df:
        sim_dist=sim_distance(xmi, m, xixjprob_df=xixjprob_df)
    else:
        sim_dist=sim_distance(xmi, m)
    if sampen:
        matches = np.sum(sim_dist<=r) - len(xmi)
    elif apen:
        matches= np.sum(sim_dist<=r, axis=1)
    return matches

def apen(data, m, r, sim_dist=None):
    A = vector_matches(data[:-1], m, r, apen=True)
    k=m+1
    B = vector_matches(data, k, r, apen=True)



def sampen(data, m, r, sim_dist=None, xixjprob_df=None):
    """
    This is a function to measure the sample entropy of a given string.

    Parameters
    ----------
    data : RAW DATA SEQUENCE

    m : THE LENGTH OF THE TEMPLATE SEQUENCE

    r : THE DISTANCE FROM THE TEMPLATE SEQUENCE THAT A POSSIBLE SEQUENCE IS CONSIDERED A MATCH

    xi_xj_prob : DICT OF 1ST ORDER MARKOV TRANSITION PROBABILITIES

    Returns
    -------
    sampen, pos, mat

    sampen : floating point representing the Sample Entropy of the sequence

    pos : integer representing number of possible matches, length m

    mat : integer representing number of matches, length m+1
    """

    if xixjprob_df:
        #get number of possible matches, store in variable 'pos'
        #use data sequence with ultimate value removed so that number of m-length vectors equals the number of m+1-length vectors.

        pos = vector_matches(data[:-1], m, r, xixjprob_df, sampen=True)
        #print(pos)

        #get number of matches, store in variable 'mat'
        k = m+1
        mat = vector_matches(data, k, r, xixjprob_df, sampen=True)
        #print(mat)
    else:
        #get number of possible matches, store in variable 'pos'
        #use data sequence with ultimate value removed so that number of m-length vectors equals the number of m+1-length vectors.

        pos = vector_matches(data[:-1], m, r, sampen=True)
        #print(pos)

        #get number of matches, store in variable 'mat'
        k = m+1
        mat = vector_matches(data, k, r, sampen=True)
        #print(mat)
    if mat == 0:
        print("The data set is unique, there were no matches.")
        sampen = None
        return sampen, pos, mat
    if pos == 0:
        print("The data set is unique, there were no possible matches.")
        sampen = None
        return sampen, pos, mat
    else:
        ratio = mat/pos
        sampen = -log(ratio)
        print(sampen, pos, mat)
        return sampen, pos, mat

def course_grain(raw_data, t):
    """
    Creates a course-grained sequence from the input sequence from non-overlapping windows of size t. Uses the mode as the representative statistic of the t-windows.

    Parameters
    ----------
    data : INPUT SEQUENCE AS LIST OR ARRAY OBJECT.
    t : THE WINDOW SIZE FOR THE COURSE-GRAINING PROCEDURE.

    Returns
    -------
    Course-grained data sequence.

    """
    data = copy.deepcopy(raw_data.tolist())
    if t == 1:
        return raw_data
    else:
        #create set of non-overlapping windows of size t.
        t_win = list(windowed(data, t, fillvalue=None, step=t))


        #initiate empty list of new course-grained sequence
        cg_seq = []
        for t in t_win:
            if None in t:
                continue
            else:
                #create list of (mode, count) for each t_win where mode is the first symbol in the sequence with the highest frequency.
                #Counter() used because in the case of multiple modes it returns the value that occurs first in the sequence,
                #rather than the value that is the least
                mode = Counter(t).most_common(1)
                cg_seq.append(mode[0][0])
        return cg_seq

def multiscale(raw_data, m, r, t):
    """
    Estimates the categorical multiscale sample entropy of the input data sequence.

    Parameters
    ----------
    data : RAW DATA SEQUENCE OF INTEGERS REPRESENTING CODED CATEGORIES.

    m : LENGTH OF THE TEMPLATE SEQUENCE.

    r : THE DISTANCE FROM THE TEMPLATE SEQUENCE THAT A POSSIBLE SEQUENCE IS CONSIDERED A MATCH.
            r SHOULD BE AT LEAST EQUAL TO THE 1 MINUS THE CONDITIONAL PROBABILITY OF THE MOST LIKELY 1ST ORDER MARKOV PAIR

    t : LIST OF WINDOW SIZES TO BE USED FOR THE COURSE-GRAINING PROCEDURE.

    Returns
    -------
    Pandas dataframe of t values and corresponding sample entropy estimates.

    """
    data = copy.deepcopy(raw_data)
    #get dict of 1st order markov transition probabilities
    xixjprob_df = transition_prob(data)
    mse = []
    sampen_est, pos, mat = sampen(data, m, r, xixjprob_df)
    mse.append([1, sampen_est, pos, mat])
    for u in range(t):
        cg_seq = course_grain(raw_data, u)
        sampen_est, pos, mat = sampen(cg_seq, m, r, xixjprob_df)
        mse.append([u,sampen_est, pos, mat])
    return pd.DataFrame(mse, columns=['t', 'sampen_est', 'possibles', 'matches'])

if __name__ == "__main__":
    m=2
    r=0.1
    N=1000
    t= [2,3,4]
    data = np.random.randint(1,10, size=N)
    print(data)
    #data = [7, 5, 2, 8, 2, 3, 2, 3, 5, 2]
    xi_xj_prob = transition_prob(data)
    mat = vector_matches(data, m, r, xi_xj_prob)
    print(mat)
    #data_minus_1 = data[:-1]
    #matches = vector_matches(data_minus_1, m, r)
    #print("Possible vector matches:", matches)

