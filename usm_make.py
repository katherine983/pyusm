# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:38:17 2020

@author: wuest
"""

import numpy as np
from timeit import Timer
import copy

class CGR():
    """
    Creates a container for CGR coordinates with attributes related to the alphabet of the sequence,
    coordinates of alphabet, specifying forward or backward generated coordinates and form of CGR (either USM or 2D, default is USM).
    """
    __slots__= ['fw', 'bw', 'coord_dict', 'form']
    def __init__(self, forward, backward=None, coord_dict=None, form='USM'):
        self.fw=forward
        self.bw=backward
        self.coord_dict = coord_dict
        self.form=form
        
    def coord_dictMake(self, alphabet, vertices):
        #creates the coord_dict from the list of alphabet and vertices in correct order
        self.coord_dict=dict(zip(alphabet, vertices))

def usm_make(sequence, A=None, seed='centroid'):
    """
    Calculates USM coordinates of a categorical sequence of arbitrary alphabet size.

    Parameters
    ----------
    sequence : LIST OR ARRAY TYPE;
        CATEGORICAL SEQUENCE OF DATA
    d : LIST, optional;
        LIST CONTAINING ALL POSSIBLE SYMBOLS OF THE ALPHABET OF THE SEQUENCE. The default is None. 
        If default, will take alphabet as set of unique characters in seq. 
    seed : STRING, default 'rand';
        INDICATES THE PROCEDURE FOR SEEDING THE USM IFS

    Returns
    -------
    List of ndarrays containing the forward SM coordinates for each symbol in seq

    """
    seq=copy.deepcopy(sequence)
    N=len(seq)
    #determine number of unique symbols in seq
    uu, J = np.unique(seq, return_inverse=True)
    
    
    
    #determine the dimension, d, of the unit hypercube whose edges correspond to each uu
    if A:
        A.sort()
        print("A", A)
        d=len(A)
        assert d >= len(uu), "Unique sequence units greater than alphabet. List of unique sequence units: {}".format(uu)
        #to ensure consistency in coordinate locations for sequences with the same generating alphabet but are missing 1 or more alphabet symbols.
        ix = []
        for i in range(len(uu)):
            assert uu[i] in A, "Unrecognized symbol in sequence. List of unique sequence units: {}".format(uu)
            print("uu[i]", uu[i])
            #get the index in A of each symbol in uu
            y = A.index(uu[i])
            print("Index of uu[i] in A", y)
            ix.append(y)
        #create sparse identity matrix in d dimensions
        Y=np.identity(d)
        #create matrix with only rows in ix so that the index of uu used in J is associated with the correct coordinate in Y.
        Y=Y[ix]
    else:
        #get number of unique symbols in seq
        d=len(uu)
        A=uu
        #create sparse identity matrix for uu
        Y=np.identity(d)
    X=Y[J]
    X_rev = copy.deepcopy(np.flip(X, 0))
    print("X", X)
    if seed=='centroid':
        f=[np.repeat(0.5, d)]
        b=[np.repeat(0.5, d)]
        for i in range(N):
            u=0.5*f[i]+0.5*X[i]
            v=0.5*b[i]+0.5*X_rev[i]
            f.append(u)
            b.append(v)
    elif seed=='circular':
        c=[np.repeat(0.5, d)]
        for i in range(N):
            u=0.5*c[i]+0.5*X[i]
            c.append(u)
        b=[c[-2]]
        for i in range(N):
            u=0.5*b[i]+0.5*X_rev[i]
            b.append(u)
        f=[b[-2]]
        for i in range(N):
            u=0.5*f[i]+0.5*X[i]
            f.append(u)
    elif seed=='rand':
        c=np.random(0,1,d)
        f=[c]
        b=[c]
        for i in range(N):
            u=0.5*f[i]+0.5*X[i]
            v=0.5*b[i]+0.5*X_rev[i]
            f.append(u)
            b.append(v)
    #print(c)
    fl = [arr.tolist() for arr in f]
    bl = [arr.tolist() for arr in b]
    USM=CGR(forward=fl[1:], backward=bl[1:], form='USM')
    vert_coords=list(map(tuple, np.identity(d)))
    USM.coord_dictMake(A,vert_coords)
    return USM

def cgr2d(seq, A=None):
    """
    Calculates 2D CGR coordinates according to the method for estimating a bijective contraction ratio proposed by Almeida & Vinga (2009).
    First estimates the contraction ratio to use based on the size of the alphabet of the input sequence.

    Parameters
    ----------
    seq : LIST-LIKE OBJECT
        CONTAINS THE SEQUENCE TO BE GRAPHED.
    A : LIST CONTAINING ALL POSSIBLE SYMBOLS OF THE ALPHABET OF THE SEQUENCE. The default is None. 
        If default, will take alphabet as set of unique characters in seq.

    Returns
    -------
    CGR object containing the coordinate array and alphabet-dict.

    """
    N=len(seq)
    #determine number of unique symbols in seq and inverse sequence of uu indices that recreate seq
    uu, J = np.unique(seq, return_inverse=True)
    
    if A:
        A.sort()
        print("A", A)
        d=len(A)
        assert d >= len(uu), "Unique sequence units greater than alphabet. List of unique sequence units: {}".format(uu)
        ix = []
        for i in range(len(uu)):
            assert uu[i] in A, "Unrecognized symbol in sequence. List of unique sequence units: {}".format(uu)
            print("uu[i]", uu[i])
            y = A.index(uu[i])
            print("Index of uu[{}] in A".format(i), y)
            ix.append(y)
        radians=[]
        for k in range(d):
            rad = (2*np.pi*k)/d
            radians.append(rad)
        x_vals= np.cos(radians)
        y_vals=np.sin(radians)
        Y= np.column_stack((x_vals, y_vals))
        Y=Y[ix]
    else:
        #get number of unique symbols in seq
        d=len(uu)
        A=uu
        radians=[]
        for k in range(d):
            rad = (2*np.pi*k)/d
            radians.append(rad)
        x_vals= np.cos(radians)
        y_vals=np.sin(radians)
        Y= np.column_stack((x_vals, y_vals))
    X=Y[J]
    k=np.rint(((d+2)/4))
    #s is the contraction ratio
    s=(2*np.cos((np.pi*(0.5-(k/d))))-2*np.cos((np.pi*(0.5-(1/(2*d)))))*np.cos(((2*k-1)*(np.pi/(2*d))))*(1+(np.tan(((2*k-1)*(np.pi/(2*d))))/np.tan((np.pi-(d+2*k-2)*(np.pi/(2*d)))))))/(2*np.cos((np.pi*(0.5-(k/d)))))
    c=[np.asarray((0,0))]
    for i in range(N):
        ci1=c[i]+s*(X[i]-c[i])
        c.append(ci1)
    cgr2d=CGR(forward=c[1:], form='2dCGR')
    verts=np.column_stack((x_vals, y_vals))
    vert_coords=list(map(tuple, verts))
    cgr2d.coord_dictMake(A, vert_coords)
    return cgr2d
        
def usm_density(c, L):
    """
    Calculates the renyi quadratic entropy of USM map coordinates

    Parameters
    ----------
    c : LIST OF FORWARD USM MAP COORDINATE ARRAYS
    L : L-TUPLE, NUMBER OF USM DIVISIONS IN EACH COORDINATE
    alpha : PARAMETER FOR RENYI ENTROPY

    Returns
    -------
    vector of L-tuple counts

    """
    
    d=2**L   #d is the number of bijective contractions for tuples length L
    nbin = d**2  #total number of subquadrants used for density calculation
    t=np.floor(d*c[L-1:])
    print(t)
    B, J = np.unique(t, axis=0, return_inverse=True)
    print("B and J",B, J)
    n=np.histogram(J, bins=nbin)
    return n

if __name__ == "__main__":
    import pandas as pd
    #import matplotlib.pyplot as plt
    
    data=np.random.randint(8, size=10000)
    coords=cgr2d(data)
    octogon = pd.DataFrame(np.asarray(coords.fw), columns=['x', 'y'])
    octogon.plot.scatter(x='x', y='y', s=2)
    #print(data)
    #c=usm_make(data)
    #n= usm_density(c, 3)
    #%%
    """
    #print(c.fw[0:20])
    forward=c.fw
    print(type(forward))
    #print(c.bw)
    """
    #%%
    """
    from entropy_renyi import renyi2usm as renyi
    sig2v = np.genfromtxt('sig2.csv', delimiter=',')
    R2 = renyi(cgr_coords=forward, sig2v=sig2v)
    #print(R2)
    t = Timer(lambda: usm_make(data))
    print(t.timeit(number=1))
    """