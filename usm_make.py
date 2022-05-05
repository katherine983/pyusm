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

    Class
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

def usm_make(sequence, A=None, seed='centroid', deep_copy=True):
    """
    Calculates USM coordinates of a discrete-valued sequence of arbitrary alphabet size.

    Parameters
    ----------
    sequence : LIST OR ARRAY TYPE;
        CATEGORICAL SEQUENCE OF DATA
    A : LIST, optional;
        LIST CONTAINING ALL POSSIBLE SYMBOLS OF THE ALPHABET OF THE SEQUENCE. The default is None.
        If default, will take alphabet as set of unique characters in seq.
    seed : STRING, default 'rand';
        INDICATES THE PROCEDURE FOR SEEDING THE USM IFS
    deep_copy : BOOL
        IF TRUE (DEFAULT) WILL USE A DEEP COPY OF THE SEQUENCE TO MAKE THE USM.

    Returns
    -------
    An instance of a CGR object. CGR.fw is a list of ndarrays containing the
    forward SM coordinates for each symbol in seq

    """
    if deep_copy is True:
        seq=copy.deepcopy(sequence)
    else:
        seq = sequence
    N=len(seq)
    #determine number of unique symbols in seq
    #uu is an ndarray of the unique values in seq, sorted.
    #J is an ndarray size = len(seq) containing the index values of uu that could recreate seq
    uu, J = np.unique(seq, return_inverse=True)

    #determine the dimension, d, of the unit hypercube whose edges correspond to each uu
    if A:
        A.sort()
        #print("A", A)
        d=len(A)
        assert d >= len(uu), "Unique sequence units greater than alphabet. List of unique sequence units: {}".format(uu)
        #to ensure consistency in coordinate locations for sequences with the same generating alphabet but are missing 1 or more alphabet symbols.
        ix = []
        for i in range(len(uu)):
            assert uu[i] in A, "Unrecognized symbol in sequence. List of unique sequence units: {}".format(uu)
            #print("uu[i]", uu[i])
            #get the index in A of each symbol in uu
            y = A.index(uu[i])
            #print("Index of uu[i] in A", y)
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
    #print("X", X)
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

def ngon_coords(verts):
    radians=[]
    for k in range(verts):
        rad = (2*np.pi*k)/verts
        radians.append(rad)
    x_vals = np.cos(radians)
    y_vals =np.sin(radians)
    return x_vals, y_vals

def cgr2d(seq, A=None, vert_dict=False):
    """
    Calculates 2D CGR coordinates according to the method for estimating a
    bijective contraction ratio proposed by Almeida & Vinga (2009).
    First estimates the contraction ratio to use based on the size of the
    alphabet of the input sequence.

    Parameters
    ----------
    seq : LIST-LIKE OBJECT
        CONTAINS THE SEQUENCE TO BE GRAPHED.
    A : LIST OR DICT
        IF LIST, SHOULD CONTAIN ALL POSSIBLE SYMBOLS OF THE ALPHABET OF THE
        SEQUENCE. FUNCTION WILL THEN CALCULATE VERTEX COORDINATES AUTOMATICALLY.
        IF DICT, SHOULD HAVE ONE ENTRY FOR EACH POSSIBLE SYMBOL OF
        THE ALPHABET OF THE SEQUENCE WHERE THE ENTRY KEY IS THE SYMBOL AND THE
        ENTRY VALUE IS THE COORDINATE TO BE ASSOCIATED WITH THAT SYMBOL.
        The default is None.
        If default, will take alphabet as set of unique characters in seq and
        calculate vertex coordinates automatically.

    Returns
    -------
    CGR object containing the coordinate array and alphabet-dict.

    References
    ----------
    Almeida, J. S., &#38; Vinga, S. (2009). Biological sequences as pictures: A generic two dimensional solution for iterated maps.
        BMC Bioinformatics, 10(100), 1–7. https://doi.org/10.1186/1471-2105-10-100

    """
    N=len(seq)
    #uu is the set of unique symbols in seq and J inverse sequence of uu indices that recreate seq
    uu, J = np.unique(seq, return_inverse=True)

    if A:
        if isinstance(A, list):
            A.sort()
            #print("A", A)
            #d is dimension of alphabet
            d=len(A)
            assert d >= len(uu), "Unique sequence units greater than alphabet. List of unique sequence units: {}".format(uu)
            ix = []
            for i in range(len(uu)):
                assert uu[i] in A, "Unrecognized symbol in sequence. List of unique sequence units: {}".format(uu)
                #print("uu[i]", uu[i])
                y = A.index(uu[i])
                #print("Index of uu[{}] in A".format(i), y)
                ix.append(y)
            #get coordinates for vertices of an equilateral d-gon
            """
            radians=[]
            for k in range(d):
                rad = (2*np.pi*k)/d
                radians.append(rad)
            x_vals= np.cos(radians)
            y_vals=np.sin(radians)
            """
            x_vals, y_vals = ngon_coords(d)
            Y= np.column_stack((x_vals, y_vals))
            Y=Y[ix]
        elif isinstance(A, dict):
            a, vrts = zip(*A.items())
            d = len(a)
            assert d >= len(uu), "Unique sequence units greater than alphabet. List of unique sequence units: {}".format(uu)
            ix = []
            for i in range(len(uu)):
                assert uu[i] in a, "Unrecognized symbol in sequence. List of unique sequence units: {}".format(uu)
                #print("uu[i]", uu[i])
                y = a.index(uu[i])
                #print("Index of uu[{}] in A".format(i), y)
                ix.append(y)
            x_vals, y_vals = zip(*vrts)
            Y= np.column_stack((x_vals, y_vals))
            Y=Y[ix]
    else:
        #get number of unique symbols in seq
        d=len(uu)
        A=uu
        #get coordinates for vertices of an equilateral d-gon
        """
        radians=[]
        for k in range(d):
            rad = (2*np.pi*k)/d
            radians.append(rad)
        x_vals= np.cos(radians)
        y_vals=np.sin(radians)
        """
        x_vals, y_vals = ngon_coords(d)
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
    Calculates the subquadrant density of USM map coordinates

    Parameters
    ----------
    c : LIST OF FORWARD USM MAP COORDINATE ARRAYS
    L : L-TUPLE, NUMBER OF USM DIVISIONS IN EACH COORDINATE
    alpha : PARAMETER FOR RENYI ENTROPY

    Returns
    -------
    vector of L-tuple counts

    """
    #d is the number of bijective contractions for tuples length L
    d=2**L
    #total number of subquadrants used for density calculation
    nbin = d**2
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