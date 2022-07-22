# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:14:53 2020

@author: wuest

packages: scipy1.8.1, matplotlib
"""


import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import datetime
import pathlib
import copy
#from Bio import Seq
#from Bio import SeqRecord

# set the default variance vector to use with Gaussian density kernels
sig2v_default = (1e-10,1.7783e-10,3.1623e-10,5.6234e-10,1e-09,1.7783e-09,
                  3.1623e-09,5.6234e-09,1e-08,1.7783e-08,3.1623e-08,5.6234e-08,
                  1e-07,1.7783e-07,3.1623e-07,5.6234e-07,1e-06,1.7783e-06,
                  3.1623e-06,5.6234e-06,1e-05,1.7783e-05,3.1623e-05,5.6234e-05,
                  0.0001,0.00017783,0.00031623,0.00056234,0.001,0.0017783,
                  0.0031623,0.0056234,0.01,0.017783,0.031623,0.056234,0.1,0.17783,
                  0.31623,0.56234,1,1.7783,3.1623,5.6234,10,17.783,31.623,56.234,100)
def renyi4d(cgr, sig2v=sig2v_default, refseq=None, filesave=False):
    """
    renyi4d matches exact formula of the renyi entropy algorithm of a 4d usm used by Vinga & Almeida 2004.

    Parameters
    ----------
    cgr : LIST or ARRAY-LIKE OBJECT
        ARRAY LIKE CONTAINING USM FORWARD COORDINATES AS ROWS.
    sig2v : TUPLE OR ARRAY-LIKE OBJECT, optional
        ARRAY CONTAINING VARIANCE VALUES TO USE FOR THE GAUSSIAN DENSITY KERNEL.
        The default is the tuple of values contained in sig2v_default, the same
        values used in the original paper by Vinga & Almeida (2004).
    refseq : BOOL, optional
        NAME OF SEQUENCE. The default is None.
    filesave : BOOL, optional
        OPTION TO SAVE RESULTS TO FILE. The default is False.

    Returns
    -------
    r2usm_dict : DICT
        DICT CONTAINING SIG2:ENTROPY AS KEY:VALUE PAIRS.

    """
    #convert cgr is ndarray
    cgr = np.asarray(cgr)
    #n is sequence length and d is the size of the alphabet or dimension of the USM
    n, d = cgr.shape
    #get d(i,j) the pairwise squared euclidean distance between all sample USM points ai and aj.
    dij = pdist(cgr, 'sqeuclidean')
    r2usm_dict = {}
    for i in range(len(sig2v)):
        sig2 = sig2v[i]
        G = 2*np.sum(np.exp((-1/(4*sig2))*dij))+n
        V = (1/((n**2)*16*(np.pi**2)*(sig2**2)))*G
        r2usm = np.negative(np.log(V))
        r2usm_dict[sig2] = r2usm
    sig2_list, r2usm_list = zip(*r2usm_dict.items())
    ln_sig2 = np.log(sig2_list)
    if refseq==None:
        seqname = "cgr_{}".format(datetime.datetime.now().isoformat())
    elif refseq:
        seqname = refseq
    plt.plot(ln_sig2, r2usm_list, 'bo')
    plt.title("Renyi Quadratic Entropy for {} N={} D={}".format(seqname, n, d))
    plt.xlabel('sig2')
    plt.ylabel('Renyi2 Entropy')
    plt.show()
    if filesave==True:
        fname = 'renyi2_{}'.format(seqname)
        plt.savefig(fname, format='png')

    return r2usm_dict

def renyi2usm(cgr_coords, sig2v=sig2v_default, refseq=None, Plot=True, filesave=False, deep_copy=True):
    """
    Calculates Renyi quadratic entropy of a set of USM forward coordinates

    Parameters
    ----------
    cgr : ARRAY LIKE CONTAINING USM FORWARD COORDINATES AS ROWS
    sig2v : ARRAY LIKE, optional
        ARRAY LIKE CONTAINING VARIANCES, SIG2, TO USE WITH GAUSSIAN KERNEL IN
        PARZEN WINDOW METHOD. The default is the tuple of values contained in
        sig2v_default, the same values used in the original paper by
        Vinga & Almeida (2004).
    refseq : STRING, optional
        NAME OF SEQUENCE. The default is None.
    Plot : BOOLEAN, optional
        OPTION TO PLOT ENTROPY VALUES AS A FUNCTION OF THE LOG OF THE KERNEL
        VARIANCES, SIG2V. ENTROPY VALUES ON THE Y AXIS AND LN(SIG2) VALUES
        ON THE X AXIS.
    filesave : BOOLEAN, optional
        OPTION TO SAVE RESULTS TO FILE. The default is False.
    deep_copy : BOOLEAN, optional
        IF TRUE (DEFAULT) WILL USE A DEEP COPY OF cgr_coords TO CALCULATE THE
        ENTROPY VALUES.

    Returns
    -------
    Dictionary containing renyi quadratic entropy of the USM for each sig2 value.

    """
    sig2v = np.array(sig2v)
    if deep_copy is True:
        #convert cgr is ndarray
        cgr=copy.deepcopy(cgr_coords)
        cgr = np.asarray(cgr)
    else:
        cgr = np.asarray(cgr_coords)
    #n is sequence length and d is the size of the alphabet or dimension of the USM
    n, d = cgr.shape
    #get d(i,j) the pairwise squared euclidean distance between all sample USM points ai and aj.
    dij = pdist(cgr, 'sqeuclidean')
    r2usm_dict = {}
    for i in range(len(sig2v)):
        sig2 = sig2v[i]
        G = 2 * np.sum(np.exp((-1/(4 * sig2)) * dij))+n
        V = (1/((n ** 2) * ((2 * np.sqrt(sig2) * (np.sqrt(np.pi))) ** d))) * G
        r2usm = np.negative(np.log(V))
        r2usm_dict[sig2] = r2usm
    if refseq==None:
        seqname = "cgr_{}".format(datetime.datetime.now().isoformat())
    elif refseq:
        seqname = refseq
    if Plot is True:
        sig2_list, r2usm_list = zip(*r2usm_dict.items())
        ln_sig2 = np.log(sig2_list)
        plt.plot(ln_sig2, r2usm_list, 'bo')
        plt.title("Renyi Quadratic Entropy for {} N={} D={}".format(seqname, n, d))
        plt.xlabel('sig2')
        plt.ylabel('Renyi2 Entropy')
        plt.show()
        if filesave==True:
            fname = 'renyi2_{}'.format(seqname)
            plt.savefig(fname, format='png')
    return r2usm_dict

if __name__ == "__main__":
    #from Bio import SeqIO
    from usm_make import usm_make
    """
    char_val = ["a","b","c","d","e","f","g","h","i"]
    data = random.choices(char_val, k=35000)
    data_string = ""
    data_string = data_string.join(data)
    """
    cwd = pathlib.Path.cwd()
    demo_dir = cwd / "demo_files"
    fname = "Es.txt"
    file_to_open = demo_dir / "seq" / fname
    with open(file_to_open, 'r') as fhand:
        seq = fhand.read()
        data_string = str(seq)
        #print(data_string, type(data_string))
        data= list(data_string)
        #print(data)
        cgr = usm_make(data, A=['A','C','G','T'])
        #print("cgr", cgr)
        #cgr = np.asarray(cgr)
        #np.savetxt("cgr_MC0-py-2.csv", cgr, delimiter=",")
        #sig2va = np.genfromtxt(demo_dir.joinpath('sig2.csv'), delimiter=',')
        np.set_printoptions(suppress=False)
        print(sig2v)

        print(sig2v == sig2va)
        print(sig2v.dtype)
        print(float(sig2v[0]))
#%%
        symbols, freqs = np.unique(data, return_counts=True)
        print("renyi2usm")
        print("----------------------------------------------------------")
        print(renyi2usm(cgr.fw, sig2v, refseq=fname))
        #print("renyi4d")
        #print("----------------------------------------------------------")
        #print(renyi4d(cgr, sig2v, refseq=fname))

