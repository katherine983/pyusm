# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:14:53 2020

@author: wuest
"""


import numpy as np
from usm_make import usm_make
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import datetime
import pathlib
from Bio import SeqIO
import copy
#from Bio import Seq
#from Bio import SeqRecord

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

#renyi4d matches exact formula of the renyi entropy algorithm of a 4d usm used by Vinga & Almeida 2004.
def renyi4d(cgr, sig2v, refseq=None, filesave=False):
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
    
def renyi2usm(cgr_coords, sig2v, refseq=None, filesave=False):
    """
    Calculates Renyi quadratic entropy of a set of USM forward coordinates

    Parameters
    ----------
    cgr : ARRAY LIKE CONTAINING USM FORWARD COORDINATES AS ROWS
    sig2v : VECTOR WITH VARIANCES, SIG2, TO USE WITH PARZEN METHOD
    refseq : STRING, optional
        NAME OF SEQUENCE. The default is None.
    filesave : BOOLEAN, optional
        OPTION TO SAVE RESULTS TO FILE. The default is False.

    Returns
    -------
    Dictionary containing renyi quadratic entropy of the USM for each sig2 value.

    """
    #convert cgr is ndarray
    cgr=copy.deepcopy(cgr_coords)
    cgr = np.asarray(cgr)
    #n is sequence length and d is the size of the alphabet or dimension of the USM
    n, d = cgr.shape
    #get d(i,j) the pairwise squared euclidean distance between all sample USM points ai and aj.
    dij = pdist(cgr, 'sqeuclidean')
    r2usm_dict = {}
    for i in range(len(sig2v)):
        sig2 = sig2v[i]
        sig = np.sqrt(sig2)
        G = 2*np.sum(np.exp((-1/(4*sig2))*dij))+n
        V = (1/((n**2)*((2*sig*(np.sqrt(np.pi)))**d)))*G
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
    
if __name__ == "__main__":
    """
    char_val = ["a","b","c","d","e","f","g","h","i"]
    data = random.choices(char_val, k=35000)
    data_string = ""
    data_string = data_string.join(data)
    """
    data_folder = pathlib.Path(r"C:\\Users\\wuest\\Google Drive\\Dissertation Writings\\Code\\renyi1v1\\seq")
    fname = "Es.seq"
    file_to_open = data_folder/fname
    with open(file_to_open, 'r') as fhand:
        seq = SeqIO.read(fhand, "fasta")
        data_string = str(seq.seq)
        #print(data_string, type(data_string))
        data= list(data_string)
        #print(data)
        cgr = usm_make(data, A=['A','C','G','T'])
        #print("cgr", cgr)
        cgr = np.asarray(cgr)
        #np.savetxt("cgr_MC0-py-2.csv", cgr, delimiter=",")
        sig2v = np.genfromtxt('sig2.csv', delimiter=',')
        symbols, freqs = np.unique(data, return_counts=True)
        print("renyi2usm")
        print("----------------------------------------------------------")
        print(renyi2usm(cgr, sig2v, refseq=fname))
        #print("renyi4d")
        #print("----------------------------------------------------------")
        #print(renyi4d(cgr, sig2v, refseq=fname))
    
