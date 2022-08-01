# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:07:18 2022

@author: Wuestney
"""

import pytest
import numpy as np
import os
import pyusm

# @pytest.fixture
# def usm_og_seqs(test_data_dir, og_entropy_data_names, dna_alphabet_complete_sorted):
#     file = os.path.join(test_data_dir, og_entropy_data_names)
#     with open(file, 'r') as fhand:
#         seq = list(fhand.read(fhand))
#     usm = pyusm.USM.make_usm(seq, A=dna_alphabet_complete_sorted)
#     return usm

@pytest.fixture(params=[('R1.txt', 'renyi2usm_AA.rn2'), ('rand.txt', 'renyi2usm_alea1c.rn2'),
                 ('m5.txt', 'renyi2usm_alea2c.rn2'), ('m7e.txt', 'renyi2usm_alea3c.rn2'),
                 ('m4.txt', 'renyi2usm_ATCG20.rn2'), ('R5.txt', 'renyi2usm_ATCGA.rn2'),
                 ('m3.txt', 'renyi2usm_ATCx.rn2'), ('Es.txt', 'renyi2usm_Es.seq.rn2'),
                 ('MC0.txt', 'renyi2usm_M0.rn2'), ('MC1.txt', 'renyi2usm_MC91.seq.rn2')])
def datafilepairs(request):
    return request.param

def parse_rn2_file(rn2file):
    with open(rn2file, 'r') as fhand:
        lines = fhand.readlines()
    rn2array = []
    row = 0
    for line in lines:
        if row < 2:
            row += 1
            continue
        else:
            row += 1
            line = line.rstrip().split('\t')
            #print(line)
            rn2array.append([float(line[0].strip()), float(line[1].strip())])
    return rn2array


def test_always_passes():
    # baseline passing test
    assert True

@pytest.mark.xfail
def test_always_fails():
    # baseline failing test
    assert False

def test_module_import():
    assert all([obj in dir(pyusm) for obj in ['USM', 'check_alphabet','get_alphabet_coords', 'usm_entropy']])

#@pytest.mark.parametrize("seq_input, expected", datafilepairs)
def test_renyiusm_regression(datafilepairs, test_data_dir,
                             expected_out_dir, dna_alphabet_complete_sorted):
    seqfile, rn2file = datafilepairs
    seqhand = os.path.join(test_data_dir, seqfile)
    rn2hand = os.path.join(expected_out_dir, rn2file)
    with open(seqhand, 'r') as fhand:
        seq = list(fhand.read())
    assert len(seq) == 2000
    usm = pyusm.USM.make_usm(seq, A=dna_alphabet_complete_sorted)
    #outfile = os.path.join(expected_out_dir, '{}_rn2_pkgvrsn{}'.format(seqfile.split('.')[0], pyusm.__version__))
    rn2 = pyusm.usm_entropy.renyi2usm(usm.fw, Plot=False)
    rn2array = list(rn2.items())
    rn2array.sort()
    rn2array = np.array(rn2array, dtype=np.float64)
    og_rn2array = np.array(parse_rn2_file(rn2hand), dtype=np.float64)
    assert np.allclose(rn2array, og_rn2array)