# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 14:00:41 2022

@author: Wuestney
"""
import os
import numpy as np
import pytest
import pyusm

def array(*args):
    return np.array(*args)

def string(*args):
    return ''.join(*args)

@pytest.fixture(scope='class', params=[list, tuple, array, string])
def dna_alph_iterables(request, dna_alphabet_complete_sorted):
    return request.param(dna_alphabet_complete_sorted)

@pytest.fixture(scope='class', params=[list, tuple, array])
def dna_coords_iterables(request, dna_usm_coord_dict):
    coord_dict = dna_usm_coord_dict
    for key in coord_dict.keys():
        coord = coord_dict[key]
        coord = request.param(coord)
        coord_dict[key] = coord
    return coord_dict

@pytest.fixture(scope='class', params=[list, tuple, array, string])
def dna_seq_iterables(request, example_dna_seq_only):
    return request.param(example_dna_seq_only)



def test_always_passes():
    # baseline passing test
    assert True

@pytest.mark.xfail
def test_always_fails():
    # baseline failing test
    assert False

def test_module_import():
    assert all([obj in dir(pyusm) for obj in ['USM', 'check_alphabet','get_alphabet_coords']])
    return

def dicts_equalish(dict1, dict2):
    keys = all([dict1.keys() == dict2.keys()])
    vals = np.array_equal(list(dict1.values()), list(dict2.values()))
    #vals = [list(dict1.values()) == list(dict2.values())]
    return keys, vals


class TestUserSeq:
    @pytest.mark.make_usm
    def test_usm_seq_list(self, dna_usm_coord_dict, example_dna_seq_short):
        # test that USM is made properly when seq is given as a list
        USM = pyusm.USM.make_usm(example_dna_seq_short['seq'], example_dna_seq_short['alphabet'], seed='centroid', deep_copy=True)
        assert all(dicts_equalish(USM.coord_dict, dna_usm_coord_dict))
        assert np.allclose(np.array(USM.fw), np.array(example_dna_seq_short['centroid_USM_fw']))
        assert np.array_equal(np.array(USM.fw), np.array(example_dna_seq_short['centroid_USM_fw']))

    @pytest.mark.get_alphabet_coords
    def test_usm_seq_noalph(self, dna_usm_coord_dict, example_dna_seq_short):
        # test that USM is made properly when seq is given as a list, no alphabet
        USM = pyusm.USM.make_usm(example_dna_seq_short['seq'], seed='centroid', deep_copy=True)
        assert all(dicts_equalish(USM.coord_dict, dna_usm_coord_dict))
        assert np.allclose(np.array(USM.fw), np.array(example_dna_seq_short['centroid_USM_fw']))
        assert np.array_equal(np.array(USM.fw), np.array(example_dna_seq_short['centroid_USM_fw']))

    @pytest.mark.make_usm
    def test_usm_seq_iterables(self, dna_seq_iterables, example_dna_seq_short):
        USM = pyusm.USM.make_usm(dna_seq_iterables, A=None, seed='centroid', deep_copy=True)
        assert np.allclose(np.array(USM.fw), np.array(example_dna_seq_short['centroid_USM_fw']))
        assert np.array_equal(np.array(USM.fw), np.array(example_dna_seq_short['centroid_USM_fw']))


#@pytest.mark.usefixtures("example_dna_seq_short", "")
class TestUserAlph:
    # class to handle varying scenarios of user provided alphabet/coord_dict to the 'A' argument

    @pytest.mark.get_alphabet_coords
    def test_alph_list(self, dna_usm_coord_dict, example_dna_seq_short):
        # test that USM is made properly when alph is given as a list
        X, coord_dict = pyusm.get_alphabet_coords(example_dna_seq_short['seq'], example_dna_seq_short['alphabet'])
        assert all(dicts_equalish(coord_dict, dna_usm_coord_dict))

    @pytest.mark.get_alphabet_coords
    def test_alph_dict(self, dna_usm_coord_dict, example_dna_seq_short):
        X, coord_dict = pyusm.get_alphabet_coords(example_dna_seq_short['seq'], dna_usm_coord_dict)
        #assert (X == example_dna_seq_short['X']).all()
        # arrays should be exactly equal
        assert np.array_equal(X, np.array(example_dna_seq_short['X']))

    @pytest.mark.get_alphabet_coords
    def test_alph_iterables(self, dna_alph_iterables, example_dna_seq_short):
        # test that X is made properly when alph is given in different iterable forms
        X, coord_dict = pyusm.get_alphabet_coords(example_dna_seq_short['seq'], alphabet=dna_alph_iterables)
        #assert (X == example_dna_seq_short['X']).all()
        # arrays should be exactly equal
        assert np.array_equal(X, np.array(example_dna_seq_short['X']))

    @pytest.mark.check_alphabet
    def test_alph_unsorted(self, dna_alphabet_complete_sorted, dna_alphabet_complete_unsorted):
        uu = np.array(dna_alphabet_complete_sorted)
        ix = [0, 1, 2, 3]
        assert pyusm.check_alphabet(uu, dna_alphabet_complete_unsorted) == ix

    @pytest.mark.check_alphabet
    def test_uu_unexpected_char(self, dna_alphabet_complete_sorted):
        with pytest.raises(AssertionError, match=r"Unrecognized symbol .*"):
            # mock list of unique characters in a sequence containing a character not found in user-defined alphabet
            uu = ['A', 'C', 'T', 'U']
            pyusm.check_alphabet(uu, dna_alphabet_complete_sorted)
            #test should fail assertion within pyusm.check_alphabet()
            return

    @pytest.mark.check_alphabet
    def test_uu_longer(self, dna_alphabet_complete_sorted):
        with pytest.raises(AssertionError, match=r".* units greater .*"):
            # mock list of unique characters in a sequence containing more characters than found in user-defined alphabet
            uu = ['A', 'C', 'G', 'T', 'U']
            pyusm.check_alphabet(uu, dna_alphabet_complete_sorted)
            #test should fail assertion within pyusm.check_alphabet()
            return

    @pytest.mark.check_alphabet
    def test_uu_shorter(self, dna_alphabet_complete_sorted):
        uu = np.array(dna_alphabet_complete_sorted)
        uu = np.delete(uu, 3)
        pyusm.check_alphabet(uu, dna_alphabet_complete_sorted)
        return

    @pytest.mark.get_alphabet_coords
    def test_user_coord_dict(self, dna_coords_iterables, example_dna_seq_short, dna_usm_coord_dict):
        # test different data types of vertex coords
        user_coord_dict = dict(dna_coords_iterables)
        X, coord_dict = pyusm.get_alphabet_coords(example_dna_seq_short['seq'], alphabet=user_coord_dict)
        #assert (X == example_dna_seq_short['X']).all()
        # arrays should be exactly equal
        assert np.array_equal(X, np.array(example_dna_seq_short['X']))
        assert all(dicts_equalish(coord_dict, dna_usm_coord_dict))

@pytest.mark.make_usm
def test_usm_deepcopy(example_dna_seq_short):
    # test that the output of USM.make_usm() is same for deep_copy == True and False
    USM_deepcopy = pyusm.USM.make_usm(example_dna_seq_short['seq'], example_dna_seq_short['alphabet'], seed='centroid', deep_copy=True)
    USM_copy = pyusm.USM.make_usm(example_dna_seq_short['seq'], example_dna_seq_short['alphabet'], seed='centroid', deep_copy=False)
    assert np.allclose(np.array(USM_deepcopy.fw), np.array(USM_copy.fw))
    assert np.array_equal(np.array(USM_deepcopy.fw), np.array(USM_copy.fw))

@pytest.mark.cgr2d
def test_cgr2d_humhbb(humhbb):
    # utilize for regression testing at some point
    hbbcgr = pyusm.USM.cgr2d(humhbb['seq'], humhbb['coord_dict'])
    pass

@pytest.mark.cgr2d
def test_cgr2d_GGA_suffix_dist(example_dna_seq_short, dna_cgr_coord_dict):
    a = example_dna_seq_short['seq']
    b = b = list('GCAGAGTCCAGGGTCCGAGAAGGGGA')
    cgr_a = pyusm.USM.cgr2d(a, dna_cgr_coord_dict)
    cgr_b = pyusm.USM.cgr2d(b, dna_cgr_coord_dict)
    # k is number of characters in shared suffix of the two sequences
    k = 3
    q = 2**(-k)
    # calc euclidean distance between last coordinate of each cgr array
    dist = np.linalg.norm(np.array(cgr_a.fw[-1]) - np.array(cgr_b.fw[-1]))
    assert dist < q


def test_usm_mc0(test_data_dir, expected_out_dir, dna_usm_coord_dict):
    seqfile = os.path.join(test_data_dir, 'MC0.txt')
    with open(seqfile, 'r') as fhand:
        seq = list(fhand.read())
    mc0 = pyusm.USM.make_usm(seq, A=dna_usm_coord_dict)
    #get original coords generated from matlab code
    coordfile = os.path.join(expected_out_dir, 'og_MC0_coords.csv')
    og_mc0 = np.genfromtxt(coordfile, delimiter=',')
    assert np.allclose(np.array(mc0.fw), np.array(og_mc0))
    assert np.array_equal(np.array(mc0.fw), np.array(og_mc0))




# coords=cgr2d(data)
# octogon = pd.DataFrame(np.asarray(coords.fw), columns=['x', 'y'])
# octogon.plot.scatter(x='x', y='y', s=2)
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
#data=np.random.randint(8, size=10000)