# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:34:26 2022

@author: Wuestney
"""
import numpy as np
import os, glob
import pytest

#@pytest.fixture(scope="class")
@pytest.fixture()
def example_dna_seq_short():
    # example short dna sequence with all 4 base pairs in list form
    seq = list('CCCAGCTACTCAGGAGGCCGAAATGGGAGGATCCCTTGAGCTCAGGAGGA')
    alphabet = ['A', 'C', 'G', 'T']
    J = [1, 1, 1, 0, 2, 1, 3, 0, 1, 3, 1, 0, 2, 2, 0, 2, 2, 1, 1, 2, 0, 0, 0, 3, 2, 2, 2, 0, 2, 2, 0, 3, 1, 1, 1, 3, 3, 2, 0, 2, 1, 3, 1, 0, 2, 2, 0, 2, 2, 0]
    X = [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    X = np.array(X)
    # USM coords for seq as called by >usm_make.usm_make(a, A=list('ACGT'), seed='centroid')
    # prior to refactoring
    USM_fw = [[0.25, 0.75, 0.25, 0.25],[0.125, 0.875, 0.125, 0.125],[0.0625, 0.9375, 0.0625, 0.0625],[0.53125, 0.46875, 0.03125, 0.03125],[0.265625, 0.234375, 0.515625, 0.015625],[0.1328125, 0.6171875, 0.2578125, 0.0078125],[0.06640625, 0.30859375, 0.12890625, 0.50390625],[0.533203125, 0.154296875, 0.064453125, 0.251953125],[0.2666015625, 0.5771484375, 0.0322265625, 0.1259765625],[0.13330078125, 0.28857421875, 0.01611328125, 0.56298828125],[0.066650390625, 0.644287109375, 0.008056640625, 0.281494140625],[0.5333251953125, 0.3221435546875, 0.0040283203125, 0.1407470703125],[0.26666259765625, 0.16107177734375, 0.50201416015625, 0.07037353515625],[0.133331298828125, 0.080535888671875, 0.751007080078125, 0.035186767578125],[0.5666656494140625,0.0402679443359375,0.3755035400390625,0.0175933837890625],[0.28333282470703125,0.02013397216796875,0.6877517700195312,0.00879669189453125],[0.14166641235351562,0.010066986083984375,0.8438758850097656,0.004398345947265625],[0.07083320617675781,0.5050334930419922,0.4219379425048828,0.0021991729736328125],[0.035416603088378906,0.7525167465209961,0.2109689712524414,0.0010995864868164062],[0.017708301544189453,0.37625837326049805,0.6054844856262207,0.0005497932434082031],[0.5088541507720947,0.18812918663024902,0.30274224281311035,0.00027489662170410156],[0.7544270753860474,0.09406459331512451,0.15137112140655518,0.00013744831085205078],[0.8772135376930237,0.047032296657562256,0.07568556070327759,6.872415542602539e-05],[0.43860676884651184,0.023516148328781128,0.037842780351638794,0.500034362077713],[0.21930338442325592,0.011758074164390564,0.5189213901758194,0.2500171810388565],[0.10965169221162796,0.005879037082195282,0.7594606950879097,0.12500859051942825],[0.05482584610581398,0.002939518541097641,0.8797303475439548,0.06250429525971413],[0.527412923052907,0.0014697592705488205,0.4398651737719774,0.03125214762985706],[0.2637064615264535,0.0007348796352744102,0.7199325868859887,0.01562607381492853],[0.13185323076322675,0.0003674398176372051,0.8599662934429944,0.007813036907464266],[0.5659266153816134,0.00018371990881860256,0.4299831467214972,0.003906518453732133],[0.2829633076908067,9.185995440930128e-05,0.2149915733607486,0.5019532592268661],[0.14148165384540334,0.5000459299772047,0.1074957866803743,0.25097662961343303],[0.07074082692270167,0.7500229649886023,0.05374789334018715,0.12548831480671652],[0.035370413461350836,0.8750114824943012,0.026873946670093574,0.06274415740335826],[0.017685206730675418,0.4375057412471506,0.013436973335046787,0.5313720787016791],[0.008842603365337709,0.2187528706235753,0.006718486667523393,0.7656860393508396],[0.0044213016826688545,0.10937643531178765,0.5033592433337617,0.3828430196754198],[0.5022106508413344,0.05468821765589382,0.25167962166688085,0.1914215098377099],[0.2511053254206672,0.02734410882794691,0.6258398108334404,0.09571075491885495],[0.1255526627103336,0.5136720544139735,0.3129199054167202,0.04785537745942747],[0.0627763313551668,0.25683602720698673,0.1564599527083601,0.5239276887297137],[0.0313881656775834,0.6284180136034934,0.07822997635418005,0.26196384436485687],[0.5156940828387917,0.3142090068017467,0.039114988177090027,0.13098192218242843],[0.25784704141939585,0.15710450340087334,0.519557494088545,0.06549096109121422],[0.12892352070969793,0.07855225170043667,0.7597787470442725,0.03274548054560711],[0.564461760354849,0.039276125850218335,0.37988937352213625,0.016372740272803554],[0.2822308801774245,0.019638062925109168,0.6899446867610681,0.008186370136401777],[0.14111544008871224,0.009819031462554584,0.8449723433805341,0.0040931850682008886],[0.5705577200443561,0.004909515731277292,0.42248617169026703,0.0020465925341004443]]
    data = {'seq' : seq, 'alphabet' : alphabet, 'X' : X, 'centroid_USM_fw' : USM_fw}
    return data

@pytest.fixture(scope="class")
def example_dna_seq_only():
    # example short dna sequence with all 4 base pairs in list form
    seq = list('CCCAGCTACTCAGGAGGCCGAAATGGGAGGATCCCTTGAGCTCAGGAGGA')
    return seq

@pytest.fixture(scope="class")
def dna_alphabet_complete_sorted():
    alphabet = ['A', 'C', 'G', 'T']
    return alphabet

@pytest.fixture
def dna_alphabet_complete_unsorted():
    alphabet = ['C', 'A', 'T', 'G']
    return alphabet

@pytest.fixture(scope='class')
def dna_usm_coord_dict(dna_alphabet_complete_sorted):
    coord_dict = dict(zip(dna_alphabet_complete_sorted, np.identity(4)))
    return coord_dict

@pytest.fixture
def example_intseq_alph3():
    seq = [1, 1, 1, 3, 1, 2, 2, 3, 1, 2]
    alphabet = [1, 2, 3]
    coord_dict = dict(zip(alphabet, np.identity(3)))
    data = {'seq' : seq, 'alphabet' : alphabet, 'coord_dict' : coord_dict}
    return data

@pytest.fixture
def dna_cgr_coord_dict():
    coord_dict = A = {'A':(0, 0), 'C':(0, 1), 'G':(1, 1), 'T':(1, 0)}
    return coord_dict

@pytest.fixture(scope='session')
def test_data_dir():
    return os.path.join(os.path.dirname(__file__), 'test_data')

@pytest.fixture(scope='session')
def expected_out_dir():
    return os.path.join(os.path.dirname(__file__), 'expected_output')

@pytest.fixture
def humhbb(test_data_dir, dna_cgr_coord_dict):
    # get humhbb sequence sourced from https://www.ncbi.nlm.nih.gov/nuccore/U01317.1
    datafile = os.path.join(test_data_dir, 'humhbb.txt')
    with open(datafile, 'r') as fhand:
        seq = list(fhand.read())
    data = {'seq':seq, 'coord_dict':dna_cgr_coord_dict}
    return data



# # USM coords for 'a' as called by >usm_make.usm_make(a, A=list('ACGT'), seed='centroid')
# # prior to refactoring
# USMa_fw =[[0.25, 0.75, 0.25, 0.25],[0.125, 0.875, 0.125, 0.125],
#               [0.0625, 0.9375, 0.0625, 0.0625],[0.53125, 0.46875, 0.03125, 0.03125],
# [0.265625, 0.234375, 0.515625, 0.015625],
# [0.1328125, 0.6171875, 0.2578125, 0.0078125],
# [0.06640625, 0.30859375, 0.12890625, 0.50390625],
# [0.533203125, 0.154296875, 0.064453125, 0.251953125],
# [0.2666015625, 0.5771484375, 0.0322265625, 0.1259765625],
# [0.13330078125, 0.28857421875, 0.01611328125, 0.56298828125],
# [0.066650390625, 0.644287109375, 0.008056640625, 0.281494140625],
# [0.5333251953125, 0.3221435546875, 0.0040283203125, 0.1407470703125],
# [0.26666259765625, 0.16107177734375, 0.50201416015625, 0.07037353515625],
# [0.133331298828125, 0.080535888671875, 0.751007080078125, 0.035186767578125],
# [0.5666656494140625,
#  0.0402679443359375,
#  0.3755035400390625,
#  0.0175933837890625],
# [0.28333282470703125,
#  0.02013397216796875,
#  0.6877517700195312,
#  0.00879669189453125],
# [0.14166641235351562,
#  0.010066986083984375,
#  0.8438758850097656,
#  0.004398345947265625],
# [0.07083320617675781,
#  0.5050334930419922,
#  0.4219379425048828,
#  0.0021991729736328125],
# [0.035416603088378906,
#  0.7525167465209961,
#  0.2109689712524414,
#  0.0010995864868164062],
# [0.017708301544189453,
#  0.37625837326049805,
#  0.6054844856262207,
#  0.0005497932434082031],
# [0.5088541507720947,
#  0.18812918663024902,
#  0.30274224281311035,
#  0.00027489662170410156],
# [0.7544270753860474,
#  0.09406459331512451,
#  0.15137112140655518,
#  0.00013744831085205078],
# [0.8772135376930237,
#  0.047032296657562256,
#  0.07568556070327759,
#  6.872415542602539e-05],
# [0.43860676884651184,
#  0.023516148328781128,
#  0.037842780351638794,
#  0.500034362077713],
# [0.21930338442325592,
#  0.011758074164390564,
#  0.5189213901758194,
#  0.2500171810388565],
# [0.10965169221162796,
#  0.005879037082195282,
#  0.7594606950879097,
#  0.12500859051942825],
# [0.05482584610581398,
#  0.002939518541097641,
#  0.8797303475439548,
#  0.06250429525971413],
# [0.527412923052907,
#  0.0014697592705488205,
#  0.4398651737719774,
#  0.03125214762985706],
# [0.2637064615264535,
#  0.0007348796352744102,
#  0.7199325868859887,
#  0.01562607381492853],
# [0.13185323076322675,
#  0.0003674398176372051,
#  0.8599662934429944,
#  0.007813036907464266],
# [0.5659266153816134,
#  0.00018371990881860256,
#  0.4299831467214972,
#  0.003906518453732133],
# [0.2829633076908067,
#  9.185995440930128e-05,
#  0.2149915733607486,
#  0.5019532592268661],
# [0.14148165384540334,
#  0.5000459299772047,
#  0.1074957866803743,
#  0.25097662961343303],
# [0.07074082692270167,
#  0.7500229649886023,
#  0.05374789334018715,
#  0.12548831480671652],
# [0.035370413461350836,
#  0.8750114824943012,
#  0.026873946670093574,
#  0.06274415740335826],
# [0.017685206730675418,
#  0.4375057412471506,
#  0.013436973335046787,
#  0.5313720787016791],
# [0.008842603365337709,
#  0.2187528706235753,
#  0.006718486667523393,
#  0.7656860393508396],
# [0.0044213016826688545,
#  0.10937643531178765,
#  0.5033592433337617,
#  0.3828430196754198],
# [0.5022106508413344,
#  0.05468821765589382,
#  0.25167962166688085,
#  0.1914215098377099],
# [0.2511053254206672,
#  0.02734410882794691,
#  0.6258398108334404,
#  0.09571075491885495],
# [0.1255526627103336,
#  0.5136720544139735,
#  0.3129199054167202,
#  0.04785537745942747],
# [0.0627763313551668,
#  0.25683602720698673,
#  0.1564599527083601,
#  0.5239276887297137],
# [0.0313881656775834,
#  0.6284180136034934,
#  0.07822997635418005,
#  0.26196384436485687],
# [0.5156940828387917,
#  0.3142090068017467,
#  0.039114988177090027,
#  0.13098192218242843],
# [0.25784704141939585,
#  0.15710450340087334,
#  0.519557494088545,
#  0.06549096109121422],
# [0.12892352070969793,
#  0.07855225170043667,
#  0.7597787470442725,
#  0.03274548054560711],
# [0.564461760354849,
#  0.039276125850218335,
#  0.37988937352213625,
#  0.016372740272803554],
# [0.2822308801774245,
#  0.019638062925109168,
#  0.6899446867610681,
#  0.008186370136401777],
# [0.14111544008871224,
#  0.009819031462554584,
#  0.8449723433805341,
#  0.0040931850682008886],
# [0.5705577200443561,
#  0.004909515731277292,
#  0.42248617169026703,
#  0.0020465925341004443]]