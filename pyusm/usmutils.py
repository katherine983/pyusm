# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:00:00 2022

@author: Wuestney
"""
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