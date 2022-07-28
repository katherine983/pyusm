# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:00:00 2022

@author: Wuestney
"""
import numpy as np

class CGR:
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

def ngon_coords(verts):
    """
    Takes number of desired vertices and outputs the x, y coordinates of a regular
    n-gon with that number of vertices.

    Parameters
    ----------
    verts : INT
        NUMBER OF DESIRED VERTICES.

    Returns
    -------

    vert_array : 2D NUMPY ARRAY
        N x 2 NUMPY ARRAY, WHERE N=verts. EACH ROW IS AN X,Y COORDINATE FOR ONE
        THE VERTICES OF THE N-GON.

    """
    radians=[]
    for k in range(verts):
        rad = (2*np.pi*k)/verts
        radians.append(rad)
    x_vals = np.cos(radians)
    y_vals =np.sin(radians)
    vert_array = np.column_stack((x_vals, y_vals))
    return vert_array