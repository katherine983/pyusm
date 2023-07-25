# Brief Overview of Package Use
Python package implementing and expanding on the universal sequence mapping tools created by S. Vinga and J. Almeida and referenced in [1](#1) [2](#2) [3](#3). 

The USM class is a basic class for holding computed USM/CGR coordinates. 
There are two class methods for instantiating a USM class:
    Use the method USM.make_usm() to get and store the USM coordinates of a data sequence.
    Use the method USM.cgr2d() to get 2d CGR coordinates using the formula derived in [3](#3)

Example:
```python
import pyusm

data = ['a', 'b', 'c']
#produces an instance of the USM class with form 'USM'
datausm = pyusm.USM.make_usm(data)
#produces an instance of the USM class with form '2DCGR'
datacgr = pyusm.USM.cgr2d(data)
```

Feed a list of USM forward coordinates to usm_entropy.renyi2usm() to compute the continuous quadratic renyi entropy of the USM map.
    Note: This function does not currently accept full instances of the USM class. When calling you must provide the attribute containing the coordinates to compute entropy for. 

Example:
```python
from pyusm import usm_entropy

#computes the quadratic renyi entropy values from the forward USM map coordinates in datausm.fw
rn2dict = usm_entropy.renyi2usm(datausm.fw)
```
See {doc}`demo_usm` for an in-depth description of the theory behind CGR and USM maps and how USM coordinates are calculated. 

See {doc}`demo_usm_entropy` for an explanation and proof of the implementation of the continuous quadratic renyi entropy formula for USM coordinates.


#### Bibliography
<a class="id">1</a>
<div class="csl-entry">[1] Vinga, S., &#38; Almeida, J. S. (2004). Rényi continuous entropy of DNA sequences. <i>Journal of Theoretical Biology</i>, <i>231</i>(3), 377–388. https://doi.org/10.1016/j.jtbi.2004.06.030</div>
<a class="id">2</a>
<div class="csl-entry">[2] Almeida, J. S., &#38; Vinga, S. (2002). Universal sequence map (USM) of arbitrary discrete sequences. <i>BMC Bioinformatics</i>, <i>3</i>. https://doi.org/10.1186/1471-2105-3-6</div>
<a class="id">3</a>
<div class="csl-entry">[3] Almeida, J. S., &#38; Vinga, S. (2009). Biological sequences as pictures: A generic two dimensional solution for iterated maps. <i>BMC Bioinformatics</i>, <i>10</i>(100), 1–7. https://doi.org/10.1186/1471-2105-10-100</div>
