# pyusm
Python package implementing and expanding on the universal sequence mapping tools created by S. Vinga and J. Almeida and referenced in [1](#1) [2](#2) [3](#3). 

For in-depth explanation of the package functions including theoretical background and proofs, visit <https://katherine983.github.io/pyusm/intro.html>


Examples:
```python
import pyusm

data = ['a', 'b', 'c']
#produces an instance of the USM class with form 'USM'
datausm = pyusm.USM.make_usm(data)
#produces an instance of the USM class with form '2DCGR'
datacgr = pyusm.USM.cgr2d(data)
```

```python
from pyusm import usm_entropy

#computes the quadratic renyi entropy values from the forward USM map coordinates in datausm.fw
rn2dict = usm_entropy.renyi2usm(datausm.fw)
```

#### Bibliography
<a class="id">1</a>
<div class="csl-entry">[1] Vinga, S., &#38; Almeida, J. S. (2004). Rényi continuous entropy of DNA sequences. <i>Journal of Theoretical Biology</i>, <i>231</i>(3), 377–388. https://doi.org/10.1016/j.jtbi.2004.06.030</div>
<a class="id">2</a>
<div class="csl-entry">[2] Almeida, J. S., &#38; Vinga, S. (2002). Universal sequence map (USM) of arbitrary discrete sequences. <i>BMC Bioinformatics</i>, <i>3</i>. https://doi.org/10.1186/1471-2105-3-6</div>
<a class="id">3</a>
<div class="csl-entry">[3] Almeida, J. S., &#38; Vinga, S. (2009). Biological sequences as pictures: A generic two dimensional solution for iterated maps. <i>BMC Bioinformatics</i>, <i>10</i>(100), 1–7. https://doi.org/10.1186/1471-2105-10-100</div>
