[![DOI](https://zenodo.org/badge/670381931.svg)](https://zenodo.org/badge/latestdoi/670381931)
# pyusm

Python package implementing and expanding on the universal sequence mapping [tools](<https://github.com/usm/usm.github.com>) created by S. Vinga and J. Almeida and referenced in [1](#1) [2](#2) [3](#3). 

For further documentation of the package functions including theoretical background and proofs, visit <https://katherine983.github.io/pyusm/intro.html>


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
#renyi entropy estimates output as a dictionary with kernel variance values as keys
rn2dict = usm_entropy.renyi2usm(datausm.fw)
```
```python
#generate a 2D cgr plot and animation
import pyusm

#produces an instance of the USM class with form '2DCGR'
datacgr = pyusm.USM.cgr2d(data)

#initiate figure
cgrfig = pyusm.cgr_plot(datacgr.fw, datacgr.coord_dict)
cgrfig.plot()
#animate plot figure
cgrfig.animate()
#save figure (alias for matplotlib .savefig() method)
cgrfig.savefig('cgrfig.txt', **kwargs)
```

## Testing
The testing suite built with pytest. For now the expected performance is for one failure, 26 passed, 2 xfailed. The failure should be for string input in the test_usm_seq_iterables() test. Test data include the sequence of Es promotor regions in B subtilis used in the original study [1](#1) which can also be found [here](<https://github.com/usm/usm.github.com/blob/master/entropy/Es.seq.txt>). Source for HUMHBB sequence data can be found [here](<https://www.ncbi.nlm.nih.gov/nuccore/U01317.1>).

#### Bibliography
<a name="1">1</a><div class="csl-entry">Vinga, S., &#38; Almeida, J. S. (2004). Rényi continuous entropy of DNA sequences. <i>Journal of Theoretical Biology</i>, <i>231</i>(3), 377–388. https://doi.org/10.1016/j.jtbi.2004.06.030</div>
<a name="2">2</a><div class="csl-entry">Almeida, J. S., &#38; Vinga, S. (2002). Universal sequence map (USM) of arbitrary discrete sequences. <i>BMC Bioinformatics</i>, <i>3</i>. https://doi.org/10.1186/1471-2105-3-6</div>
<a name="3">3</a><div class="csl-entry">Almeida, J. S., &#38; Vinga, S. (2009). Biological sequences as pictures: A generic two dimensional solution for iterated maps. <i>BMC Bioinformatics</i>, <i>10</i>(100), 1–7. https://doi.org/10.1186/1471-2105-10-100</div>
