# **CoChem-TORQ: Torsional Discovery & Statistical Mechanics**

## **Overview**

**CoChem-TORQ** is the heavy-lifting spectroscopic and thermodynamic engine of the pipeline. It bridges the gap between static quantum chemistry (optimized geometries) and experimental reality (spectra at specific temperatures).

TORQ calculates highly accurate partition functions (![][image1]), implements Variational Perturbation Theory (VPT2) for anharmonic frequencies, and computes the 1D-Schrödinger torsional energy levels required for accurate microwave and infrared spectroscopy.

## **Scientific & Technical Trade-offs**

* **JAX GPU Diagonalization vs. CPU:** Torsional Hamiltonians can result in massive matrices (e.g., ![][image2]). Traditional numpy.linalg.eigh on a CPU will choke. TORQ utilizes JAX (XLA compilation) to offload these ![][image3] operations to the GPU. You trade memory overhead (JAX pre-allocates VRAM) for a 100x speedup in spectral generation.  
* **PyArrow Chunking:** Generating a massive thermal catalog of transitions (e.g., up to ![][image4]) will instantly trigger an Out-Of-Memory (OOM) crash if stored in a Pandas DataFrame. TORQ streams data directly to disk via PyArrow chunking, trading slightly slower I/O for infinite RAM stability.  
* **Persistence of Calculation:** For DLPNO-CCSD(T) refinements, TORQ actively writes .gbw and .hess files to disk iteratively. If a 24-hour job crashes at hour 23, it resumes from the last step, sacrificing disk space for extreme fault tolerance.

## **Installation**

git clone \[https://github.com/CoChem/CoChem-TORQ.git\](https://github.com/CoChem/CoChem-TORQ.git)  
cd CoChem-TORQ

## **How to Run**

TORQ is highly sensitive to hardware limits. Ensure no other massive GPU jobs are running before executing the primary Hamiltonian diagonalization.

1. **Tensor Extraction (Principal Axes & Dipoles):**  
   python cochem\_tensor\_extractor.py \--target landscape.h5  
2. **High-Accuracy Refinement (VPT2/CCSD(T)):**  
   python cochem\_spcat\_bridge.py  
3. **High-Performance Catalog Compilation:**  
   python cochem\_catalog\_compiler.py \--temp 298.15

## **Key Validation Check**

Before running, check fit\_provenance.json. TORQ explicitly records the CODATA version used for physical constants to guarantee decade-long reproducibility.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGEAAAAaCAYAAACn4zKhAAAEDklEQVR4Xu2ZTUhUURTHHTQo+sLKDGec51dIEURIheUqCmpRCxe1aNMqCFoVFC1buGjX2oiCiCDsA0IKEpLaZcuisFwUkqCEm5RU1P7/N+fUmdN8vCFnZOD94DL3nnOu95x77sd7z5qamJiYmJiYAgRBcARlGeVLS0vL2lQqtRP1ecq8bTWDeB4zptbW1gNoJhDrObabm5tPeduKkk6nv9GRjo6OTTl0o9C99/JqA4tqiyyyj17X1dW1hjrEeszrKgIG76UDdMTrSFtb22ZxvtXrqgmZ5FEvVyTGyu96bMlABt/ndRaxGfDyagG+9xebYOgnJFFtXldWMOhiMeeIJOGrl5cLjPUE5aKXK9ydmKzTXp6L9vb27eJ/r9dZGJ8k4aTXlQ0MOCjOFV3hlU4CwYW5g+OimrByyIYwUVesrBDie5SFNrsaSQidw5G01+ssTU1N28S5O15XbpLJZMpOYKkJ4NFSQhJCO1zgSa8rC42NjeuNc3Veb0EgV8V2l9flo6GhYYPW0e+ZJLHe2kSFieDEMwHYHde8vhDoc0l8n/A6j5mPysBJijoobOZQFr28EPK3/yQtyjiFQP+fSMRTLy+GWUDDXmfRJ0DY3/S6EkmUFGuUJGDlPSxmE4G6/znKMP4IfhI6UajXept8wP5ElCTQhrF6eakEmZ33wcvzAuNxyX54TKA+JQ6HL2Z8o5T2UHbPcIW9gNMXpJlA+7zqYH+Ll6q2oesJMm/jn0o5zwn6jNg+JhFZl3UB6iSGBTb0spe/EcL4bFuB7RnI30j9Ovw4bPVo34V+DPPUKW0emZzDdyj91jYv+o6AAV5zUnH2bqWcr+8o+8XZe74fLq4O/qrj7BfIk5OsJm7J72qP+gADknpf1C1Pv2B/2ctNIiLBCVF7/M4Z+VmURxLnP/cdxj8O+X3pm3XMoD5v6uOIqUvqy3yQUV0kgr/fi5b0jZmrmjL87vb2BLpXKN1BdkDhYy6PHblv+owuXIWi5xld9FE3WMH3BBJkVvsvlDER1TJGlEnWra2BE/8D4xxlQ3cCkvMW8gdqhPowyqA+Rar8v2BW0+at0R4HCicW8h5t2+wH5tuMdwz1GSZK26uJnzBdzUpnZ+dGb4PTo5EyPl2qTJLJXTWwYrHRGfnD3MbTKJ+9DfX6GKrHkxCewZCto2Pi9IzYUZb3G1WlkRgnUW6znkPPb2pZj7aYm3pri2N7j7bxu8BjXOrRL+dc6AuSFvvMr0A+hW15sCazrV+qXFb+TGAuc9SXajJbe8YlbFWxMaJM59AP8/j08nTmq/IhnSddVKjPUcZLnMel71cy4tis3XYeDJTmdxkv1wvewv9NeNlqg8m6IXE+9zqCVd3kZQpjzLM4u70sJiYmJiYmpgr4DQk5VGnAPOeDAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJIAAAAZCAYAAADaDHeVAAAFHElEQVR4Xu1ZTYhcRRCeEAVFRfxZ18z+9MyusAiiyKIi6FlzUDyKHnPw4klBwVNyyFUQRGQRxIOsiN497CHoJZCrOUnASMSbQlAhhmT8vjdVsaZe1Zs3u2SRpD9o5vXX1d1V36vp16/fYFBRUVFRUVFRUTEXKysrq+vr6596XjEajZ4rpfyOMkF517f3xXg83kL/X1F+Hg6Ha75dwTk4F3z6CtWjvp1A+xso/7DA7hXfftiARqc2Nzcf8byC+op+nbHPg2hzDeXPQaLN6urq3Wg/O08b6is+/YgceMi398LS0tK9CP41DHKOg+H6C29DbGxs3I/2P3B5hHVcfwgH3nNmc4F+OxRgMA3+qIixE9iR/5LXa2trz9A3JGCxNqg/C3+/1zpjIGdtDgO4YSuY+0SZJvQE5Zi3IST2E1JtYt/e3r5zxqgHrDZSD7Uhz8VBbHh/ef9mAG4PGr7Ea0m8yUDu8UJAx+Mon2CwEQfJEgltVziR4yZMMMt1Af0fk6CXlUN9TI6/ysGHr1G/rnWx25UgFUdcvUHE3WzA34/4p2JSSCytRNLYLSexX7LcPMgcvbRBedtwjTbw9aTW8Qd9MfDpeZSzllsYMlErkSDShp+QIIe2zz2fociqF/AM+kxWJ3RV0jqu38zGgu2rnj8siO+tROqKnU8Fz2eA/elMG9yLbbFptPF+oH7J+oDr3wKf9A+6+Kqk4ABRIoH/JphQRWvxGTJ7y0OMFyI/wB0jr6KXWAQd66LnPWCzV5J9Htv289gmZP4okciH/mKu9z2fAfZXM22KJFgRbXyCst34cIf0+cvaEOJTuqeaCw7gnRT+gnHA8nSkxWfI7C2P33ciP8p/YjU3ac5YLXEijKb7q5l/XjlAEhHWR8Xy8vI9Xf76WLsQ2XM+Gb/5A+lcXYnE7YXYtbQiv0hytxA5KfxFdcDxjcOez5DZK8/AGUDkRxGxzPLdOZbnM4htk0zc7x0kiQiZfyaRGFfml/BnPJ+B9pk2RZJC5+pKJN/H2bXmWAjZAOUWTiS+1ci4DyzSL4PMXxMpGgD8eXXA8Y3Dns+Q2Vse878V+VHaj7brHWO1xOkC7J9mPz6CfNuisD4qEMtdwof++li7ENlzPhlfH22NNl2JNBwOH5Y+La3I36xH2wfqgOPpyGXPZxD7bJyG1xWimHMSAs/0p2xfXP/QMdae5zPA9pw+zqRfeLjXFzL/QpttlOOez1Cmb16hNiifiU2jzdbW1n3Wrri9rvT529oozzE93xscIEmk5qwn4Bd9/ecpazhOMcu71M8bE3I877rRF36+no3V9/VfTvJn9kQyZrNn2g/E9yiR0tj9ytEFxH0y00Yf+6oNyuPO7rL1objjAEJfDAYH0CBNJAJtV/zhI+0tp46h7Fo7hR7KFXP4iJv+hOdG0wPJmQAh0neOiw4kIy4E5ng0spUT/H0LKbG0Eik6kJTYZw4kpf+MHhZ66Gm5TBsmXRcXHUii/eWynwNJnhcY52eKtROB9dMGA/42+Df/JH2vWt6iTD8TTHhKrkfy5AK7ayP5/MEVhnbj+DPABa3Dn1/IWZsIo+mehbGEyWKSqReKvIxExdntYO5TvNbY/ScS0/e05S2K0WYgCZJoQ/5J1qlNST6RFPlsA58eFJ9DXQ4dJdjAVfQHbvrGgTa7twLKdD/1secr+gP67Y7N98jbElw+/XJd0R/yWL3xuL4twb1Mnz1KRY4yfSP7f+xRKioqKioqKioq9o1/AQOPal9CrtT4AAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADsAAAAaCAYAAAAJ1SQgAAADIklEQVR4Xu2Yu2tUQRTGbzAW4gvRZWFfs7usLkHBYmNnIcHCyiIWKrGw08JO0D/Af0DESlaChQQ0bcBiCzFNIIUISlojkkZsBAM+Nvqde8+YuV/uYy+7V4PsD4a9+50z58yZnbkzrOeN2V0YY26jbaD1W63WIbb/N9Tr9XMock2eq9XqRTz/wuMkuf0bMJhV1oYBxV7RAr1ardaR51KpdMza8X0W7fl2j4yg84wERXvd6XT2ViqV43jeFA3J6+wviJ/Y5ZNtMjiN96exD9vRvpDLBLQ+0l8n3SuXy0ejYqaC2fsgHVHgPrYh0au4oNDfot1j3aXRaJyGz5bEwJI8xXYvKEji72EDxnUHtp8orMI2AWN7VigUDrAeiwmWRGQyQSZAZ73h6iiiqP0mXJ2BTxdFntEY/j50kUJMwjawefBp2NZsNg/DtsJ6JBJAAmEGT7LNRQe66Gro8yJq8Ax8PnvBr9eVOLx6EOcu9DlXw+RM41e7YL9r/qeuj0VsXsqE+8Cxr86JaLJ11jCgG64WBfy+yqezQh6S/T0KPkKa+PnjkmUqzzIpro9FbedZDwGnJQ2augzUb0exPEgGPnPuhNgiZPmpNGmCXz6Evvi+oz3WPjPsY5H+JmEb+NjEabMCnyn17ZI+yIpYpSNjQWMtyHfkPovnB9s9soP+y0ZXTyTFYnG/Lbbdbh9kuwsGNC9++Gy6+oDFhnz0heLn9YJ9vIg25fpkBf17nCeE3QeJTkqcX5TmIkscPu9Yh7amMW+hbbE9K9gmT9LGEluEiwnOUXkR3YywpfX1jxzWvWCf+rnRltmYFcR4ibbJegg4fJSEMQe92C/pgHpsE8Qm24F1iwn2UeR9FrYVjT3Ltqwgxro01kPYMxZtg2328m0S7p9ij7vZCCZhicImyVPfF4OAOD9MzBkcwuh9GMv0qtX0iibHwwnXl9F+O85ZnSS3vWEfwSS9QTMgOeStzvpIMSN6wQwLxvCNtTzwL/BpV808kSule63MFRR6DQV/Yv1vgdx91nIFCXuY3cus5w0m+j5yP2I9d0za3XTEmGH/qRgzZsyu5Td4XQORio0GbAAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAAAaCAYAAAAUqxq7AAACGUlEQVR4Xu2XsUvDQBTGI3TTQcQabNMmpZPO6iAOjoooIoKg/0I3B/F/EEFwEHFxcKujW4eCi+Cqo4MOdhDpIg4Wrd/TF7y+3jUJSNrhfvBo8r3vXS7vmkvrOBaLxTKA+L7/imjLkL40KRQKs1JTwfy2eZ43rusOyzyB3AriDfGJ2JX5xGCQe7posVhckLk0CIJgEXEUtUDIfSBu6djzvDH2V1QPGjyD+3gKz3F8As+B6kmMMrGMzKUBrn2KWMbNrJoaBP1M5tDUJanx+ZDUyuXyhKolghvUknraRDSI5vioatlsdkT149uzpqvn2iupx4YHuJB62pgalM/nPZ5jXeZYn+Lja109e76kHgvaFGmAXC43LnNpY2oQtA2+ya5vAemo2+Pjd0M91XbpsUBhNUlxeLEE8SDHMGFqEDWAdOw55zKn6uE1dR4KeiRlLhIUtnx+M/SbQWxQhot3ZKIfmBrk/z1iVU3u/x4xmOYd5RUYroxiiQT+ySSB1R2VY5gwNahUKrmk+92bdLjA4SZd09Wzp/cmDUOTjTVFa+gG7AVueB03shk3/N9FiYWpQQTPvaFq9GJR/brfRQTXdm3wHbCpjdWYo3MMFrD2LL39IqJBxzKH84pGo/OOH7ykRf5QhOkOcUjHmMg0N+dF+voBz0UXdeGjvxo/b8VwgfGxpXrorwb0psNbCY4v4dlXPRaLxWKxWCwWSyy+AR0e7xsfkS7mAAAAAElFTkSuQmCC>