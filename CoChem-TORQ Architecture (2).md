# **CoChem-TORQ Architecture: Part 3 — Tensor Extraction & Thermodynamic Partitioning (Stages 4.1 \- 5.0)**

## **1\. Architectural Intent**

To translate raw quantum mechanical wavefunctions and optimized geometries into precise microwave spectral predictions, CoChem-TORQ enforces strict physical constant anchoring (CODATA 2022\) and robust axis diagonalization protocols.

## **2\. Module Responsibilities**

* **TorqTensorExtractor (cochem\_tensor\_extractor.py)**: Centers molecules to their Center of Mass (COM), computes the 3x3 inertia tensor, and diagonalizes it to derive principal moments of inertia and rotational constants (![][image1] in MHz).  
* **Cartesian Protection Layer**: Intercepts linear molecules via singular value decomposition (SVD) rank analysis (s\[1\] \< 1e-4), preventing ZeroDivisionError singularities during rotational constant conversions.  
* **TorqSpcatBridge (cochem\_spcat\_bridge.py)**: Evaluates Rigid-Rotor Harmonic-Oscillator (RRHO) partition functions (![][image2]) and assigns rotational symmetry numbers (![][image3]). Detects floppy low-frequency Large Amplitude Motions (LAM ![][image4]).

## **3\. Data Flow & Inter-Process Communication**

\[Optimized Coordinates\] ──► \[Stage 4.1: COM Center & Inertia Tensor Diagonalization\]  
                                     │  
                                     ▼  
                            \[Cartesian Linearity Trap & Rotational Constants (MHz)\]  
                                     │  
                                     ▼  
                            \[Stage 5.0: CODATA 2022 Partition Functions (Q\_rot x Q\_vib)\]  
                                     │  
                                     ▼  
                            \[Pickett SPCAT Seed Synthesis (.var / .int)\]

## **4\. Key Dependencies & Failure Points**

* **Dependencies**: numpy, scipy, molsym (optional point group wrapper).  
* **Failure Points**:  
  * Division by zero on linear diatoms if rotational constant parsing lacks protection.  
  * Symmetry number over-weighting if point-group heuristics default to ![][image5].  
* **Validation Checkpoints**: Cryptographic SHA-256 coordinate hashing and immutable CODATA constant enforcement to guarantee multi-year reproducibility.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEMAAAAaCAYAAADsS+FMAAAC3ElEQVR4Xu2XPYjUUBSFZ1kFBUFERlhmJm9+RBBBhUFEUFRQsPEHLbWy0cLKQkGblcXKbhpB1sJCBNdOtlAsrES0FWx1kQUREQQFF3b1nJkbeR6TTJLNQop8cGFy7rv3vdy83Lyp1SoqKoqg2WxudM6tqF5m2u32Eaz5HmzQ7/fXmzyB6/P/DMwKEryD/VY9C4i/GQTBAvPAlmGfeG3aoulvNS4LKMBh5HhvuW5AmqRBfwM7Y/N+kLD0YLG7LPmqikGQa4vl+u/pQDtKHxY9rb40IHbOcr9QH4H+jX6s4br6UoPFPUaSz0UUAzl2Mk+n09kT4Zuym5nHnBvUn0S3291ssXP8rX4C3xPYCopxTH2paDQaW7Gw50hym5PV6/VNOiYLyDGIKyoWeZC+PDvDCvFDdR/4z+Yp9BA0ze0IXqxZ07EJp3RcFhD/FfZddeJG7/OS6uNAzCzXxtdZfT4YczX3rkDwT1TxMn8ziU3Y13FZsILOhte9Xm8b5jht+h1/bFosdu2+dLjpC7Dj4TXfcSvGSX9cFpz1i1ardciN+sPQwtywgcaMw4udV19R8LX44gu28FV1Yt4sc6hOkLebJz8fDuOwuy6prxCQ/CnsvhsdWELjNZ/AQx2fFpfQL9iYLf9L9SURFmPcjkWxTmDcAdXH4mKaWJ7F+lh85KEKX62m+TNt97SvL8Y84ila9UQQNBPXJG2xH1XnU/WOu5EEdtiK2848IdLPT7n62GRV87F1vVY9BAVzmP+u6okgaDcTqx5ik+q3fJ3psXHE2aeZRREX+9Pw+IyC7BXf37OHSzhCw3eOY9CYT6kP2j7keKZ6EpPhDUXdGBb5QP2B973G9SvYr6gDGfSLGiu2jPy3NM6DxVriWHX44NS5Q/IO/+sg934du+a40dlhQvWicBGvZ2nBTllQrSjsL8G06qWEDQrMqF4UKPQ1/jVQvZS4HCfHLCD/FdUqKioqKkrGH36z79cyNRQWAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGEAAAAaCAYAAACn4zKhAAAEhklEQVR4Xu1YO2hUQRTdEAXF/ydu/rObRBY/iLJoiIoIKmihiI2fYGWhhSJYKGJhRG1SSbAKQbFQQQQb01kEK1HQxkIkARUhYAhikUIliee8d8dcb3bf7qq7MfIODG/uPXfeuzN35s7MSyRixIgR499HQ0PDCudcD0pvc3Pzfsv/D0ilUvNQrrCPLPX19U3WZkYAp57DoS8Y+LNKdxG6yXQ67bTtbEVbW9ti9gd9vA2xStTVkD9A/03bVhTZbHYuHcOAd1mOgIM3yCemnJ6VQB8OST9rLUeAG0d5bfUVgSzJSQbDcgS4DvJNTU3rLTebwEHGhHpr9R7gn8lkqyzg1B6ZHe2W80gmkwtog3LXcmVEtVVY5Js0Fq2tras4w2WA865m8Odogz2x0XJlhQzuZ6s3mCN2A5YoFyRF5s3RzOEtLS1LrD4X8J4x8f+05TTAd9IO785armzA7D/Cj/JpOQ3Y1FU6CIQEYtzqoRspIQBp8Z2rYI7lNTAOdyoeBHxwiB+tqalZaDkNP0Pg5EnLlRuNjY3zdSBKCQAB+z4JwpjlLGAzLP2cZ7mygY7JDIkEbD6KXeRMKhckEBzIcdSXWz4KaDMgbZ9YzsCn3I+WyAcEayPsX6BaxcCh/gll2NpFwhURBLm4+XP1H0E62WP1hYA2W9HJp3jew/Ow5aPgJAhMNZbTgM1VsUtZLh8wJrvQ5rqXmSm0XBTQYLRQEPDiU7T5G5c1CUKn1RcC2nxmCpI94nsi4oRjAft++e5Dy2mAH3E59p9SwG8gMLutPhIY4C46yOVOGfUv4nBwWmLHRZ52qkDboyjtLty07eyuhq4b5YTY7oRzZyAP4nkc32sw9nlBX/QeoAJRFJzccXyfePwU2R9Xq+DTecp2r5HbdTcmYIYy7LZ5G+n7MT92BP2CvNqF41H0RGFD3hKfSMMg54ujwe0SF7QDpgnbwK90kjyeW1w4i/YJx3a9rHOwnZyo+H5y6jUF4UwAPEoNBC9oytf3oq5C/RL1LHowPS9ZgH4H2QLyA7zrAso6uXtwIx8Q+2BP4Z2Kbem7f1ExYINBceaxqr8iRwN7KWJqSoUb0S8DAfm+d5iAs8ucLHE8RzOZzKIp62jAfk2uAHjQJ/ixw+rzQa8AF/4h+IYypgdf/8RTgznBflHHUySPryhr5bDwFaWDHCbrZtT7WZexiUzzBeHCFPPOy3jpU0UHYO5z5gYtHRxSMo+2PD0E3JTlvwHtk6yum5pHUFZCN4y+tlBOqftUKkzlP9O0C4/CQTaA/XbURz33W5AgcEB9mZZGoOvnx4yOp61gNoj8VfaVNU6CwVxa6F5SKZg+TjuCc9an1KkKNrdUfSIRpqAOWRV9Pliov4auzdv+Fpg2tHM2HRHQv+NM0TrmTOhHWEeq2ODkbO7CIDwEX4vnI91mJqH7yH3D8rLvveQxnSkY2OQ5tpHBD7IB+raXRXQl7Qd5Aacu42XPcmxYAWwANNCuLmF+wMHBpbmCOZOATwfh6xuUa5bzkL0naX2nbFc0ZfZT62LEiBEjRowYswQ/AOHKYs7w+fORAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAbCAYAAABIpm7EAAAApUlEQVR4XmNgGAVDH6ioqPDJy8s/BeL/6FhBQeEQimKgQCVUYiaQ7gPihyC+nJxcKBCHANmSyIrNQZLGxsasSGYwQDW4IIvBJH4C8RQs4iANvujiYAmgLR5owoxQDTYookABQZCEqKgoD5q4C0gcyGREFgcBFqgEC0xAVlbWFurMYiR1CACUmAXES6HsaCD+C7QhH10dCgA5CRR8QL9woMuNgmEAADpILV3MjgILAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAF0AAAAZCAYAAABTuCK5AAADCUlEQVR4Xu1YO4hTQRTNsisoio2GuPm9JARFEJv4wVZsFvGDWAhaq4UoCOoKdouVjYUghLXQThAbsQ9YbLGFjYt1tlDYQhE2goHd9ZzkTvbm5uUHyfKic+AyM+fe+Z03k5lJLObhMYnI5XLziURir+U9xoAgCOYg+C2kW/F4fJ/1e4wRkRGdA4F9zufzp5DOYkVcQ1q3cel0eg/43xK/lUql0jYm6oia6G0G4Q+ZsBkR+oCUp1jGhyi2RUUcURV9A4KfDok5Q7/hlmGLmttpoP/z2Wz2ajfD7g1M/HhELxaL+9H4L3T63PrCgNgaYkuWd0gmkwcRswKrah51HtkPEXWMXHQ0mGejEOOe9fVCP9Hhv8J2YRXNo84F8oNcwfDhMogt0wqFwmHHozxL46pkuVQq7WI+k8mc2K7d7As78DKyU5ofFqMSnb+tS7AqD0DrHARB83BcgK2jjbccGNKc8j/tJTpF07wGD1v4lxG7yjInLHXuoIvdSF/BqsKV8XNwnHHI/4R9gX1F3GtyaOMm45CdUV0MBDVWZxUbMwimOSDYUmwEXx8Tu+3KWGXHyLkVwUmHDdRNpM8uaUwyJkI50dHHRRfD+jKG1qJB+brUzTtOeNY9qbmxg9sPHa/B3lvfKCETfif5RSlXdEy/lQ7+qPhr1qeBdgqM09veta3jCHL0WX6skO3KibStgFFD+lhhPisHZjC86HPi7yk66zMusqI7yAOGW7LjejcMMIGXIswHzQvXuK0g5hzym7BPJua+CBP6Gxv8a6I7UHQOBOklFKetvx9QtyLCtESXA47cC1X+CPu+XbP1s7OpOQ15wbKdDvHQ5rzLu907MaI74NQ/ggG9ga1xstbfDTzEIMDjmDqMwT2UCbdWMF+ilkP5D2JvuHIYguY1kcKXHce/G3Q9jmEiRdfAwJ7A6urJ3hOYxF3Gw75Jyud9x4cDf1YE5DVvA/bMxnQBr7W8abEu7QFJdX2sYwyrbFNuSjXYD+EaddxHEG49MA+1yGDHr1YeHh4eHh4eHh4e/x/+Av03IlevJKsMAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAaCAYAAABctMd+AAABMElEQVR4Xu2TMWvCUBSFBelWnAzSJOQlkK1jwD8gTs6lhY4OXTrXH+HiZMFFXF1d3R38AU5uruJkN23PxfdKehqSFzsV8sElj3PuPeG9vNRqFWWJoqgdBMGbUmqC54PRse6k+0qBsBXqEyEj1L1ovu/H0M6oHurIM4UkSXIjoagNewJ20tIvnbJXCIbeZdjzPJ89g/hhGL6wnguCu3rwkb006PlwXbfJei76OE6sM9LHWi4YeNbhr+z9GYRuJbz0dm1A8LH0djVxHDcwO2H9G9tw9OxT6z5qjBriEszSfT8wx8I6ozLuP27ZIDcc5pOEoyL2DPCWjuPcsl4YLmD4oC5XsZ7hreX3Z12wChfQNNc7kNrppxzFrxcarMOv4f+Fy8fFkd2py3VcyJp7Kirs+QKHI1IJnr83bQAAAABJRU5ErkJggg==>