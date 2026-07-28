# **CoChem-TORQ User Manual: Part 4 — Quantum Mechanical Escalation & SPCAT Interfacing**

## **4.1 High-Fidelity ORCA Execution (Stage 3.0)**

Escalate the extracted extrema to ab initio DFT/Coupled Cluster methods using ORCA 6.1.1, utilizing automated memory backoff in the event of OOM errors:

\# %% \[Cell 4: Run ORCA Escalation with Memory Backoff\]  
orca\_runner \= TorqOrcaExecution(extrema\_file="torq\_mace\_surface.json", params\_file="torq\_run\_params.json")  
\# orca\_runner.execute\_extrema\_escalation() \# Uncomment to run actual compute  
print("ORCA Execution Engine initialized and ready for dispatch.")

## **4.2 Tensor Extraction & Rotational Constants (Stage 4.1)**

Extract principal moments of inertia and calculate rotational constants (![][image1]) in MHz with strict Cartesian protections against linear singularities:

\# %% \[Cell 5: Tensor Extraction & Provenance\]  
extractor \= TorqTensorExtractor(symbols, coordinates, point\_id="001")  
extractor.apply\_cartesian\_protections()  
constants \= extractor.diagonalize\_and\_derive\_constants()  
provenance \= extractor.export\_provenance()  
print(f"Rotational Constants (MHz) \-\> A: {constants\[0\]:.2f}, B: {constants\[1\]:.2f}, C: {constants\[2\]:.2f}")

## **4.3 Partition Functions & SPCAT Synthesis (Stage 5.0)**

Calculate CODATA 2022 thermodynamic partition functions (![][image2]) and synthesize Pickett .var and .int files:

\# %% \[Cell 6: Statistical Mechanics & SPCAT Bridge\]  
bridge \= TorqSpcatBridge(tensor\_json\_path="torq\_tensors\_001.json", orca\_out\_path="test\_orca.out", temperature\_k=298.15)  
bridge.calculate\_partition\_functions()  
bridge.generate\_spcat\_files()  


[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEMAAAAaCAYAAADsS+FMAAAC3ElEQVR4Xu2XPYjUUBSFZ1kFBUFERlhmJm9+RBBBhUFEUFRQsPEHLbWy0cLKQkGblcXKbhpB1sJCBNdOtlAsrES0FWx1kQUREQQFF3b1nJkbeR6TTJLNQop8cGFy7rv3vdy83Lyp1SoqKoqg2WxudM6tqF5m2u32Eaz5HmzQ7/fXmzyB6/P/DMwKEryD/VY9C4i/GQTBAvPAlmGfeG3aoulvNS4LKMBh5HhvuW5AmqRBfwM7Y/N+kLD0YLG7LPmqikGQa4vl+u/pQDtKHxY9rb40IHbOcr9QH4H+jX6s4br6UoPFPUaSz0UUAzl2Mk+n09kT4Zuym5nHnBvUn0S3291ssXP8rX4C3xPYCopxTH2paDQaW7Gw50hym5PV6/VNOiYLyDGIKyoWeZC+PDvDCvFDdR/4z+Yp9BA0ze0IXqxZ07EJp3RcFhD/FfZddeJG7/OS6uNAzCzXxtdZfT4YczX3rkDwT1TxMn8ziU3Y13FZsILOhte9Xm8b5jht+h1/bFosdu2+dLjpC7Dj4TXfcSvGSX9cFpz1i1ardciN+sPQwtywgcaMw4udV19R8LX44gu28FV1Yt4sc6hOkLebJz8fDuOwuy6prxCQ/CnsvhsdWELjNZ/AQx2fFpfQL9iYLf9L9SURFmPcjkWxTmDcAdXH4mKaWJ7F+lh85KEKX62m+TNt97SvL8Y84ila9UQQNBPXJG2xH1XnU/WOu5EEdtiK2848IdLPT7n62GRV87F1vVY9BAVzmP+u6okgaDcTqx5ik+q3fJ3psXHE2aeZRREX+9Pw+IyC7BXf37OHSzhCw3eOY9CYT6kP2j7keKZ6EpPhDUXdGBb5QP2B973G9SvYr6gDGfSLGiu2jPy3NM6DxVriWHX44NS5Q/IO/+sg934du+a40dlhQvWicBGvZ2nBTllQrSjsL8G06qWEDQrMqF4UKPQ1/jVQvZS4HCfHLCD/FdUqKioqKkrGH36z79cyNRQWAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGEAAAAaCAYAAACn4zKhAAAEfklEQVR4Xu1YTUhUURR+gxMY/VLZpI7zxnHApKLF9EMhrQpqUYQE/bhrY4EQtSgqKKNatAuJFiFIi1q1aKHQwoW4klq0SYgoqBBaiIiBLRSz75s5V4/HmXHGmlHjfXB5957vvPvOueee+/M8L0CAAAFWPmpra7f6vt+J8iwWi520/P+AeDxeiXKPPrLU1NTUWZ1lAYx6C4PGMfBXlOwmZDP19fW+1l2tSCaTG+kPfOxGMyTiCrS/Qz6pdcuKVCq1hoZhwDssR8DAx+S9OaNXJeBDi/i5w3IEuGmUD1ZeFkhKzjAYliPAHSJfV1e323KrCRxkTKhPVu4AflAmW3kBo47J7DhoOYdIJLKOOigvLFdCVFiBRa5JY9HQ0LCdM1wGOGc2g79GHeyJUcuVFDK4Y1ZuEBa9fkuUCrJE5lyjuYYnEolNVp4N6GdC7G+3nAb4Vuqh75TlSgbM/nP8KJ+W04BOdbmDQEggpq0cspEiAlAvtjMLwpbXwDg8L3sQ8MEv/GhVVdV6y2m4GQIj2yxXakSj0bU6EMUEgIB+lwRhwnIW0PkhflZarmSgYTJD8gI6w6KXdyZpwJHL2Mj3se6CbXUKhQSCAzmN+hbL5wPe6Zd3+yxn4JbcYUuUFIUEQS5u7lxdMBCEq27jxEVoG/r4YXUKBd49jP4G8HyJ51nL54MLApcay2lA577oxS1XDDBOCfQzZOU5AeXRxYIAmy5R528uazCsGX30WnmhwLtjXIJkj5jy8pxwLPhd2o/yynIa4Ef8LPtPsYCvRxf71jxggDtoINOdbdTHxeD0aYmOS3vBqQJBaXQ3axmcVsehfiGujrw0CuUEylN9Gy8EtEXvASoQBcGXO47zicdPabvjagg2XWfb7jVcTukLxwc6Z+DTeUWH4nO/PaopoA7q71Buaf8XhZ+5JfahdHqy5ouh6dslDDllXkmfu/HBXeCHaACM3OPLxsfMwSMkTqaB+hRnCOvobz/aTxyXD74JgEOxgeAFjfZg4hzA85uIaeNtylncRFQIw5fj1Cfv7kokJDizewzq/e7fE+q/vSIy1YHGfBZjelT9PTkq2EuRLE3zBhqOvqYej3e8WYMbFIp6NMzpMV3dQOQEdJqyBcCB34IdR6w8F3QG+JnZO4kyoQdf/8TjoHviI8pDJyfE79lDip/J9IsIWiWev5Tq0uFn7gVfXRudDyg6DbP2hd1MJyAfdL84MFB7/blZwxMIDWbWLTtkMNOQ7JqXoXKgGNY3aPHnp9bzM5mekDH5N38VJAgzqrRYHXzwRkx+c3Nt1JxzDs92MTo9k9wm7y0hXUsB4+OCIzjsbXO2OyBzan11+kE7yeWOdch7UW9mBnNFmHtrCWhsbNygjbPLEcH1EuUR69Dp0Rzao+B2cu+QTe2NyMdQurTuckL76AZSA7JuneECLlHpUxQzhKsE2k1s+5kgMCPuUG/eW0sBOrqLTgezbFiz4E2b94hschO4CmRERLVXBDCAp+HjR5QHliPAb7YyB7xTnW1yUm5lAQIECBAgQIBVgD9GHlq8t9kCJwAAAABJRU5ErkJggg==>