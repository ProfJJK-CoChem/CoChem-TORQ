# Source-bound revDSD component reconciliation

The genuine Santra–Sylvetsky–Martin 2019 supporting information now verifies its **D3(BJ)** sample inputs. The official DFTD4 source separately verifies the target method's dispersion parameters. The original-2019 **D4 electronic and orbital-generating GKS definitions remain unresolved**; DH0/DH1 and named-method activation remain blocked.

The [machine-readable component ledger](revdsd_component_recipe_ledger.json) records this agent source review, the exact source assignments and the code/SRS/matrix hashes observed before this documentation update. A second agent independently checked the assignments and exact excerpt binding. This is neither a human curator attestation nor independent-engine qualification. The [source evidence ledger](revdsd_source_evidence.json) preserves the earlier denied-download receipts and native-2021 comparisons as historical observations. The [scientific protocol](../wiki/RevDSD_Spectroscopy_Protocol.md) continues to require separate method and derivative acceptance.

## Primary supporting information

Santra, Sylvetsky and Martin, *J. Phys. Chem. A* **123** (2019), 5129–5143, DOI [10.1021/acs.jpca.9b03157](https://doi.org/10.1021/acs.jpca.9b03157); SI DOI [10.1021/acs.jpca.9b03157.s001](https://doi.org/10.1021/acs.jpca.9b03157.s001). The actual PDF was acquired on 2026-10-08 through the [official publisher download](https://ndownloader.figshare.com/files/15437975), matching the [publisher metadata](https://api.figshare.com/v2/articles/8258924): **787,293 bytes**, MD5 `efe4fd1654c9eab5a5f4e9536d433c7f`, SHA-256 `e386997549ceb0655248f53a4ccfffdbce1656398aabd9a378bf28e016910c7a`. The publisher metadata declares [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

The PDF and extracted text remain in an external review archive, identified in the JSON by logical artifact names and hashes; they are not files distributed at repository-relative paths. `pdftotext -layout`, version 25.03.0, produced the 28-page extraction, SHA-256 `211ae0dcf8d6aedeb2ba36c95ead480931ae108a6ddf2285f1aa01bd0e4e963d`; repeated extraction was byte-identical. One-based text-line locators count newline delimiters without treating form feeds as additional lines.

Printed page **S23**, PDF page **23**, text lines **1298–1314** supplies the ORCA D3(BJ) sample. Printed **S21** supplies the Gaussian counterpart. The exact S23 excerpt is reproducible from the extracted text's half-open Unicode character range `[97048:97460]`, encoded as UTF-8 without whitespace changes: **412 bytes**, SHA-256 `62b726a236090a9c1ab69f2ce0c9369f97711492aeb9a46a63a9ae64343d0b6e`. Its CO geometry is illustrative input, not an optimized equilibrium geometry.

| Component | Source-bound observation | Accepted original-2019 D4 definition |
|---|---|---|
| Orbital-generating GKS functional | D3 sample: `X_PBE`, `C_P86`, `ScalHFX=0.69`, `ScalDFX=0.31`, `ScalGGAC=ScalLDAC=0.4296`. The publisher abstract distinguishes xrevDSD's full-correlation initial orbitals. | Unresolved. Neither the D3 sample nor that abstract supplies the target D4 GKS equation. |
| Final electronic energy | D3 sample: literal `PS=0.5785`, `PT=0.0799`, `ScalMP2C=1`. Separate secondary D4 candidates are `cX=0.69`, PBE exchange `0.31`, `cC=0.4210`, `cOS=0.5922`, `cSS=0.0636`. | All accepted target coefficients remain null until primary reconciliation. Cross-engine OS/SS normalization also requires matched native component validation. |
| D3 dispersion | Primary sample: `s6=0.4377`, `s8=0`, `a1=0`, `a2=5.5`. | No transfer of D3-fitted electronic or dispersion coefficients to D4. |
| D4 dispersion | Official `revdsdpbep86`, `bj-eeq-atm`: `s6=0.5132`, `s8=0`, `s9=1`, `a1=0.44`, `a2=3.60`, `alp=16`, damping `bj`, many-body selector `approx-atm`. | Separately verified parameter component; electronic identity and derivative qualification remain incomplete. |
| P86 implementation | The primary sample's literal selector is `C_P86`; installed LibXC variants yield distinct values in the retained numerical observation. | Exact P86/FT/LDA correspondence remains unresolved. |
| Frozen core | Tables S13/S17, printed S14/S18, distinguish frozen-core/noFC variants. | Exact target shell/orbital selections and cross-engine convention remain unresolved. |

## Official D4 component

The [official parameter artifact](https://github.com/dftd4/dftd4/blob/d5f891808c2145becdcce742f67ecc7e711067d7/assets/parameters.toml) is pinned to commit `d5f891808c2145becdcce742f67ecc7e711067d7`, SHA-256 `8254bfc673e763f7589b9be20506438b202a59ce795dc175d6ea9ba623e23e1f`. The entry is at lines 442–444, the default selector at line 2 and BJ-EEQ-ATM defaults at line 6. The [upstream Python implementation](https://github.com/dftd4/dftd4/blob/d5f891808c2145becdcce742f67ecc7e711067d7/python/dftd4/parameters.py), SHA-256 `e75c3244aa06d1771796831592aac3d9b3d873a689e667bbeebe7309d8b93e57`, resolves the entry by copying defaults and overlaying method values in `_get_params`.

This commit-bound observation is separate from the protocol's historical DFTD4 3.7.0 source receipt; no release tag or version equivalence is inferred. Parameters are retained in the upstream API convention, without unverified unit conversions. The full versioned D4 kernel, EEQ/ATM behavior and response require their own review and validation.

The ledger verifies **21 literal artifact assignments**: 13 selectors/scales from the primary D3 sample and eight values/selectors from the official D4 component. It verifies **zero primary target-D4 electronic coefficients**. These counts measure source facts, not completed SRS requirements or scientific acceptance gates.

## Implementation consequences and remaining evidence

At the source hashes recorded by this review, TORQ's research model separates `orbital_xc` from `energy_xc` and permits explicitly named unqualified custom models. Its exact revDSD constructor raises; the registry entry remains documented, nonrunnable and exposes no products. Analytic HF/PT2 response is a separate narrow milestone. Full moving-grid GKS/OS–SS orbital-response derivatives and higher derivatives remain unqualified. Future accepted capability tuples must bind the complete verified recipe, dispersion definition, method/basis/reference/property/derivative, engine version and hardware.

The next prerequisite is the original main-article method definition and D4 coefficient table, or authentic primary-author material explicitly resolving the original-2019 D4 orbital and final-energy definitions. The current genuine main-article probe returned HTTP 403. Matched independent native components, derivative acceptance and spectroscopy/observable calibration then remain separate gates. The acquired D3-only SI and native-2021 output do not close those gates. This documentation integration changes no engine code, launches no native calculation and performs no student deployment.
