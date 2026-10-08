This packet contains six experimental ground-state rotational constants reproduced in the supporting information to Piccardo, Penocchio, Puzzarini, Biczysko and Barone, DOI [10.1021/jp511432m](https://doi.org/10.1021/jp511432m) (2015). The original publisher PDF, its verbatim text extraction, publisher metadata, extraction locations and SHA-256 pins are retained. The SI explicitly labels Table 1 as experimental `(B0_beta)EXP`, with all data in MHz.

| Species label in the source | A / MHz | B / MHz | C / MHz |
| --- | ---: | ---: | ---: |
| H2O, parent | 835839.10(13) | 435347.353(27) | 278139.826(57) |
| H2CO, parent | 281970.5578(61) | 38833.98715(31) | 34004.24349(31) |

The parentheses are quoted exactly. Their implied last-digit magnitudes are recorded separately in `published-value-provenance.json`; they are not asserted to be standard uncertainties because the confidence convention and fit covariance were not retrieved. All machine-readable `standard_uncertainty` fields consequently remain null.

The original measurement references printed in the SI are Matsushima et al., *Journal of Molecular Structure* **352**, 371–378 (1995), for H2O, and Brünken et al., *Physical Chemistry Chemical Physics* **5**, 1515–1518 (2003), for H2CO. Their full article bytes were not retrieved. This packet is a peer-reviewed secondary reproduction of experimental values, not a claim to have inspected the original measurements.

The requested comparison identities explicitly name 16O/1H/1H and 12C/16O/1H/1H, but the reproduced parent rows do not print their isotope mass numbers. Electronic-state labels, the Hamiltonian reduction, axis representation and fit covariance remain unconfirmed. These omissions are retained in the review and provenance; no independent scientific acceptance or human curation is claimed.

For a descriptive comparison, pass `references.json`, `automated-review.json`, `source-locations.json` and `derived-texts.json` to `reference-compare`, using the exact manifest/review digests in `pins.json` and an actual publication bundle with a typed ground-state B0 result. The strict importer checks every original source digest, selects the exact printed numeric token, and replays `pdftotext -layout -enc UTF-8` with the recorded version before trusting extracted PDF text. The packet does not need live publisher access after installation.

An equilibrium Be result is a different observable. The default HF/STO-3G water worked example does not produce B0; its equilibrium constants must not be compared to these experimental ground-state constants by relabelling them or introducing synthetic vibrational corrections. An unavailable B0 stage remains unavailable.

The publisher PDF and derived text carry [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), with the original attribution retained. Read `LICENSE-NOTICE.md`; these third-party scientific files are not project MIT material. `retrieval-proofs.json` records actual HTTPS acquisition, original publisher MD5 agreement and byte-identical PDF text replay.
