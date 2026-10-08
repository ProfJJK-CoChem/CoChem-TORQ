This packet contains genuine historical NH3 calculation output captured in an official MolSSI QCArchive tutorial, pinned to commit `60ee49a3dc881cd64d42840a2a5f3bf31f7b049f`. It provides a reproducible computational component comparison. QCSchema is a format; this source is a captured QCArchive calculation record, rather than a QCSchema fixture or an experimental measurement.

The notebook selects molecule 2 and record 1847316. Its cell 7 captures real Psi4 B2PLYP/aug-cc-pVTZ output. The printed B2PLYP recipe is 0.53 HF + 0.47 B88 exchange, 0.73 LYP correlation, and 0.27 MP2 correlation with one frozen occupied orbital. The fixed geometry is printed in bohr. No geometry optimization, gradient, Hessian, rotational constant, revDSD implementation, or scientific profile qualification is established by this example.

The original notebook and BSD-3-Clause license were verified against the downloaded pinned GitHub archive. `source-stdout.txt` is an exact extraction of the notebook output, not rewritten data. Original public source paths appearing in these source bytes are retained to preserve source integrity. The original Molecular Science Software Institute copyright and license are in [source-license.txt](source-license.txt); redistribution does not imply endorsement. [source-metadata.json](source-metadata.json) records actual source hashes and acquisition facts. Direct live QCArchive API requests encountered a proxy CONNECT 403; no live API record is claimed.

Verify the source bytes without running an electronic structure calculation:

```bash
python scripts/run_qcarchive_component_comparison.py --verify-source-only
```

With the declared calculations dependencies, including PySCF 2.14.0 and psutil, run the bounded real comparison from the TORQ repository:

```bash
python scripts/run_qcarchive_component_comparison.py \
  --output-dir /tmp/cochem_exec_qcarchive_nh3 \
  --max-wall-seconds 300 --max-rss-mib 1024
```

The output directory must be fresh. The helper runs one Linux-affinity CPU and one numerical thread, requires 256 MiB free scratch, samples the RSS of its owned worker tree, waits for its actual original worker, and retains genuine failures. Limits are sampled process observations, rather than an operating-system memory reservation. The first calculation uses PySCF density-fitted SCF with separate JKFIT and RI MP2 auxiliary bases and an unpruned 100 by 302 Treutler grid. The second uses TORQ's existing experimental GKS double-hybrid energy evaluator with conventional SCF/PT2 and its level-5 grid. Actual native logs, checkpoints, expanded bases, orbitals, grids, component energies, source hashes, and an execution receipt are written. Missing convergence fails the calculation; no synthetic energy is substituted.

[precommit-native-verification-summary.json](precommit-native-verification-summary.json) reports the genuine earlier acquisition-helper calculations. Its exact artifact hashes were checked against the retained original files; it does not claim the subsequently packaged helper has already executed. The density-fitted total residual against printed Psi4 output was approximately −9.67 × 10⁻⁸ hartree; the conventional TORQ residual was approximately −8.68 × 10⁻⁶ hartree. Different integral and quadrature conventions remain visible. These residuals have no invented acceptance tolerance, uncertainty, curator attestation, or benchmark qualification. Historical Psi4 version and original basis bytes are unavailable. The first genuine attempt failed during JSON serialization after SCF/PT2 and was retained before the successful retry.

Original sources: [pinned notebook](https://github.com/MolSSI/QCArchiveExamples/blob/60ee49a3dc881cd64d42840a2a5f3bf31f7b049f/cookbook/molecule_records.ipynb), [pinned license](https://github.com/MolSSI/QCArchiveExamples/blob/60ee49a3dc881cd64d42840a2a5f3bf31f7b049f/LICENSE). The printed functional cites S. Grimme, *J. Chem. Phys.* **124**, 034108 (2006). This packet does not claim that this tutorial is an independent peer-reviewed benchmark campaign.

The packaged helper was subsequently run genuinely from clean TORQ commit `00aa3ffcc8f3c218752791861f8f4363f3af383a`; both density-fitted and existing conventional routes completed, with the original owned worker waited, no remaining observed descendants, and a 786,640,896-byte sampled peak RSS. The [public evidence](../../../docs/development/release_candidate_evidence/published-reference-20261008/summary.json) retains exact component residuals and source/array digests. These different integral/grid conventions remain unqualified as a matched numerical or scientific-accuracy benchmark.
