# Computational Prediction of Spectroscopic Observables for van der Waals Complexes

> Production ML automation follows [ML safeguards](../ML_SAFEGUARDS.md).
> This contract supersedes ML-only rejection/cutoff and uncalibrated allocation
> rules below. Defer reversibly, retain candidates, validate QC sentinels, and
> never infer pathway absence from an ML barrier or failed search.

**PI/Developer:** Dr. Joshua John Klaassen
**ORCiD:** https://orcid.org/0009-0007-1506-4401
**GitHub CoChem Organization:** https://github.com/ProfJJK-CoChem

## A method matrix for microwave-led assignment of fluxional 5–10 atom complexes

**Version 4 — 9 August 2026.** Supersedes v3 of the same date. Prepared by the chair following a second eight-agent conference (GPU electronic structure; potential-energy surfaces and throughput; codes and acquisition; conformer and isomer search; integration and adversarial audit; state persistence and job chaining; composite and combined methods; heterogeneous orchestration).

**Revision note.** v4 does three things v3 did not. It **reverses v3's ruling on the GPU**, which was wrong, and replaces it with a measured crossover and a protocol the reader can run on his own machine. It adds the three operational layers v3 lacked entirely — **concurrency**, **state reuse across chained jobs**, and a **persistent HDF5 store** — and it adds a **composite/frozen-monomer** treatment and a **GOAT-versus-CREST** verdict. And it restructures for use: a one-page decision card first, a quick start second, stable row IDs on every tier row, seven visible columns with per-row expansion blocks, and a provenance tag `[M]` / `[D]` / `[E]` on every number. v3's correct content — the three product classes, the accuracy specification, the ΔB/B = 2ΔR/R propagation, the corrected ORCA syntax, the secondary observables, the two-stage deduplication protocol, the salvaged path-integral protocol, the discrete-variable-representation treatment, the validation protocol, the failure modes, the licensing analysis and the claims-not-made list — survives, corrected where the second conference found errors.

A note on tone, which is itself a correction. v3 swung too far toward pessimism and read in places as though nothing worked. Six of those over-corrections are walked back in v4. The document should leave a reader with a route, not an apology.

---

## Changes in this revision

All twenty improvements from the second conference are implemented. The table is the index; the section named in each row carries the detail and the citations.

| # | Area | What v3 said | What v4 says |
|---|---|---|---|
| 1 | **GPU role** (§8.2) | "0.556 TFLOPS FP64 against 0.61–0.64 TFLOPS AVX2 … the ceiling is about 1× … no legitimate role in electronic structure" | **Overturned.** Gaussian integral kernels are memory-, register- and occupancy-bound, not FP64-peak-bound: LibintX sustains 25–70 % of FP64 peak *in double precision* and measures 107–1171× against a 73:1 peak ratio ([LibintX](https://arxiv.org/html/2405.01834v2)). gpu4pyscf's production path is FP64 and reproduces CPU PySCF to <10⁻¹¹ Ha ([Wu *et al.*](https://arxiv.org/html/2404.09452v2)). Projected **5–30× [E]** over 8 P-cores for DF-SCF/def2-TZVPP on 40–80 atoms. Version corrected to **v1.8.0**; ECP, D3/D4 and analytic HF/DFT Hessians are on the GPU |
| 2 | **GPU crossover** (§8.3) | no crossover analysis; a blanket exclusion | A quantitative crossover table: A100 vs 32 Xeon cores gives **0.182× at (H₂O)₂ (~118 bf) [M]**, 1.37× at (H₂O)₃, 2.67× at (H₂O)₄, 8.03× at (H₂O)₁₀ ([gpu4pyscf benchmarks](https://raw.githubusercontent.com/pyscf/gpu4pyscf/master/benchmarks/scf/scf_pyscf_qchem.md)); crossover ≈150–170 bf against 32 cores, hence **≈50–90 bf against 8 P-cores [D]**. Plus a matched-input fair-comparison protocol, and the honest limits: no GPU DLPNO or CCSD(T) in any free code, no double hybrids, a g-function ceiling, 24 GB |
| 3 | **Concurrency** (§8A) | no concurrency model at all; every tier costed as if it owns the machine | A `Concurrency` tag (C/G/P/S) on every tier row, a contention budget, and the **scout-and-anchor** heterogeneous pipeline with a Parsl two-executor configuration. Parallelism is **~85 % real [D]**: a profiled MACE step is 31.9 ms wall of which 18.1 ms is host-side ([MACE profiling](https://arxiv.org/html/2510.23621v1)). Reserve one P-core; run ORCA at 7 ranks. Costed 9-atom example: **11.09 h → 3.6 h, 3.1× [E]** |
| 4 | **MPS** (§8A.4) | absent | MPS is the mandatory GPU configuration for small-job concurrency: without it contexts from different processes "cannot execute concurrently" ([NVIDIA MPS architecture](https://docs.nvidia.com/deploy/mps/architecture.html)); the limit is 48 client contexts ([MPS overview](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)) but the practical ceiling here is **2–4 workers, bounded by host cores [D]**. MIG is verified unsupported on GeForce ([MIG supported GPUs](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html)) |
| 5 | **State reuse** (§8B) | no state columns; chaining unaddressed | `State-in` / `State-out` columns on every tier row, an 11-arrow canonical pipeline, a working driver script, and the finding that **the highest-value transfer is the converged geometry, not the wavefunction** — in the DLPNO row each optimisation cycle is 6N = 60 single points ([ORCA numerical gradients](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/numericalgradients.html)) |
| 6 | **Initial Hessians** (§8B.3) | `Calc_Hess true` used freely | **Forbidden unless the Hessian is itself the product.** ORCA: "the use of the exact Hessian as initial one is only of little help … much more time is spent in the calculation of the initial Hessian" ([ORCA optimizations](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations.html)); an exact PM6 Hessian gave 24/21/110/120 steps against 32/36/109/95 for a free GAFF preconditioner ([Mones, Ortner & Csányi](https://pmc.ncbi.nlm.nih.gov/articles/PMC6143621/)). Use `InHess XTB2` or `Lindh` |
| 7 | **Restartability** (§8B.6) | a footnote | A routing input. ORCA: "Numerical frequency calculations are restartable (but analytical frequency calculations are not)" ([ORCA frequencies](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/frequencies.html)); MDCI documents no restart ([ORCA MDCI](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/mdci.html)); CFOUR restarts CC from `JOBARC`/`JAINDX`/`MOINTS`/`MOABCD`. Non-restartable work is **decomposed**, not checkpointed; CFOUR `FD_IRREP` is the worked example |
| 8 | **HDF5 store** (§8C) | implied ASE `db` could serve as an HDF5 layer | Corrected: ASE `db` has **no HDF5 back-end** (JSON/SQLite3/PostgreSQL/MySQL/MariaDB only, [ASE database docs](https://wiki.fysik.dtu.dk/ase/ase/db/db.html)); **PySCF's chkfile is the genuine HDF5 layer** ([PySCF lib API](https://pyscf.org/pyscf_api_docs/pyscf.lib.html)). A working resizable, chunked, gzip+shuffle+fletcher32 `PESStore` class with [QCSchema](https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html) field names; `scaleoffset` rejected as lossy |
| 9 | **Isotopologues** (§6.10, §8B.4) | absent | The free re-analysis shortcut: once a Hessian exists, every isotopologue's constants and vibrational corrections come from re-analysing the same force field at zero electronic-structure cost — a **6–15× saving [D]** on a parent + ¹³C + D campaign |
| 10 | **Frozen-monomer composite** (§9A.1–9A.2) | not present | Adopted as the default, with the emphasis corrected. At R = 2.836 Å in CO₂···H₂O, ΔR = 0.002 Å costs the same in B as a **16.8 mÅ uniform monomer bond error [D]** — and no method errs by that much covalently (fc-CCSD(T)/VTZ is 3 mÅ, [Fortenberry *et al.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10537648/)). **Freeze good monomers to fix A; spend the remaining budget on R to fix B and C** |
| 11 | **ChS / junChS / junChS-F12** (§9A.4) | unnamed composite rows | Named and costed. ChS MAE in B_e is **0.13 % [M]** for ≤16 atoms against fc-CCSD(T)/VTZ's 0.80 % ([Puzzarini & Stanton](https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04706c)); junChS-F12 gives A14 MUE 0.06 kJ/mol and SE100 MUE(r) 0.0011 Å and is "one order of magnitude faster than the CBS+CV counterparts". **junChS-F12 is the buy** where a Molpro licence exists |
| 12 | **Additive diffuse corrections and ONIOM** (§9A.5) | not addressed | **Prohibited**: an additive diffuse correction degrades the energy MAE from 1.52 % to **12.74 % [M]** and CH₄···NH₃ geometry by 0.2 Å. ONIOM / QM-QM2 **rejected at 5–10 atoms** — nothing to demote, no bond to cut, no saving. Many-body expansion is trivial for dimers but matters at trimers, where 3-body terms are 15–20 % and D3/D4 is pairwise |
| 13 | **B_e vs B₀** (§3) | conflated | Split. B_e is reachable to **0.13 % [M]** by composite schemes; B₀ additionally requires ΔB_vib, which is where the floppy-complex error lives. Every tier row states which quantity it delivers, and a **frozen-monomer flag** shows whether the monomers were relaxed |
| 14 | **Two tracks** (§9, §13–§14) | one implied driver | Separate **ORCA track** and **CFOUR track** with an acquisition table. **CFOUR has analytic CCSD(T) second derivatives; ORCA has analytic Hessians for SCF only.** At N = 10 the anharmonic force field is 49 analytic CFOUR Hessians against **176,400 ORCA/DLPNO single points**, a ratio of 36N² = 3,600 [D]. CFOUR has no 10 s or 1 min entry; ORCA has no 1 month CCSD(T)-anharmonic entry |
| 15 | **Two ORCA capabilities restored** (§9.3) | implied ORCA cannot optimise at CCSD(T); understated its VPT2 output | Corrected. ORCA **does** have canonical CCSD(T) analytic gradients via AUTOCI (RHF and UHF) — it is DLPNO that lacks them. ORCA's VPT2 **does** emit centrifugal distortion, Watson parameters, α constants and a Pickett/SPCAT export. Sextic distortion remains CFOUR-only |
| 16 | **GOAT vs CREST** (§9B.1–9B.2) | implied CREGEN was the better filter | Corrected. GOAT's filtering is "precisely the same as that of CREST" ([ORCA GOAT manual](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/goat.html)); the one independent head-to-head gives GOAT average F1 **0.93 [M]** against CREST's 0.74–0.80 ([racer benchmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977065/)). **Keep GOAT primary; add CREST `--nci --nocross --noreftopo` as an independent second search and carry the union** |
| 17 | **MLFF-driven GOAT** (§9B.4) | treated as an off-label hack | Published as a supported recipe — tutorial Example 3 is GOAT + AIMNet2 with `! EXTOPT GOAT PAL8` ([ORCA external-methods tutorial](https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extopt.html)) — with the three details v3 lacked (two venvs, server mode, `TolE 1e-5`) and the hard limit: foundation-model *interaction* energies carry 3.5–7.3 kcal/mol errors on S30L ([MACE-POLAR-1](https://arxiv.org/html/2602.19411v1)). **An excellent enumerator and a poor judge** |
| 18 | **PES re-tiering** (§13.2, Table 2) | 1 w = 10³–10⁴ points; 1 mo = 6-D campaign | Re-tiered around active learning, worth 20–100×: **472 [M]** actively selected points reach 0.3253 cm⁻¹ weighted RMSE against a 47,945-point test set ([active-learning PES](https://chemrxiv.org/engage/chemrxiv/article-details/675b9e3bf9980725cfe8476a)), and Δ-learning needs "as few as 200 CCSD(T) energies" ([Δ-learning PES](https://arxiv.org/abs/2011.11601v1)). **The 1 w PES row becomes a 12 h row and the 1 mo row becomes 3 d.** autoPES's fit error corrected (v3 understated flex-autoPES by ~8×); PySCF campaigns run as N concurrent serial jobs |
| 19 | **Provenance discipline** (§12.5, §21) | unsourced ratios decided routing | Every number tagged `[M]` measured, `[D]` derived, `[E]` estimated, with the standing rule that **no `[D]` or `[E]` value may be the sole support for a hardware exclusion, a routing gate or an accuracy claim** — and where one currently is, the document says so and marks it for local measurement. All 18 catalogued internal inconsistencies fixed. Three new quantified error channels added: BSSE as a first-order error in B (+4.1 pm ≈ 2.8 % at cc-pVTZ, [Plumley & Dannenberg](https://pmc.ncbi.nlm.nih.gov/articles/PMC3073166/)); frozen core as a **−0.81 % [M]** bias at cc-pVQZ ([Bologna 74-isotopologue benchmark](https://cris.unibo.it/bitstream/11585/656295.2/1/Benchmark_paperS.pdf)); and a split-conformal replacement for the Bayesian anchor, which excludes its own subject matter ([Lee & McCarthy](https://par.nsf.gov/servlets/purl/10149706)) |
| 20 | **Structure and calibrated optimism** (front matter, §12, §15, §21) | 400 KB, 24 sections, 282 references, indexed by wall clock | A one-page decision card first, quick start second, Pareto frontier and failure modes third and fourth, seven visible columns with per-row expansion blocks, stable row IDs, derivations demoted to appendices. Six over-corrections walked back: classical MD reinstated as a **diagnostic**; "no tier reaches 0.1 %" inverted into "what 1 % buys you" (formamidinium formate was assigned from constants off by 1.7 %, [Zhou *et al.*](https://research-repository.st-andrews.ac.uk/bitstream/handle/10023/16997/Zhou_2019_JCP_Synthesismicrowavespectra_AAM.pdf)); the MLFF geometry ban restricted to Product A; three withdrawn tight rows reinstated as Product B; the apology split into seven hard limits plus a roadmap; inflated `n.a.` cells cleared. The teaching tier's false constraint removed — [ChemCompute](https://chemcompute.org/) gives classes free access to Expanse, Bridges2 and Delta |

### Preamble: Corrections to v3's GPU ruling

**v3 was wrong.** Its Ruling 9 concluded that the RTX 3090 has "no legitimate role in electronic structure" and instructed that every GPU column entry be replaced with `MLFF inference only` or `n.a.` That ruling is withdrawn in full.

**Why the reasoning failed.** v3 divided a published specification — 35.6 TFLOPS FP32 ÷ 64 for consumer Ampere FP64 = 0.556 TFLOPS — compared it to an AVX2 peak of 0.61–0.64 TFLOPS from eight P-cores, and emitted a conclusion about *application* performance. Two peak numbers can be compared only if both machines realise comparable fractions of their peak, and in Gaussian-basis electronic structure they do not. gpu4pyscf's own roofline puts uncached electron-repulsion-integral arithmetic intensity at ~3/16 FLOP/byte against a machine balance of 6.1, i.e. memory-bound, with registers, local memory and streaming-multiprocessor occupancy as the named limiters ([Li, Zhang, Sun & Chan](https://arxiv.org/html/2407.09700v1)); TeraChem's authors state that "all of the HF exchange matrix kernels" are memory-bound ([TeraChem](https://arxiv.org/html/2406.14920v3)). The decisive control experiment already existed: LibintX evaluates integrals **in double precision** at 25–70 % of hardware peak and measures GPU-versus-CPU-core speedups of **107–1171× against a peak-FLOP ratio of only 73:1** ([LibintX](https://arxiv.org/html/2405.01834v2)). A measured ratio that exceeds the peak ratio is arithmetically impossible if peak FLOPS governed; it proves the CPU realises a far smaller fraction of its own peak.

**A second, independent error.** v3's precision premise was a factual substitution: it applied TeraChem's mixed-precision model to gpu4pyscf. gpu4pyscf's production path is double precision and reproduces CPU PySCF to **<10⁻¹¹ Ha in energy, <10⁻⁷ Ha/bohr in gradients and <10⁻⁶ Ha/bohr² in Hessians** ([Wu *et al.*](https://arxiv.org/html/2404.09452v2)). There is no accuracy penalty for routing a DFT energy, gradient or Hessian to gpu4pyscf on this card.

**Consumer-card evidence, which v3 said did not exist.** Rowan benchmarked gpu4pyscf on T4 (0.254 TFLOPS FP64), L4 (0.473) and A10 (0.976) — **the 3090's 0.556 sits between the L4 and the A10** — reporting 13× over Psi4 on 78-atom maraviroc ([Rowan](https://www.rowansci.com/blog/gpu4pyscf)). BrianQC's entire vendor benchmark suite runs on a GTX 1080 Ti, a card with half the 3090's FP64 rate ([BrianQC](https://www.brianqc.com/benchmarks)). TeraChem's own timings show a GeForce 1080 Ti beating a Tesla K80 despite the K80's ~9× FP64 advantage ([PetaChem](http://www.petachem.com/performance.html)).

**What replaced it.** Section 8 now carries a corrected routing rule keyed on **capability, size, algorithm, memory and precision** rather than on peak FLOPS; a crossover table; and a matched-input fair-comparison protocol (§8.4) so the reader can settle the question on his own machine rather than taking either ruling on trust. Section 8A adds the concurrency model that makes the card pay: **the GPU earns its place on this hardware through throughput on many small jobs, not latency on one.**

**What survives from v3's hardware analysis, unchanged.** ORCA has no GPU path — that is a software-capability fact and it is correct; no ORCA row runs its own electronic structure on a GPU. The 0.556 TFLOPS derivation itself is arithmetically right; it is the inference from it that failed. The eight-rank / `%maxcore 3000` guidance, the GitHub Actions limits and the ORCA licence findings are unaffected. And the specific exclusions stand: there is **no GPU MPQC CCSD(T)-F12 in any free code**, no double hybrids in gpu4pyscf, a g-function basis ceiling, and 24 GB is a hard wall with no graceful spill.

**The generalised lesson, adopted as a standing rule.** The failure was an instance of a repeatable move — divide a specification, emit a conclusion about application performance, never check a benchmark. Fifteen further instances of it were found in v3. Section 12.5 therefore tags every number `[M]`, `[D]` or `[E]` and forbids a `[D]` or `[E]` value from being the sole support for a hardware exclusion, a routing gate or an accuracy claim. Applied mechanically, that rule would have caught this error before it reached a ruling.

---

## The one-page decision card

*Read this page. If it answers your question, stop.*

**Step 0 — which product are you building?** Answer this before anything else; it changes what is reachable by an order of magnitude.

| | You have | You can claim | Go to |
|---|---|---|---|
| **A** | nothing measured | **0.3–0.5 %** semi-rigid · **1–2 %** floppy in B₀; **0.13 %** in B_e by composite | Tables 3, 4 |
| **B** | a measured parent or close analogue | **0.03–0.06 %** | Table 4, Product-B rows |
| **C** | you need only a *difference* (isotopologue, conformer) | **0.02–0.1 %** | Table 4, Product-C rows |

**Step 1 — convert accuracy to a search window.** At 12 GHz: ±0.1 % = ±12 MHz · ±0.5 % = ±60 MHz · ±1 % = ±120 MHz · ±2 % = ±240 MHz. At roughly one line per MHz in a dense chirped-pulse spectrum a Product-A prediction leaves 60–240 candidate lines. **That is a normal, publishable situation** — formamidinium formate was assigned from constants agreeing "within 1 %", the best functional reproducing A to 98.3 % ([Zhou *et al.*](https://research-repository.st-andrews.ac.uk/bitstream/handle/10023/16997/Zhou_2019_JCP_Synthesismicrowavespectra_AAM.pdf)).

**Step 2 — spend in this order (binding).** ① geometry, best affordable · ② cheap anharmonic ΔB_vib on that geometry · ③ quartic distortion (free) · ④ dipole components · ⑤ ¹⁴N quadrupole tensor. Never buy a better equilibrium structure before you have bought ΔB_vib.

**Step 3 — pick the row.**

| Budget | Recommended row | What you get | Do not expect |
|---|---|---|---|
| 10 s – 1 min | `T3-10s`, `T3-1min` | topology screen, ±1–5 % in B_e | a search window |
| 30 min – 1 h | `T3-30min`, `T1-1h` | ±0.5–3 %, conformer enumeration, CREST cross-check | branch-type reliability |
| 3 h – 5 h | **`T3-3h` (recipe R2: frozen monomers + ωB97M-V/QZ + CP + VPT2)** | ±0.4–1.5 % in B_e, **A to <0.2 %**, ΔB_vib, a CP-bracketed energy | 0.1 % in B₀ |
| 12 h | **`T3-12h` (junChS)** — the best de novo accuracy-per-core-hour row in the document | **0.13 % MAE in B_e [M]** for ≤16 atoms | that B₀ inherits it; ΔB_vib is still 0.1–0.7 % |
| 1 d | `T4-1d` | ΔB_vib and B₀ for the semi-rigid manifold | floppy-mode averaging |
| 3 d – 1 mo | `T3C-3d`…`T4C-1mo` (CFOUR track) | sextic distortion, isotopologue force fields, CCSD(T)-quality anharmonicity | that the ORCA track can reach here |

**Step 4 — the five free observables.** Inertial defect Δ = I_c − I_a − I_b (the sign must be right, always) · planar moments P_aa > P_bb > P_cc · quartic distortion constants · direct dipolar coupling D ∝ r⁻³ · vibrational satellites. Each costs nothing once the force field exists and each independently constrains the structure.

**Step 5 — five ways this goes silently wrong.** ① a loose `!Opt` leaves ~2 % in B · ② frozen core costs **−0.81 %** at cc-pVQZ ([Bologna benchmark](https://cris.unibo.it/bitstream/11585/656295.2/1/Benchmark_paperS.pdf)) · ③ a small basis without counterpoise shifts an intermolecular distance by 4–9 pm ([Plumley & Dannenberg](https://pmc.ncbi.nlm.nih.gov/articles/PMC3073166/)) · ④ an MLFF geometry is never a Product-A geometry · ⑤ S66/GMTKN55 statistics do not transfer to this class (ωB97X-D: 0.58 pm on A21, **36.34 pm on rare gases**).

**Hardware, in one line.** ORCA is CPU-only — a software fact, not a physics one. For GPU acceleration use gpu4pyscf, which runs FP64 SCF, analytic gradients and analytic Hessians on any card from Volta / RTX 20-series upward ([gpu4pyscf](https://github.com/pyscf/gpu4pyscf)) and is measured at 13–30× a 32-core node for DFT ([Wu *et al.*](https://arxiv.org/html/2404.09452v2)) — but **below roughly 50–90 basis functions on this box the CPU wins**, and the card's real value here is throughput on many small jobs under MPS (§8A.4).

**Codes, in one line.** ORCA owns search, energetics and DFT-quality VPT2; CFOUR owns every CCSD(T)-quality anharmonic tier and sextic distortion. Start the CFOUR licence paperwork on day one — it is a printed, signed form posted to Mainz ([CFOUR download](https://cfour.uni-mainz.de/cfour/index.php?n=Main.Download)), and it is the long pole, not the science.

---

## Quick start

Three workflows cover most of what this document is for. Each is a numbered recipe with commands you can run.

### QS-1 — "I have a complex and no spectrum" (de novo prediction, one working day)

1. **Enumerate binding topologies by hand.** For a 5–10 atom complex this is 3–9 structures and it is the only step in the document that can support a completeness claim, as done for HFIP⋯Ne/Ar ([HFIP⋯Rg](https://pubs.acs.org/doi/10.1021/acs.jpca.1c03757)). Save as `seed01.xyz … seedNN.xyz`.
2. **Search, on both devices at once.** CPU:
   ```
   ! GOAT XTB2 PAL7
   %goat maxen 12.0 confdegen auto gfnuphill gfnff end
   * xyzfile 0 1 seed01.xyz
   ```
   GPU, concurrently, from the same seed (start the server first — `~/bin/oet-aimnet2/oet_server aimnet2 --nthreads 4 -d cuda &`):
   ```
   ! GOAT-EXPLORE ExtOpt TightOpt PAL1
   %method ProgExt "<USER_HOME>/bin/oet-aimnet2/oet_client" Ext_Params "-b localhost:8888" end
   %scf TolE 1e-5 end
   %goat maxen 12.0 conftemp 298.15 confdegen auto end
   * xyzfile 0 1 seed01.xyz
   ```
   The two runs cost the same wall time as either alone and explore different surfaces. Take the union.
3. **Cross-check with CREST and referee the union once.**
   ```bash
   crest seed01.xyz --nci --gfn2 --ewin 12 --nocross --noreftopo --T 7
   cat *.finalensemble.xyz crest_conformers.xyz > union.xyz
   crest --screen union.xyz --gfn2 --ewin 12 --T 7     # one common level, then CREGEN
   ```
4. **Optimise the survivors at r²SCAN-3c with frozen monomers (recipe R1),** then at ωB97M-V/def2-QZVPP with frozen monomers (recipe R2), using the mandatory `%geom` block of §4.4 and an xTB model Hessian:
   ```
   ! wB97M-V def2-QZVPP def2/J RIJCOSX TightOpt TightSCF DEFGRID3
   %base "s4"
   %pal nprocs 7 end
   %maxcore 3400
   %geom InHess XTB2
         TolE 1e-7  TolRMSG 3e-6  TolMaxG 1e-5  TolRMSD 5e-5  TolMaxD 1e-4
         Constraints { ... intramolecular internals ... } end
   end
   * xyzfile 0 1 s2.xyz
   ```
5. **ΔB_vib from a DFT VPT2 on the semi-rigid manifold**, then quartic distortion, inertial defect and planar moments for free; export a Pickett file with `%output Pickettname "x.txt" end`.
6. **Report** B_e, ΔB_vib, B₀, the final `MaxG`, the softest force constant, the residual gradient on the frozen coordinates, and the window half-width in MHz.

Row IDs traversed: `T1-1min` → `T1-30min` → `T1-1h` → `T3-1min` → `T3-3h` → `T4-1h`.

### QS-2 — "I have a measured parent and need an isotopologue or an analogue" (Product B/C, under an hour)

1. **Take the measured A, B, C of the parent** and any reasonable computed geometry.
2. **Scale the geometry to reproduce the parent constants** (Kisiel's structural programs), then substitute masses. This is the only route in the document that reliably reaches **0.03–0.1 %**.
3. **If you already have a Hessian for the parent, every isotopologue is free.** An ORCA `Freq` run that finds a `.hess` repeats only the analysis ([ORCA frequencies](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/frequencies.html)); `orca_vib file.hess` re-diagonalises it; CFOUR does the same with `ISOMASS` + `xjoda` against a saved `JOBARC`. For a parent + ¹³C + D campaign this is a **6–15× saving [D]**.
   ```bash
   for iso in 13C 18O D; do cp s5.hess iso_$iso.hess; orca_vib iso_$iso.hess > iso_$iso.vib.out; done
   ```
4. **Report the shift, not the absolute constant** — the geometry error cancels in the difference, which is why the difference is ten times more accurate than either endpoint.

Row IDs: `T4-1min` (Product B) → `T4-30min`.

### QS-3 — "I need an intermolecular surface" (PES campaign, one day instead of one month)

1. **Register the campaign in one HDF5 file**, not thousands of `.out` files (§8C). `PESStore(...)`, `register_method(...)`, `register_grid(...)`.
2. **Generate a DFT base grid of ~2,000 points as N concurrent single-thread jobs**, never one big threaded job — PySCF's developers state a small system "may not see much difference of speed" from threading ([PySCF issue 1360](https://github.com/pyscf/pyscf/issues/1360)):
   ```bash
   seq 0 1999 | parallel -j 16 --joblog pes.log --eta ./pes_run.sh {}
   seq 0 1999 | parallel -j 16 --joblog pes.log --resume-failed ./pes_run.sh {}   # after any interruption
   ```
3. **Fit a committee and select 300–800 points by active learning**, using an error-based or two-set acquisition — **not** pure variance maximisation, which plateaus an order of magnitude worse ([Uteva *et al.*](https://nottingham-repository.worktribe.com/OutputFile/1190028)).
4. **Escalate only the selected points to CCSD(T)-F12 or DLPNO-CCSD(T1)** and fit a Δ-correction: "as few as 200 CCSD(T) energies" have sufficed ([Δ-learning PES](https://arxiv.org/abs/2011.11601v1)).
5. **Budget a held-out validation grid separately and report its residual in cm⁻¹.** The 472-point claim was earned against a 47,945-point test set ([active-learning PES](https://chemrxiv.org/engage/chemrxiv/article-details/675b9e3bf9980725cfe8476a)). A 500-point surface with a 100-point held-out set is not a spectroscopic surface.
6. **Solve the nuclear problem matrix-free**, and if you use JAX put `JAX_ENABLE_X64=True` on line 1 — JAX "by default enforces single-precision numbers" and the flag "only works on startup" ([JAX sharp bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html)).

Row IDs: `T2-1h` → `T2-1d` → `T2-12h` (Δ-learned surface) → `T2-3d` (6-D variational).

---

## 1. The three products and the routing question

### 1.1 Why the previous versions were internally inconsistent

The 0.1 % target is not an opinion of this conference. Puzzarini and Stanton write that "the level of accuracy associated with predicted rotational constants should be at least 0.1% (corresponding to 10 MHz for a constant with magnitude 10000 MHz)", that "an accuracy of 1% for ground-state rotational constants can be achieved with relative ease, [but] it is not particularly useful for guided searches and spectra interpretation", and that "accuracy rivaling that of experiments (0.01%) is beyond the reach of computation", concluding that "realistic prospects lie somewhat intermediate between these regimes with 0.1% (and slightly better than that) an achievable but still challenging goal" ([Puzzarini & Stanton, *PCCP* 25, 1421 (2023)](https://pubs.rsc.org/en/content/articlepdf/2023/cp/d2cp04706c)).

The delivery side is equally well documented. The 21-dimer optimisation benchmark of Glick, Kumawat and Sherrill reports that the most efficient good performer, MP2D/cc-pVDZ-F12 (paired with CABS), achieves an average least-RMSD of atomic coordinates of 0.02 Å and an absolute centre-of-mass distance error of 0.02 Å against CCSD(T)/CBS ([Glick, Kumawat & Sherrill, *J. Chem. Phys.* 162, 174106 (2025)](https://pubs.aip.org/aip/jcp/article/162/17/174106/3345631/Evaluating-wavefunction-methods-the-counterpoise)). At a 3.8 Å centre-of-mass separation, 0.02 Å is 0.5 % in R and therefore about 1 % in B — five to ten times the target. For direct ab initio equilibrium constants in the semi-rigid world the same gap appears: mean deviations of computed equilibrium rotational constants over a seven-molecule test set are 0.5 % for revDSD, 0.4 % for the template-molecule scheme and 0.3 % with linear-regression refinement ([Alessandrini & Puzzarini, *JPCA* 2021](https://cris.unibo.it/retrieve/handle/11585/868614/e1dcb339-596b-7715-e053-1705fe0a6cc9/acs.jpca.1c07828_templaggio.pdf)).

For dispersion-bound complexes the situation is worse. A systematic survey of rare-gas complexes found that the best method tried for pyrrole–Ne, CCSD/def2-TZVPP, still gave 1.54 % error in the rotational constants, with CCSD/cc-pVDZ-F12 (paired with CABS) at 20.12 %, MP2/cc-pVDZ-F12 (paired with CABS) at 25.66 % and B3LYP/6-311++G(d,p) at 46.34 % ([Rotational spectra of van der Waals complexes, CSIC repository](https://digital.csic.es/bitstream/10261/230932/4/Rotational%20spectra%20of%20van%20der%20Waals%20complexes.pdf)). Basis-set behaviour there is non-monotonic and the method/basis pairings are close to accidental, which is a signature of error cancellation rather than convergence.

Versions 1 and 2 attached sub-0.1 % accuracy claims to de novo tiers. There are no such tiers. Every such claim is withdrawn.

### 1.2 The three products, defined

| Product | What is computed | Precondition | Defensible accuracy in A, B, C | Evidence |
|---|---|---|---|---|
| **A — absolute, de novo** | best affordable equilibrium geometry plus ΔB_vib | none | **0.3–0.5 % semi-rigid; 1–2 % floppy or dispersion-bound** | [Alessandrini & Puzzarini 2021](https://cris.unibo.it/retrieve/handle/11585/868614/e1dcb339-596b-7715-e053-1705fe0a6cc9/acs.jpca.1c07828_templaggio.pdf); [Glick, Kumawat & Sherrill 2025](https://pubs.aip.org/aip/jcp/article/162/17/174106/3345631/Evaluating-wavefunction-methods-the-counterpoise); [CSIC rare-gas survey](https://digital.csic.es/bitstream/10261/230932/4/Rotational%20spectra%20of%20van%20der%20Waals%20complexes.pdf) |
| **B — semi-experimental / template-anchored** | B_e^SE = B_0^exp − ΔB_vib^calc, or a trial geometry scaled to reproduce the measured parent | one measured parent isotopologue, or a measured structural analogue | **≤0.1 %, typically 0.03–0.06 %** | [Puzzarini & Stanton 2023](https://pubs.rsc.org/en/content/articlepdf/2023/cp/d2cp04706c); [Alessandrini & Puzzarini 2021](https://cris.unibo.it/retrieve/handle/11585/868614/e1dcb339-596b-7715-e053-1705fe0a6cc9/acs.jpca.1c07828_templaggio.pdf); [Melli *et al.* 2022](https://pubmed.ncbi.nlm.nih.gov/36149341/) |
| **C — differences** | isotopologue shifts Δ(A,B,C), conformer-to-conformer differences, vibrational satellite spacings, inertial defect | a parent measurement for the shift, or two structures at one level for the difference | **0.02–0.1 %** | [Melli *et al.* 2022](https://pubmed.ncbi.nlm.nih.gov/36149341/); [Demaison *et al.* 2021](https://pubs.aip.org/aip/jcp/article/154/19/194302/565922/How-accurate-is-the-determination-of-equilibrium); [Kisiel, structural programs](http://info.ifpan.edu.pl/~kisiel/struct/struct.htm) |

Why Products B and C work is not mysterious: they cancel systematic error rather than removing it. The vibrational correction ΔB_vib is only 0.1–0.7 % of B_e, "with 0.5% being the largely dominant case in semi-rigid systems", and errors in the force constants are "usually less than 5%" for good coupled cluster and "can increase up to 20%" for low-level methods, so "the error in the vibrational contribution to the rotational constants can be estimated to be no larger than about 0.05% of the total value of the ground state constants B_0", with "a maximum of 0.1% … in the case of a vibrational contribution of 0.5% evaluated with an error of 20%" ([Puzzarini & Stanton 2023](https://pubs.rsc.org/en/content/articlepdf/2023/cp/d2cp04706c)). Semi-experimental constants built with cheap B3LYP-class vibrational corrections deviate about 0.06 % on average, 0.03 % excluding outliers, from full semi-experimental values ([Alessandrini & Puzzarini 2021](https://cris.unibo.it/retrieve/handle/11585/868614/e1dcb339-596b-7715-e053-1705fe0a6cc9/acs.jpca.1c07828_templaggio.pdf)). The fragment-based template-molecule plus linear-regression approach delivers constants with "relative accuracy well within 0.1%" ([Melli *et al.*, PubMed 36149341](https://pubmed.ncbi.nlm.nih.gov/36149341/)).

Product B works even for a genuine van der Waals complex. For N₂O⋯CO the semi-experimental equilibrium structure from two isotopologues plus an ab initio anharmonic force field gives R_cm = 3.825(13) Å and r(C⋯O) = 3.300(9) Å — about 0.01 Å, five times better than the best direct optimisation. The same study warns that diffuse-function and counterpoise contributions "were not additive; did not compensate each other; had almost the same value but opposite signs", and that the mass-dependent method "did not permit the determination of reliable intermolecular parameters" ([Demaison *et al.*, *JCP* 154, 194302 (2021)](https://pubs.aip.org/aip/jcp/article/154/19/194302/565922/How-accurate-is-the-determination-of-equilibrium)).

### 1.3 The routing question

```
                    ┌─────────────────────────────────────────────────────────┐
   START  ───────►  │  Do you have a measured parent isotopologue, or a       │
                    │  structurally analogous measured complex?               │
                    └──────────────┬───────────────────────┬──────────────────┘
                                   │ YES                   │ NO
                                   ▼                       ▼
                    ┌──────────────────────────┐  ┌────────────────────────────┐
                    │  PRODUCT B / C           │  │  PRODUCT A                 │
                    │  semi-experimental or    │  │  absolute de novo          │
                    │  template-scaled         │  │                            │
                    │  window  ±0.03–0.1 %     │  │  window ±0.3–0.5 % rigid   │
                    │  = ±4–12 MHz at 12 GHz   │  │        ±1–2 %  floppy      │
                    │  Cost: DFT-level         │  │  = ±36–240 MHz at 12 GHz   │
                    └──────────────┬───────────┘  └──────────────┬─────────────┘
                                   │                             │
                                   ▼                             ▼
                    ┌──────────────────────────┐  ┌────────────────────────────┐
                    │ Spend on: ΔB_vib,        │  │ Spend on: geometry, then   │
                    │ isotopologue shifts,     │  │ ΔB_vib, then dipoles, then │
                    │ dipoles, χ tensor        │  │ χ; expect an AUTOFIT-scale │
                    │ Assignment: pattern      │  │ combinatorial search       │
                    │ matching, hours          │  │                            │
                    └──────────────────────────┘  └────────────────────────────┘
```

The operational difference is large and is the whole argument for the 0.1 % target. In a real CP-FTMW campaign on furonitrile–water complexes, junChS-type equilibrium geometries with revDSD/B3LYP vibrational corrections gave 0.08–0.16 % discrepancies in the rotational constants, and "typically, all the transitions were found within 20 MHz from the corresponding predictions", with assignment straightforward ([Alessandrini *et al.*, *PCCP* 2023](https://pubs.rsc.org/en/content/articlepdf/2023/cp/d3cp03984f)). At 1 %, the search window is ±120 MHz, of order 10³ candidate lines per transition, which is the regime in which one runs a brute-force triples fit: AUTOFIT evaluates 35–50 triples per second per core, so "a 10,000,000 triples fit run (a standard scan size) will take around 11 hours" ([pategroup AUTOFIT](https://github.com/pategroup/bband_scripts/tree/master/autofit); [HS-AUTOFIT, *Electronics* 10, 2251](https://cris.unibo.it/retrieve/e1dcb339-6b53-7715-e053-1705fe0a6cc9/electronics-10-02251-v2.pdf)).

### 1.4 Two use cases, stated separately

The matrix serves two purposes with opposite cost rankings, and pretending they are one problem was a defect of version 2.

- **Assignment.** Geometry first, then ΔB_vib, then the secondary observables. A one-month coupled-cluster binding energy contributes nothing. Section 3.3 gives the binding spend priority.
- **Method benchmarking.** The ranking inverts: the reference energy is the deliverable, and the sampling is a means. Rows useful for benchmarking are marked as such in the Notes column.

Where a row is useful for one and not the other, the Notes column says which.

### 1.5 A warning that governs the whole document: accuracy is not monotonic in cost

The tier ladder is a cost axis, not a truth axis. For C₅H₂ isomers, frozen-core CCSD(T)/cc-pVQZ gives an error of **−6.56 % in A_e** and −1.15 % in B_e, while the cheap double hybrid DSD-PBEP86-D3BJ shows "percentage errors below 0.5% in nearly all cases"; the authors attribute better-looking all-electron/cc-pVTZ agreement to "fortuitous" cancellation ([Fortenberry *et al.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10537648/)). Independently, Lee and McCarthy report that "the 6-31G(d) (8.5%) results are of comparable uncertainty to those obtained with the cc-pVQZ (8.2%) basis, despite the latter being significantly more computationally expensive" ([Lee & McCarthy, Bayesian analysis](https://par.nsf.gov/servlets/purl/10149706)).

Two consequences are enforced in the tables. Every tier whose accuracy is not better than an earlier tier carries an explicit **"dominated by tier X"** label in its Notes column. And no tier's accuracy claim is stated as a mean alone: each carries a maximum observed benchmark error in its own column (Section 12.2).

---

## 2. Scope, system class and vocabulary

### 2.1 System class

Weakly bound van der Waals and hydrogen-bonded complexes of 5–10 atoms (the nuclear-dynamics discussion extends to about 25 atoms), bound by London dispersion, induction and hydrogen bonding, with at least one large-amplitude intermolecular coordinate, to be assigned in a 2–22 GHz CP-FTMW experiment at a rotational temperature of roughly 1–2 K.

The energy scale spans roughly **0.1–20 kcal/mol**. S66 runs from −19.49 kcal/mol (acetic acid⋯uracil) to −1.43 kcal/mol (benzene⋯ethene), with the water dimer at −4.92 kcal/mol ([Řezáč, Riley & Hobza, S66](https://pmc.ncbi.nlm.nih.gov/articles/PMC3152974/)); pure-dispersion rare-gas dimers sit an order of magnitude below that floor, the benchmark Ar₂ potential having a well depth of 99.351 cm⁻¹ ≈ 0.28 kcal/mol at r_min = 3.762 Å ([Patkowski & Szalewicz](https://pubmed.ncbi.nlm.nih.gov/20831315/)). A 0.3 kcal/mol error is 6 % of the water-dimer binding energy and more than 100 % of Ar₂, so method selection is stated per interaction class throughout.

### 2.2 Explicit scope exclusions

- **Open-shell radicals and their magnetic hyperfine structure.** Fermi-contact and dipolar hyperfine of radicals is a different problem with different methods. It is excluded by decision, not by omission: a reader with an Ar–OH complex should not assume this document covers it.
- **Periodic solids.** ORCA has no periodic boundary conditions, no k-point sampling and no phonon module. Crystalline terahertz work routes to CP2k, Quantum ESPRESSO or VASP with phonopy, under its own convergence protocol, and is out of scope.
- **Resonance Raman.** The autocorrelation formalism of Table 9 assumes the Placzek polarizability approximation, far from resonance.
- **Time-Dependent / Laser Dynamics.** Time-dependent Schrödinger equation (TDSE), attosecond integration, surface hopping, and laser field interactions are strictly out of scope. The ecosystem is explicitly confined to static ab initio geometry optimization, electronic structure, and VPT2 / anharmonic force fields for predicting rotational constants in the microwave regime.

### 2.3 Structure vocabulary

| Symbol | Definition | Comparable to experiment? | Comparable to an optimisation? |
|---|---|---|---|
| r_e | minimum of the Born–Oppenheimer potential, vibrationless | No | **Yes** — this is what an optimiser produces |
| r_0 | effective structure fitted to ground-state B_0 of several isotopologues | Yes (it is an experimental fit) | No |
| r_z | ground-state average structure after removing the harmonic vibrational part | Yes, with a computed harmonic correction | Only after the same correction is added to r_e |
| r_s | substitution structure from Kraitchman's equations | Yes | No |
| r_m, r_m^(1), r_m^(2) | Watson's mass-dependent structures | Yes | Approximately, by construction |
| r_e^SE | fit to B_e^SE = B_0^obs + ΔB_vib^calc | Yes, and it is the reference of choice | **Yes** — the only like-for-like comparison with r_e |

The only theory–experiment comparisons this document reports for a semi-rigid system are B_0^calc against B_0^obs, and B_e^calc against B_e^SE. **For a genuinely fluxional complex none of r_0, r_z, r_s or r_m is physically meaningful**, because each assumes a single small-amplitude expansion about one reference geometry; Kraitchman applied to a tunnelling complex returns the coordinates of a vibrationally averaged, symmetry-lowered fiction.

The mass-dependent structures are Watson's ([J. K. G. Watson, *J. Mol. Spectrosc.* 1973, journal listing](https://scispace.com/journals/journal-of-molecular-spectroscopy-2bqujokn/1973); [later r_m^(1)/r_m^(2) refinements](https://pubmed.ncbi.nlm.nih.gov/11336516/)), and the KRA/STRFIT suite fits them ([Kisiel, structural programs](http://info.ifpan.edu.pl/~kisiel/struct/struct.htm)).

**The Costain error is the reason theory still has a job in a substitution structure.** Coordinate uncertainties come "from propagation of uncertainties in the measured rotational constants, and then the usually much larger Costain's error is added (ie. dz=0.0015/|z| Angstr.)" ([Kisiel, structural programs](http://info.ifpan.edu.pl/~kisiel/struct/struct.htm), citing [Costain](https://pubs.aip.org/aip/jcp/article/29/4/864/205101/Determination-of-Molecular-Structures-from-Ground)). An atom at z = 1.5 Å carries a 0.001 Å Costain error; an atom at z = 0.1 Å carries 0.015 Å; an atom within about 0.04 Å of a principal plane has an error larger than its own coordinate, and Kraitchman can return an imaginary coordinate. Supplying those coordinates is a legitimate and specific role for a high-level calculation.

### 2.4 The experimental floor

Chasing 0.2 pm in a computed intermolecular separation is not a category error for **line prediction**, and is one for **structure**. For Ar–oxazole the substitution Ar–ring distance is 3.447 Å against 3.458 Å from the r_0 fit — a 0.01 Å difference between two experimental structural methods, which is the real precision of an experimental structure on a floppy complex ([Kraka, Cremer, Spoerel, Merke, Stahl & Dreizler, *J. Phys. Chem.* 99, 12466 (1995)](https://s3.smu.edu/dedman/catco/publications/pdf/JPhysChem_99_12466_1995.pdf)). The document distinguishes the two uses everywhere.

---

## 3. Required accuracy specification

### 3.0 B_e and B₀ are different specifications, and v3 conflated them

This is the single most consequential clarification in v4. **B_e is the equilibrium rotational constant of a structure on the Born–Oppenheimer surface. It is not an observable.** B₀ is what a microwave experiment measures, and B₀ = B_e + ΔB_vib, where the vibrational correction is **0.1–0.7 % of B_e** for the species in scope ([Puzzarini & Stanton](https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04706c)) — that is, larger than the 0.1 % target the whole exercise is aimed at.

The consequence is a two-line specification rather than one:

- **B_e is reachable to 0.13 % [M]** by composite coupled-cluster schemes for molecules of ≤16 atoms — the ChS (CBS+CV) mean absolute error against semi-experimental references, against fc-CCSD(T)/cc-pVTZ's 0.80 % ([Puzzarini & Stanton](https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04706c)). v3's blanket "0.3–0.5 % semi-rigid" understated what the electronic structure alone can do.
- **B₀ is reachable to 0.3–0.5 % [D] for semi-rigid complexes and 1–2 % [D] for floppy ones**, and the gap between that and the 0.13 % of B_e is ΔB_vib, not the electronic structure. v3's honest statement therefore survives — but for a different reason than v3 gave, and the difference matters operationally: **buying a better equilibrium structure past the composite level does not move B₀ at all until ΔB_vib is bought first.** That is the quantitative justification for the spend priority of §3.3.

Every tier row in this document states which quantity it delivers, in the accuracy cell, as `B_e` or `B₀`. Rows that deliver B_e and are read as though they delivered B₀ are the commonest way this document can be misused.

**A frozen-monomer flag accompanies every geometry row** with values `relaxed`, `frozen-iso` (monomers frozen at isolated-monomer geometries) or `frozen-inc` (frozen at in-complex geometries), because the distinction changes A by about 1 % and is otherwise invisible (§9A.2).

### 3.1 The specification table

This is the binding accuracy specification. Every row of every tier table is checked against it before publication, and no row may claim an accuracy that its stated optimisation thresholds, method error or averaging scheme can support.

| Observable | Required accuracy | Claimable by this matrix | Why the requirement is what it is | Anchor |
|---|---|---|---|---|
| **B_e** absolute, de novo, semi-rigid, composite | 0.1 % | **0.13 % [M]** | ChS (CBS+CV) MAE for ≤16 atoms; CCSD(T)/CBS+CV+fT+fQ reaches 0.04 % | [Puzzarini & Stanton](https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04706c) |
| **B_e** absolute, de novo, floppy vdW | 0.1 % | **0.3–1 % [D]** | dominated by the intermolecular separation: junChS intermolecular parameters average −0.005 Å with a 0.03 Å maximum, i.e. 0.34 % average and 2.0 % maximum in B | [junChS-F12](https://cris.unibo.it/retrieve/handle/11585/868585/ae4939e6-d216-426d-9d79-edb47b92c82c/junChS-F12.pdf) |
| **B₀** absolute, de novo, semi-rigid | **0.1 %** | **0.3–0.5 % [D]** | B_e is better than this; ΔB_vib at 0.1–0.7 % of B_e is the limiting term | [Puzzarini & Stanton](https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04706c) |
| **B₀** absolute, de novo, floppy vdW | 0.1 % | **1–2 % [D]** | best published dimer geometry benchmark is 0.02 Å ≈ 1 % in B; pyrrole–Ne best case 1.54 % | [Glick *et al.* 2025](https://pubs.aip.org/aip/jcp/article/162/17/174106/3345631/Evaluating-wavefunction-methods-the-counterpoise); [CSIC survey](https://digital.csic.es/bitstream/10261/230932/4/Rotational%20spectra%20of%20van%20der%20Waals%20complexes.pdf) |
| A₀, B₀, C₀ semi-experimental (parent measured) | 0.1 % | **0.03–0.06 %, ≤0.1 % [M]** | the vibrational correction tolerates 5–20 % force-constant error once the parent anchors the structure | [Puzzarini & Stanton](https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04706c) |
| **A₀ / A_e for a T-shaped or near-linear complex** | 0.1 % | **<0.2 % [D] with frozen high-level monomers**; ~1.7 % with B3LYP-class monomers | A is set by the monomer geometry and is almost independent of R: a 10 mÅ uniform monomer error is −1.71 % in A and −0.080 % in B | own propagation, §4.5; corroborated by [Lego-brick approach](https://ricerca.sns.it/retrieve/fe3d5821-f41e-48ed-927f-34be686e050b/acs.jpca.1c07828.pdf) |
| Isotopologue shifts Δ(A,B,C) | **0.02 %** | 0.02–0.1 % | geometry error cancels in the shift; obtained by scaling to the experimental parent, not by a better method | [Kisiel, structural programs](http://info.ifpan.edu.pl/~kisiel/struct/struct.htm) |
| Conformer-to-conformer Δ(A,B,C) | 0.1 % | 0.1 % | near-degenerate conformers can be ranked but not certified from A, B, C alone | [conformer table, UVa](https://uvadoc.uva.es/bitstream/handle/10324/6112/TFG-G592.pdf?sequence=1) |
| Binding topology discrimination | **1–3 %** | 1–3 % | distinct topologies differ by ≥0.2 Å in R_cm, i.e. ≥11 % in B | [Gutowsky, Arunan *et al.*](https://pubs.aip.org/aip/jcp/article/103/10/3917/481163/Rotational-spectra-and-structures-of-the-C6H6-HCN) |
| Orientational isomers of one topology | not achievable from A, B, C | use isotopic substitution and the χ tensor | "geometries 1 and 4 could not be distinguished experimentally" | [Kraka *et al.* 1995](https://s3.smu.edu/dedman/catco/publications/pdf/JPhysChem_99_12466_1995.pdf) |
| ΔB_vib | — | 0.05 % of B₀ with good coupled cluster; 0.1 % at 20 % force-constant error | it is the highest accuracy-per-CPU-hour item in the document | [Puzzarini & Stanton](https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04706c) |
| **Frozen-core bias in B_e** | — | **−0.81 % mean, 2.70 % max at cc-pVQZ [M]**; −0.04 % all-electron at cc-pCVQZ | 74-isotopologue, 15-approach benchmark; the cheap fix is fc/CBS(Q,5) + core/cc-pCVTZ at 0.107 % mean absolute | [Bologna 74-isotopologue benchmark](https://cris.unibo.it/bitstream/11585/656295.2/1/Benchmark_paperS.pdf) |
| **BSSE-driven geometry error in B** | — | **+4.1 pm ≈ 2.8 % at B3LYP/cc-pVTZ [M]**, +1.1 pm ≈ 0.76 % at cc-pVDZ-F12 (paired with CABS), +0.2 pm ≈ 0.14 % at aug-cc-pV5Z | counterpoise-optimised versus normally optimised water-dimer O···O distance; BSSE is a first-order error in B, not only in energy | [Plumley & Dannenberg](https://pmc.ncbi.nlm.nih.gov/articles/PMC3073166/) |
| μ_a, μ_b, μ_c signed, in the PAS | **±0.1 D per component** | ±0.1 D is the *requirement*; hybrid DFT does not currently meet it for flexible or weakly bound species (§6.1) | an 0.08 D component decides whether a whole branch exists | [Kisiel *et al.*, camphor](https://pubs.rsc.org/en/content/articlelanding/2003/cp/b212029a); [Fatima *et al.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC9961461/) |
| χ_aa, χ_bb − χ_cc (¹⁴N, Cl, Br, I) | **~5 %** | 5 % from MP2/6-311++G(2d,2p) | fixes the projection angle of a substituent to a few degrees | [Dohmen, Fedosov & Obenchain](https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04067k) |
| χ(D), deuterium quadrupole coupling | 5 % | **<1 % demonstrated**, needing at least a quadruple-zeta core-valence basis plus a vibrational correction | the TZ→6Z basis shift is 9.9 kHz = 6.3 %; the vibrational correction is 1.7 % | [Cazzoli *et al.*, D₂O Lamb-dip](https://hal.science/hal-00604410/document) |
| Quartic centrifugal distortion | **factor of 2** | factor of 2, free with the harmonic force field | keeps the predicted pattern from walking off at high J; yields k_σ and ω_σ | [Fraser *et al.* 1987](https://www.sciencedirect.com/science/article/pii/0022285287900919/pdf) |
| **Sextic centrifugal distortion** | order of magnitude | 3–4 % demonstrated on oxirane — **CFOUR only** | not among ORCA's documented VPT2 outputs | [Puzzarini *et al.*, oxirane](https://pmc.ncbi.nlm.nih.gov/articles/PMC4630858/) |
| Inertial defect Δ = I_c − I_a − I_b = −2P_cc | **correct sign, always**; magnitude to ~0.05 amu Å² for semi-rigid species | as required, at zero marginal cost | an algebraic function of A, B, C that probes the out-of-plane force field directly | [Groner](https://www.sciencedirect.com/science/article/abs/pii/S0022285216300959); [Oka](https://okaionfactory.web.illinois.edu/publications/PDF/oka194.pdf) |
| Planar moments P_aa, P_bb, P_cc | ordering check P_aa > P_bb > P_cc | as required, zero cost | each isolates mass distribution along one axis, whereas A, B, C each mix two | [Groner](https://www.sciencedirect.com/science/article/abs/pii/S0022285216300959) |
| Direct dipolar spin–spin coupling D | pure geometry quantity, D ∝ r⁻³ | as accurate as the geometry, at seconds of cost | a model-free measurement of an internuclear distance in a weakly bound complex | [Andrews *et al.*](https://pubs.aip.org/aip/jcp/article/85/6/3180/219289/Rotational-spectrum-H-19F-spin-spin-and-D-nuclear) |
| Nuclear spin–rotation C_aa, C_bb, C_cc | **~10 %** | ~3 % demonstrated for D₂O — one of the few places computation sits at experimental accuracy | unresolved spin–rotation broadens lines and biases fitted centre frequencies | [Cazzoli *et al.*](https://hal.science/hal-00604410/document) |
| Vibrational satellites B_v, three lowest modes | order of magnitude for intermolecular modes; 0.1 % for intramolecular | as stated, zero marginal cost once α_r exist | satellites are the second-strongest features in a jet spectrum of a complex | [Gordon *et al.*](https://pubs.rsc.org/en/content/articlelanding/2018/cp/c8cp01102h) |
| V₃ internal-rotation barrier | ±10 % | **±14 % [M]** — the only in-domain benchmark, and **no tier may claim tighter** | the A/E splitting depends steeply on V₃; the ammonia–formic acid computed span is 168.3–212.8 cm⁻¹ against 195.18(7) measured | [Roehling *et al.* 2024](https://experts.arizona.edu/en/publications/ammonia-formic-acid-complex-internal-rotation-analysis-calculatio/) |
| Tunnelling splitting | **factor of 3** | **estimate only** — no tier reliably delivers it; report barrier, reduced mass and path | splittings span 6 MHz to 279,650 MHz within one molecule | [Mukhopadhyay, Cole & Saykally](https://escholarship.org/content/qt1j70w8wt/qt1j70w8wt.pdf) |
| Binding energy D₀ | **not an assignment observable** | post-assignment validation only; junChS-F12 A14 MUE **0.06 kJ/mol [M]** | D₀ does not appear in the rotational Hamiltonian | [Puzzarini group composite review](https://pmc.ncbi.nlm.nih.gov/articles/PMC9863398/) |
| Dynamical-averaging absolute constants | — | **not claimed as absolutes** — the salvaged protocol reports ΔB_vib with a factor-2 uncertainty, plus a basin count and a zero-point-omitted flag (§13.4) | classical trajectories at 5 K return essentially B_e | [*Path Integral Methods in Atomistic Modelling*](https://pure.mpg.de/rest/items/item_3702731_1/component/file_3702732/content) |

**Window conversion, stated once and used in every tier legend.** At 12 GHz, ±0.1 % = ±12 MHz; ±0.5 % = ±60 MHz; ±1 % = ±120 MHz; ±2 % = ±240 MHz. For a complex with B + C ≈ 2 GHz the J+1←J transitions scale as (J+1)(B+C), so a fractional error ε displaces a transition near 15 GHz by ε × 15 GHz.

**Provenance discipline for this table.** Every entry carries `[M]`, `[D]` or `[E]`. Two entries in the "claimable" column are `[D]` and therefore, under the standing rule of §17.4, may not be the sole support for an accuracy claim on a specific system: the floppy-vdW B₀ band (1–2 %) and the semi-rigid B₀ band (0.3–0.5 %). **Both require local calibration against the six-system working set of §17 before they are quoted for a new complex**, and §17.5 gives the split-conformal procedure that turns them into a coverage-valid interval.

### 3.2 What 1 % actually buys you

v3's headline was "de novo theory does not reach the 0.1 % assignment threshold." That is true and it was the wrong sentence to lead with, because it reads as a verdict on the enterprise rather than as a specification. The operationally useful inverse is this:

**0.3–1 % plus a documented brute-force triples search is a complete, cheap, published route to an assignment.** Formamidinium formate — a hydrogen-bonded ion pair, squarely in this document's class — was assigned from constants agreeing "within 1 %", with the best functional reproducing A to only 98.3 %, a **1.7 % error on A = 5881.714 MHz**, and the calculated constants "were used in the SPCAT program … to predict the rotational transitions in the 3.7–14.3 GHz range" ([Zhou *et al.*, *JCP* 2019](https://research-repository.st-andrews.ac.uk/bitstream/handle/10023/16997/Zhou_2019_JCP_Synthesismicrowavespectra_AAM.pdf)). Eleven hours of one workstation running AUTOFIT is cheaper than a month of an HPC allocation.

Recent practice agrees. Low-cost DFT continues to assign microsolvated complexes ([Valladolid microsolvation study](https://uvadoc.uva.es/bitstream/handle/10324/76017/Manuscript_corr_final.pdf?sequence=1&isAllowed=y)) and flexible molecules ([proline methyl ester](https://pubs.rsc.org/en/content/articlehtml/2025/cp/d5cp00898k)); the "cheap but accurate" composite route predicts "rotational constants with an accuracy of 0.3 % or better" for glycine and serine ([Barone & Puzzarini](https://pmc.ncbi.nlm.nih.gov/articles/PMC9863398/)).

**What 0.1 % buys that 1 % does not** is a search window narrow enough to identify a J-manifold without a triples search, and the ability to distinguish two conformers whose constants differ by less than 1 %. If neither is your problem, the 12 h tier is the end of the road and the 1 mo tier is a waste.

**What no accuracy buys** is a completeness proof over binding topologies. That comes from hand enumeration (§9B.1), not from compute.

### 3.3 Spend priority (binding)

Under a fixed compute budget, for the assignment use case, spend in this order.

1. **Geometry, to the best affordable level** — and specifically the *intermolecular* separation, which carries roughly two orders of magnitude more of the error in B and C than the monomer geometry does (§4.5). Consumes the majority of any tier's budget.
2. **ΔB_vib from a cheap anharmonic force field** on that geometry. Mandatory at every tier above 30 minutes. "A cheap anharmonic force field on top of a good geometry buys more accuracy per CPU-hour than any further improvement of the equilibrium structure" ([Puzzarini & Stanton](https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04706c)).
3. **Freeze high-level monomers** — essentially free, and worth about 1.2 percentage points in A (§4.5, §9A.2).
4. **Quartic centrifugal distortion** — free, from the harmonic force field already computed in step 2. Never a separate budget line.
5. **Inertial defect and planar moments** — free, three subtractions on A, B, C.
6. **Dipole components μ_a, μ_b, μ_c, signed, in the principal axis system** — minutes. Determines which transitions exist at all.
7. **Nuclear quadrupole coupling χ_aa, χ_bb − χ_cc** — hours. The decisive discriminator between orientational isomers that A, B, C cannot separate.
8. **V₃ and internal rotation** — days; ±14 % at best; report with the reduced mass and the path.
9. **Tunnelling splittings** — factor of 3; report barrier, reduced mass and path, and flag as an estimate.
10. **D₀** — post-assignment validation only.

The 1-week and 1-month tiers are accordingly repurposed from "a better D₀" to **"a better geometry plus a fitted or Δ-learned potential surface for the large-amplitude coordinates"**, which is the only expenditure at that scale that improves an assignment. Active learning plus Δ-learning has moved that campaign from 26–43 days to roughly one day on the workstation (§18, Table 2), which is why the PES rows have been re-tiered downward rather than up.

---

## 4. Error propagation: the geometry–constant relation

### 4.1 The identity

For a complex treated as a pseudo-diatomic of reduced mass μ and separation R,

\[
B = \frac{h}{8\pi^2\mu R^2} = \frac{505379.1\ \mathrm{MHz\cdot u\cdot \AA^2}}{\mu R^2},
\qquad \frac{\Delta B}{B} = -2\,\frac{\Delta R}{R}.
\]

This single relation is the most useful sentence the conference produced and it belongs at the top of any working copy of this document. The conversion constant is [A(MHz)][I_a(amu Å²)] = 505 379.005(36) ([Groner](https://www.sciencedirect.com/science/article/abs/pii/S0022285216300959); [NIST CCCBDB moment-of-inertia conversion](https://cccbdb.nist.gov/convertmomintx.asp)).

### 4.2 Worked propagation

| ΔR | Ar–HCl (μ 18.93, R 4.00 Å, B 1669 MHz) | Ar–H₂O (μ 13.6, R 3.63 Å, B 2820 MHz) | benzene–Ar (μ 32.4, R 3.58 Å, B 1217 MHz) | water dimer (μ 9.0, R 2.91 Å, B 6631 MHz) |
|---|---|---|---|---|
| 0.001 Å | 0.83 MHz (0.05 %) | 1.55 MHz (0.055 %) | 0.68 MHz (0.056 %) | 4.56 MHz (0.069 %) |
| **0.01 Å** | **8.34 MHz (0.50 %)** | **15.5 MHz (0.55 %)** | **6.80 MHz (0.56 %)** | **45.6 MHz (0.69 %)** |
| 0.02 Å | 16.7 MHz (1.00 %) | 31.1 MHz (1.10 %) | 13.6 MHz (1.12 %) | 91.2 MHz (1.38 %) |
| 0.034 Å | 28.4 MHz (1.70 %) | 52.8 MHz (1.87 %) | 23.1 MHz (1.90 %) | 155 MHz (2.34 %) |
| 0.056 Å | 46.7 MHz (2.80 %) | 87.0 MHz (3.09 %) | 38.1 MHz (3.13 %) | 255 MHz (3.85 %) |
| 0.14 Å | 117 MHz (7.00 %) | 218 MHz (7.71 %) | 95.2 MHz (7.82 %) | 638 MHz (9.62 %) |

Inverting: **to hit 0.1 % in B at R = 4 Å you need R to 2.0 mÅ, i.e. 0.20 pm**; to hit 0.02 % you need 0.04 pm. For comparison, a 0.001 Å bond-length error produces about 0.1 % in B at r ≈ 2 Å and about 0.2 % at r ≈ 1 Å, and composite coupled-cluster schemes reaching 0.001 Å and 0.1° still move A by up to ±50 MHz and B, C by up to ±15 MHz ([Alessandrini & Puzzarini](https://cris.unibo.it/retrieve/handle/11585/868614/e1dcb339-596b-7715-e053-1705fe0a6cc9/acs.jpca.1c07828_templaggio.pdf)).

### 4.3 The threshold arithmetic, and why v2's Table 5 was internally inconsistent

A residual gradient component *g* on a coordinate with force constant *k* leaves the geometry displaced by Δr ≈ g/k. ORCA's optimisation thresholds are ([ORCA 6.1 manual, geometry optimizations](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations.html); [MPI mirror](https://orca-manual.mpi-muelheim.mpg.de/contents/structurereactivity/optimizations.html)):

| Level | TolE (a.u.) | TolRMSG | TolMaxG | TolRMSD | TolMaxD |
|---|---|---|---|---|---|
| `!LooseOpt` | 3e-5 | 5e-4 | 2e-3 | 7e-3 | 1e-2 |
| `!Opt` (default) | 5e-6 | 1e-4 | 3e-4 | 2e-3 | 4e-3 |
| `!TightOpt` | 1e-6 | 3e-5 | 1e-4 | 6e-4 | 1e-3 |
| `!VeryTightOpt` | 2e-7 | 8e-6 | 3e-5 | 1e-4 | 2e-4 |

With 1 mdyn Å⁻¹ = 100 N m⁻¹ = 6.423 × 10⁻² Eh bohr⁻²:

| Intermolecular stretch k | k (Eh bohr⁻²) | Δr at `!Opt` (g = 3e-4) | Δr at `!TightOpt` (1e-4) | Δr at `!VeryTightOpt` (3e-5) |
|---|---|---|---|---|
| 0.05 mdyn Å⁻¹ | 3.2e-3 | 4.9 pm | 1.6 pm | 0.49 pm |
| **0.069 mdyn Å⁻¹** (H₂CO–HCl, [Fraser *et al.* 1987](https://www.sciencedirect.com/science/article/pii/0022285287900919/pdf)) | 4.4e-3 | **3.6 pm** | 1.2 pm | **0.36 pm** |
| 0.20 mdyn Å⁻¹ (stiff hydrogen bond) | 1.3e-2 | 1.2 pm | 0.4 pm | 0.12 pm |
| 5.0 mdyn Å⁻¹ (C–C covalent) | 0.32 | 0.09 pm | 0.03 pm | 0.01 pm |

Converting through ΔB/B = 2Δr/R at R ≈ 3.5 Å:

- `!Opt` on a typical weak complex: 3.6 pm → **ΔB/B ≈ 2.1 %**, worse than this document's own screening class.
- `!TightOpt`: 1.2 pm → **0.69 %**.
- `!VeryTightOpt`: 0.36 pm → **0.21 %** — still about twice the 0.1 % target.
- On a covalent coordinate even `!Opt` gives 0.09 pm. **The default thresholds are entirely adequate for covalent bonds and inadequate for intermolecular ones by roughly a factor of 40 in the resulting error in B.**

**Version 2 specified `VeryTightOpt` throughout its microwave table and then claimed 0.05–0.1 % accuracy classes. Those two statements cannot both hold, and the accuracy claims are withdrawn.** The field already knows this: the S66 authors used "tight optimization limits (energy change 3 × 10⁻⁴ kcal/mol (5 × 10⁻⁷ au), max. gradient component 0.06 kcal/mol/Å (5 × 10⁻⁵ au), root-mean-square (RMS) gradient 0.03 kcal/mol/Å (2.5 × 10⁻⁵ au)) … **to ensure good convergence, even in the intermolecular degrees of freedom**" ([Řezáč, Riley & Hobza](https://pmc.ncbi.nlm.nih.gov/articles/PMC3152974/)).

### 4.4 The corrected `%geom` block (mandatory for any row claiming ≤0.5 % in B)

```
%geom
  TolE     1e-7      # below VeryTightOpt
  TolRMSG  3e-6
  TolMaxG  1e-5      # 3x tighter than VeryTightOpt -> ~0.12 pm at k = 0.069 mdyn/A
  TolRMSD  5e-5
  TolMaxD  1e-4
end
```

with `TightSCF` or `VeryTightSCF` mandatory, so that numerical gradient noise sits below `TolMaxG`, and `DEFGRID3` for any row that also computes a Hessian. **Report the final maximum gradient component and the softest computed force constant in the output record**, so that a reader can redo this arithmetic on the actual system.

Two caveats, stated against the recommendation itself. First, driving the optimiser to `TolMaxG` 1e-5 on a soft mode may fail to converge if the SCF or grid noise floor sits above that, which is why the convergence gate, the grid gate and the SCF gate in Section 16 are one coupled gate rather than three independent ones. Second, an MLFF geometry optimisation to `TolMaxG` = 1e-5 Eh bohr⁻¹ is below the reported float32 energy precision of about 4 × 10⁻⁶ Eh for AIMNet2 ([`oet_aimnet2` readme](https://raw.githubusercontent.com/faccts/orca-external-tools/main/readmes/aimnet2.md)); **an intermolecular geometry cannot be converged to spectroscopic tightness on a float32 GPU machine-learned potential**, and every MLFF geometry in this document is a pre-optimisation only.

For `xtb` pre-optimisations the analogous levels are `vtight` (Econv 1e-7 Eh, Gconv 2e-4 Eh α⁻¹) or `extreme` (5e-8, 5e-5); `normal` (5e-6, 1e-3) is far too loose for intermolecular coordinates ([xtb documentation, geometry optimization](https://xtb-docs.readthedocs.io/en/latest/optimization.html)).

---

### 4.5 Which coordinate carries the error: monomer geometry versus intermolecular separation

v3 propagated only ΔR. That is half the picture, and the missing half changes what a composite scheme should buy.

Rigid-rotor moments recomputed exactly, with no linearisation, using CONV = 505 379.0 MHz·u·Å² ([Groner](https://www.sciencedirect.com/science/article/abs/pii/S0022285216300959)). **CO₂···H₂O, T-shaped, 6 atoms, R = 2.836 Å, A/B/C = 11 433.66 / 4 621.35 / 3 341.41 MHz** (own computation, `mm/conference2/a7/propagate.py`; all values `[D]`):

| Perturbation | ΔA/A % | ΔB/B % | ΔC/C % | ΔB (MHz) |
|---|---|---|---|---|
| all monomer bonds +0.001 Å | −0.173 | −0.008 | −0.053 | −0.37 |
| +0.003 Å (fc-CCSD(T)/VTZ MAD) | −0.518 | −0.024 | −0.160 | −1.11 |
| +0.006 Å (r²SCAN-3c class) | −1.032 | −0.048 | −0.319 | −2.23 |
| +0.010 Å (B3LYP class) | **−1.711** | −0.080 | −0.532 | −3.72 |
| +0.020 Å (poor double-zeta DFT) | −3.379 | −0.161 | −1.063 | −7.44 |
| **R +0.002 Å** | 0.000 | −0.135 | −0.098 | −6.26 |
| **R +0.005 Å** | 0.000 | −0.338 | −0.245 | −15.63 |
| **R +0.010 Å** | 0.000 | −0.675 | −0.489 | −31.18 |
| **R +0.020 Å** | 0.000 | **−1.343** | −0.974 | −62.05 |
| HOH angle +0.5° | −0.018 | +0.024 | +0.023 | +1.12 |

CH₄···H₂O (8 atoms, R = 3.70 Å) reproduces the pattern: a +0.010 Å monomer error gives ΔB = −0.093 %, a +0.010 Å error in R gives −0.513 %.

**Break-even on B, CO₂···H₂O.** ΔR = 0.002 Å is equivalent to a **16.8 mÅ** uniform error in every monomer bond; 0.005 Å ≡ 41.9 mÅ; 0.010 Å ≡ 83.1 mÅ; 0.020 Å ≡ 163.5 mÅ. **No electronic-structure method errs by 83 mÅ on a covalent bond.** fc-CCSD(T)/cc-pVTZ has a mean absolute deviation of 0.003 Å against CCSD(T)/CBS over 122 species ([Fortenberry *et al.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10537648/)); r²SCAN-3c has a covalent MAD of 0.6 pm ([r²SCAN-3c](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60c752f6bb8c1a21633dbf6c/original/r2scan-3c-an-efficient-swiss-army-knife-composite-electronic-structure-method.pdf)).

**The statement this document now carries, in these words:**

> *For a weak complex, monomer geometry error is the dominant error channel in **A** and a minor one in **B** and **C**; the intermolecular separation is the dominant channel in **B** and **C** and essentially absent from **A**. A composite that fixes one and not the other fixes one constant and not the others.*

Corroboration from an independent group, on the sensitivity scale: "variations of 0.001 Å in the bond distances and 0.1° in valence angles can lead to changes up to **±50 MHz for the A constant and ±15 MHz for B and C**" ([Alessandrini & Puzzarini, Lego-brick](https://ricerca.sns.it/retrieve/fe3d5821-f41e-48ed-927f-34be686e050b/acs.jpca.1c07828.pdf)). For the *intramolecular* case the measured derivative is −0.10 % per +0.001 Å at r ≈ 2 Å (SiS) rising to −0.22 % at r ≈ 1 Å (HF) ([Puzzarini & Stanton](https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04706c)) — which is exactly the regime the A constant of a T-shaped complex sits in.

**What this buys, in money terms.** Upgrading monomers from B3LYP-class to CCSD(T)-class buys **1.2 percentage points in A** and only **0.056 pp (2.6 MHz) in B**. Tightening R from a 0.020 Å error to a 0.005 Å error buys **1.0 pp (46 MHz) in B**. Both are worth having; only one is decisive. This is the arithmetic behind recipe R2 in §9A.6 and behind the reordered spend priority of §3.3.

### 4.6 The measured Δr → ΔB sensitivity, replacing v3's derived table

v3's §4.3 obtained Δr = g/k from an assumed force constant and reported the implied ΔB. That derivation is retained below as `[D]`, but it is now calibrated against measured derivatives from all-electron CCSD(T)/cc-pCVQZ equilibrium structures ([Puzzarini & Stanton](https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04706c)), all `[M]`:

| Molecule | r_e (Å) | B_e (MHz) | ΔB for +0.001 Å | Relative |
|---|---|---|---|---|
| SiS | 1.9316 | 9 077.97 | −9.40 MHz | −0.10 % |
| PN | 1.4913 | 23 563.42 | — | −0.13 % |
| CN | 1.1674 | 57 386.18 | — | −0.17 % |
| CO | 1.1289 | 57 841.66 | — | −0.18 % |
| HCl | 1.2736 | 318 069.32 | −498.90 MHz | −0.16 % |
| OH | 0.9689 | 567 822.08 | −1 170.29 MHz | −0.21 % |
| HF | 0.9158 | 629 664.95 | −1 372.92 MHz | −0.22 % |
| H₂O (A_e/B_e/C_e) | r(OH) 0.9584 | 805 164.3 / 441 462.4 / 285 129.3 | → 803 486.8 / 440 542.6 / 284 535.2 | −0.21 % each |

"An accuracy of 0.001 Å on the bond distance leads to a deviation of 0.1 % on the rotational constant for bond lengths of about 2 Å … and of 0.2 % for distances of about 1 Å"; "to meet the 0.1 % accuracy … an accuracy of 0.0005–0.001 Å is required for bond lengths" ([Puzzarini & Stanton](https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04706c)). v3's derived VeryTightOpt figure (0.36 pm ⇒ 0.21 %) lands close to these numbers, but the agreement is coincidental: v3's ratio came from an assumed force constant and these are measured derivatives. Both are printed; the derived one is labelled `[D]` and is not used to gate anything.

### 4.7 BSSE is a first-order error in the geometry, not only in the energy

v3 treated basis-set superposition error as an energetics issue. It is a *geometry* issue, and at double-zeta-plus-polarisation it is the single largest error in B — larger than functional choice. Counterpoise-optimised versus normally optimised water-dimer O···O distances ([Plumley & Dannenberg](https://pmc.ncbi.nlm.nih.gov/articles/PMC3073166/)), all `[M]`, with the implied ΔB/B = 2ΔR/R marked `[D]`:

| Method / basis | Normal R(O···O) | CP-OPT R(O···O) | ΔR | Implied ΔB/B |
|---|---|---|---|---|
| B3LYP/aug-cc-pV5Z | 2.919 Å | 2.921 Å | +0.2 pm | **0.14 %** |
| B3LYP/cc-pVDZ-F12 (paired with CABS) | 2.912 | 2.923 | +1.1 pm | **0.76 %** |
| B3LYP/cc-pVTZ | 2.909 | 2.950 | +4.1 pm | **2.8 %** |
| B3LYP/6-311G(d,p) | 2.886 | 2.978 | +9.2 pm | **6.3 %** |
| B97D/cc-pVTZ | 2.917 | 2.971 | +5.4 pm | **3.7 %** |
| B2PLYP/6-311++G(d,p) | 2.899 | 2.958 | +5.9 pm | **4.1 %** |

The same work reports that M05 and M06 with 6-311G(d,p) give *qualitatively wrong* structures (2.792 and 2.758 Å): "Several of the smaller basis sets lead to qualitatively incorrect geometries when optimized on a normal potential energy surface… This problem disappears when the optimization is performed on a counterpoise corrected PES", and "all anomalous geometries encountered disappear, indicating that these anomalies are due to BSSE" ([Plumley & Dannenberg](https://pmc.ncbi.nlm.nih.gov/articles/PMC3073166/)).

**Binding consequence.** Any row using a non-augmented triple-zeta basis or smaller, without `BSSEOptimization.cmp`, has its accuracy in B **capped at 3 %**. This is not in conflict with the finding that counterpoise "tends to degrade the quality of the optimized geometries" at double zeta ([Glick, Kumawat & Sherrill](https://pubs.aip.org/aip/jcp/article/162/17/174106/3345631/Evaluating-wavefunction-methods-the-counterpoise)): that result concerns energy-optimal counterpoise at double zeta, whereas the table above concerns structures at triple zeta and above. The operating rule remains: **do not counterpoise-correct the optimisation below triple zeta; do above it; always counterpoise the energy and report raw, CP and half-CP.**

One consequence for the dominated-row list: v3's §15.1 declared the diffuse-triple-zeta geometry row dominated because "diffuse functions change the intermolecular energy more than the intermolecular distance … the constants barely move." That is contradicted — going from cc-pVTZ to cc-pVDZ-F12 (paired with CABS) changes the CP-versus-uncorrected discrepancy from 4.1 pm to 1.1 pm, a ~2 % versus ~0.8 % effect on B. **The row is removed from the dominated list.**

### 4.8 Frozen core is a systematic bias, not a rounding error

Fifteen-approach benchmark, 9 closed- and 6 open-shell molecules, 74 isotopologues; relative errors in equilibrium rotational constants ([Bologna 74-isotopologue benchmark](https://cris.unibo.it/bitstream/11585/656295.2/1/Benchmark_paperS.pdf)), all `[M]`:

| Approach | Mean (%) | Mean abs (%) | Std (%) | Max (%) |
|---|---|---|---|---|
| fc-CCSD(T)/cc-pVTZ | −1.538 | 1.543 | 0.765 | 4.819 |
| fc-CCSD(T)/cc-pVQZ | **−0.806** | 0.812 | 0.423 | 2.701 |
| fc-CCSD(T)/cc-pV5Z | −0.444 | 0.466 | 0.208 | 1.583 |
| fc/CBS(T,Q) | −0.094 | 0.345 | 0.501 | 2.684 |
| fc/CBS(Q,5) | −0.276 | 0.312 | 0.308 | 0.643 |
| fc/cc-pV5Z + core/cc-pCVTZ | −0.118 | 0.183 | 0.192 | 0.787 |
| **fc/CBS(Q,5) + core/cc-pCVTZ** | 0.051 | **0.107** | 0.191 | 1.049 |
| all-electron CCSD(T)/cc-pCVQZ | **−0.037** | 0.164 | 0.206 | 0.874 |

"Going from the former to the latter, the standard deviation halves and the mean error decreases from −0.81 % to −0.04 %." Note also the counter-intuitive result that with extrapolation "the cc-pCVTZ basis set interestingly provides slightly better mean errors, which is somewhat unexpected" — the core correction does not need a large core-valence basis when combined with CBS extrapolation, which is a real cost saving. For deuterium quadrupole coupling the requirement is stricter: "a basis set as large as a core-valence quintuple-zeta set is required" ([acetylene semi-experimental structure study](https://pubmed.ncbi.nlm.nih.gov/21322673/)).

**Binding consequence.** −0.81 % at quadruple zeta is larger than every target interval above the 12 h tier. **No row may claim ≤0.5 % in B_e with `fc` and no core correction.** The named cheap recipe is **fc/CBS(Q,5) + core/cc-pCVTZ**, at 0.107 % mean absolute error.

---

## 5. Corrected working equations, counts and constants

### 5.1 Vibrationally averaged constants: average the inverse inertia tensor

The vibrationally averaged rotational constant is **one half the expectation value of the inverse effective inertia tensor**, not a function of the expectation value of the inertia tensor. In the Eckart–Watson formulation, with μ_αβ the generalised inverse inertia tensor,

\[
\mu_{\alpha\beta} = \left(I'^{-1}\right)_{\alpha\beta},\qquad
I'_{R\gamma} = I_{R\gamma} - \sum_{k,l,m}\zeta^{km}_R\,\zeta^{lm}_\gamma\,Q_k Q_l ,
\]
\[
A_V \simeq \tfrac12\langle\mu_{xx}\rangle_V,\qquad
B_V \simeq \tfrac12\langle\mu_{yy}\rangle_V,\qquad
C_V \simeq \tfrac12\langle\mu_{zz}\rangle_V ,
\]

as written explicitly by [Czakó, Mátyus and Császár, *JPCA* 113, 11665 (2009)](https://www2.sci.u-szeged.hu/czako/papers/JPCA_H2O_113_11665_2009.pdf). In MHz, ⟨μ_αα⟩ is evaluated with B[MHz] = 505379.0 / I[amu Å²]. **The inverse is taken before the average, element-wise on the 3×3 tensor, after Eckart alignment to a single reference, and ⟨μ⟩ is then diagonalised.** Any script that accumulates ⟨I⟩ and inverts it afterwards is wrong.

**Size of the bias.** By Jensen's inequality ⟨1/I⟩ ≥ 1/⟨I⟩, so averaging the wrong quantity always underestimates the constant, and the leading fractional bias is 3σ_R²/R₀². For a floppy intermolecular stretch with reduced mass 10 amu and mean separation 3.80 Å:

| σ_R (Å) | B from ⟨1/I⟩ (MHz) | B from 1/⟨I⟩ (MHz) | bias (MHz) | bias (%) |
|---:|---:|---:|---:|---:|
| 0.05 | 3501.69 | 3499.26 | 2.43 | 0.069 |
| 0.10 | 3507.18 | 3497.46 | 9.72 | 0.278 |
| **0.15** | **3516.38** | **3494.45** | **21.93** | **0.628** |
| 0.20 | 3529.40 | 3490.25 | 39.16 | 1.122 |
| 0.30 | 3567.50 | 3478.27 | 89.24 | 2.566 |

*Why 1/⟨I⟩ is not an acceptable approximation.* A typical van der Waals stretch zero-point amplitude of σ_R ≈ 0.15 Å produces a **0.63 % / 22 MHz systematic error — six times the 0.1 % target — purely from averaging the wrong quantity.** (Referee's own calculation on the model above; script retained at `mm/conference/ref/jensen.py`.)

**Standing caveat, to be repeated wherever averaged constants meet experiment.** Czakó, Mátyus and Császár state that "this route does not lead to constants that can be compared directly with experimental effective spectroscopic constants but provides a theoretically sound starting point for such comparisons", and for Ar–CO₂ found A and B from the averaging route in excellent agreement with eigenvalue fitting while C deviated ([Czakó *et al.*](https://www2.sci.u-szeged.hu/czako/papers/JPCA_H2O_113_11665_2009.pdf)). Averaged constants must not be fitted directly against a `.par` file without stating the approximation, and **C is flagged as the least reliable of the three** whenever this route is used.

A second caveat cuts against the fix as well: the Czakó formulation defines μ in the Eckart frame with the Coriolis ζ corrections included, and drops the off-diagonal ⟨μ_αβ⟩ and the Coriolis term ⟨π̂_α μ_αβ⟩. For a complex with a genuinely large-amplitude internal rotation that term is not small, and the resulting "rotational constants" are effective parameters that may not map one-to-one onto the A, B, C an experimentalist fits from a tunnelling-split spectrum.

### 5.2 The semi-experimental relation

\[
B_e^{SE} = B_0^{exp} - \Delta B_{vib}^{calc},\qquad
\Delta B_i^{vib} = -\tfrac12\sum_r \alpha_i^r ,\qquad
B_v = B_e - \sum_r \alpha_r\left(v_r + \tfrac{d_r}{2}\right).
\]

The third form is printed because it is the free deliverable of Section 6.5: the same α_r set that produces ΔB_vib also produces the vibrational satellites, and version 2 computed them, used one linear combination and discarded the rest.

### 5.3 Hessian and displacement counts

ORCA's semi-numerical VPT2 "will compute the Hessian and then generate two displaced geometries for each degree of freedom and for each displacement another Hessian … will be computed", with default `AnharmDisp 0.05`, and `%freq Delq 0.5` documented as "the displacement in dimensionless coordinates used during the VPT2", which establishes that the displacements are along **normal modes**, not Cartesian coordinates ([ORCA 6.1 manual, VPT2](https://www.faccts.de/docs/orca/6.1/manual/contents/spectroscopyproperties/vpt2.html)). Hence

\[
n_{\text{Hess}}^{\text{VPT2}} = 2(3N-6) + 1 = 6N - 11 \quad (= \mathbf{49}\ \text{at}\ N = 10);
\qquad 6N-9 \ \text{for a linear molecule}.
\]

The count 61 = 6N + 1 that appeared in version 2 and in one position paper is the count for two-sided *Cartesian* displacements, which is not what ORCA does; independent confirmation of 6N − 11 comes from a reported Gaussian VPT2 job needing 205 Hessians for 36 atoms, exactly 6 × 36 − 11 ([CCL mailing list, 6 July 2016](https://server.ccl.net/chemistry/resources/messages/2016/07/06.003-dir/index.html)). **Any tier whose budget was derived from 61 Hessians was over-costed by 24 % and has been re-derived.**

Numerical Hessian counts, from the `%freq` documentation (`CentralDiff true` default, `DX/Increment 0.005`, and "`NumHessTransInvar` reduces the number of gradients that need to be calculated by 6 (for two-sided differentiation) or 3 (for one-sided differentiation) for numeric Hessians, but has no effect for analytic Hessians") ([ORCA 6.1 manual, frequencies](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/frequencies.html)):

| Route | Formula | N = 10 |
|---|---|---:|
| Numerical Hessian from **gradients**, two-sided (default) | 6N | 60 |
| … with `NumHessTransInvar` | 6N − 6 | 54 |
| Numerical Hessian from gradients, one-sided | 3N | 30 |
| … with `NumHessTransInvar` | 3N − 3 | 27 |
| Numerical Hessian from **energies**, central differences | 6N + 2·3N(3N−1) + 1 | **1801** |

The 1801-point energy-only route is the reason energy-only Hessians are never scheduled below the 1-day tier. **ORCA VPT2 requires analytic Hessians**, so no `ExtOpt`, MLFF, double-hybrid, DLPNO or F12 tier may claim ORCA VPT2; those tiers get numerical Hessians and a finite-difference anharmonic treatment, or nothing. Where a coupled-cluster anharmonic force field is genuinely wanted, the answer is CFOUR, not a larger ORCA allocation (Section 9.2).

### 5.4 Path-integral bead counts

\[
P > \frac{\hbar\omega_{\max}}{k_BT};\qquad
\text{use } P = \lceil 2.2\,\beta\hbar\omega_{\max}\rceil;\qquad
k_BT/hc = T/1.438777\ \mathrm{cm^{-1}} .
\]

| T (K) | k_BT/hc (cm⁻¹) | ω_max = 3000 cm⁻¹ | ω_max = 600 cm⁻¹ | ω_max = 200 cm⁻¹ |
|---:|---:|---:|---:|---:|
| 2 | 1.390 | P > 2158 | P > 432 | P > 144 |
| 5 | 3.475 | **P > 863** | P > 173 | P > 58 |
| 30 | 20.85 | P > 144 | P > 29 | P > 10 |
| **50** | **34.75** | P > 86 | **P > 18 (use 40 with the 2.2 factor)** | P > 6 (use 14) |
| 100 | 69.50 | P > 43 | P > 9 | P > 3 |

The minimum-bead criterion and the worked room-temperature examples (rigid water, librations to ~1000 cm⁻¹, needs P > 5, in practice 6; flexible water, to ~4000 cm⁻¹, needs P > 20, in practice 32) are from [*Path Integral Methods in Atomistic Modelling*, Eq. 3.28](https://pure.mpg.de/rest/items/item_3702731_1/component/file_3702732/content), which also warns that "for properties that involve computing fluctuations … the number of replicas required for convergence is typically considerably higher (by a factor of 2 or more)", and quantifies the error: at Pk_BT/ħω = 1 the energy error is 10 % and the heat-capacity error 100 %, falling to 2 % and 38 % at Pk_BT/ħω = 2. The ×2.2 practical factor used above is that recommendation. The minimum bound n_min ≈ βħω_max is the Markland–Manolopoulos rule as quoted in the [RPMD review, CJCP](https://cjcp.ustc.edu.cn/hxwlxb/cn/article/pdf/preview/10.1063/1674-0068/cjcp1808186.pdf), with 1 cm⁻¹ ≡ 1.438777 K from the [NIST CODATA second radiation constant](https://physics.nist.gov/cgi-bin/cuu/Value?c22ndrc).

**Both of the bead counts argued at the conference are arithmetically correct and they do not conflict.** 863 is the minimum bound at 5 K with ω_max = 3000 cm⁻¹; roughly 4800 is a practically converged count at 2 K obtained by scaling the recommended flexible-water P = 32 with the ×2.2 safety factor. Neither is affordable at ab initio cost, which is why full-dimensional path-integral dynamics at jet temperature is deleted from this document.

### 5.5 PNO-space extrapolation

\[
E_{\mathrm{CORR,CPS}} = E^X + F\,(E^Y - E^X),\qquad F = 1.5\ \text{fixed},\qquad Y = X+1 .
\]

The ORCA tutorial states that the "Optimal value of F was found as 1.5 on a wide range of diverse interactions in the GMTKN55 superset, and thus it should NOT be changed", that the TCutPNO exponents "must be consecutive (e.g. 6/7, 7/8, etc.)", and that "For CPS extrapolation, it is recommended to keep all other DLPNO thresholds the same" ([ORCA 6.1 extrapolation tutorial](https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extrapol.html)). F is optimal at 1.5 ± 0.2 ([Altun, Neese & Bistoni](https://pmc.ncbi.nlm.nih.gov/articles/PMC7586325/)).

**The LAM document's ladder "TCutPNO 1e-5 → 1e-7" is invalid because the exponents are not consecutive, and it skips a decade outside the regime in which F = 1.5 was fitted.** Use CPS(5/6) = 1e-5/1e-6 or CPS(6/7) = 1e-6/1e-7. Accuracy gain on S66: TightPNO 0.20 → CPS(6/7) 0.11 → CPS(7/8) 0.08 → CPS(8/9) 0.06 kcal/mol ([Altun *et al.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7586325/)).

A note on what this extrapolation certifies. CPS is calibrated against tighter PNO calculations, not against canonical CCSD(T); the PNO ladder (LoosePNO outliers to 1.3, NormalPNO below 0.8, TightPNO below 0.3 kcal/mol — [Herbert, *JCP* 161, 054114](https://www.asc.ohio-state.edu/herbert.44/reprints/JCP_161_054114.pdf)) is a **convergence** measure, not an **accuracy** measure, and the tables say so.

### 5.6 Basis-set extrapolation syntax

`! Extrapolate(X/Y,basis)` is valid syntax, but the automatic-extrapolation section of the ORCA 6.1 manual carries the note "This functionality is deprecated - it may still be usable but we will not actively maintain this part of code anymore. For basis set extrapolation please use the respective compound scripts" ([ORCA 6.1 MDCI manual](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/mdci.html); [CompoundScripts EnergyExtrapolation](https://github.com/ORCAQuantumChemistry/CompoundScripts/tree/main/EnergyExtrapolation)). **Every extrapolation cell in this document uses the compound scripts**, with the deprecation recorded here rather than in a footnote a reader will skip. ORCA extrapolates SCF and correlation separately, with fixed β = 2.4 for the 2/3 pair and β = 3 for 3/4 and higher, and optimised α throughout; the tabulated exponents are α(2,3)/β(2,3) = 4.42/2.46 and α(3,4)/β(3,4) = 5.46/3.05 for cc-pVnZ. Independent fits on S66-type data give α = 3.22 for {T,Q} and 3.00 for {Q,5} ([Kesharwani *et al.*, arXiv:2111.01882](https://arxiv.org/pdf/2111.01882)).

### 5.7 Counterpoise

`%geom Counterpoise` is not an ORCA keyword. CP-corrected geometry optimisation is driven by the `BSSEOptimization.cmp` compound script; the manual states that "one should NOT simply add `!Opt` to the above input files, but should instead use the dedicated compound script" ([ORCA 6.1 counterpoise manual](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/counterpoise.html); [BSSEOptimization.cmp](https://github.com/ORCAQuantumChemistry/CompoundScripts/blob/main/GeometryOptimization/BSSEOptimization.cmp)). The manual's worked water-dimer example at MP2/cc-pVTZ gives ΔE_dim = −6.07, ΔE_BB-CP = +1.67, ΔE_corrected = −4.40 kcal/mol.

Policy, applied everywhere:

1. Apply full Boys–Bernardi CP to the single-point energy and report the raw value, the CP-corrected value and their difference as a basis-quality diagnostic. On S66, CP reduces the MAE from about 0.7 to about 0.2 kcal/mol at aug-cc-pVDZ and to 0.1 or below at cc-pVDZ-F12 (paired with CABS), at roughly 2.5× the cost ([Herbert, *JCTC* 18, 6742](https://www.asc.ohio-state.edu/herbert.44/reprints/JCTC_18_6742.pdf)).
2. Half-CP is used as a documented convergence bracket. On S66x8 half-CP generally beats full CP, and full CP degrades hydrogen bonds while improving the other subsets ([Kesharwani, Martin *et al.*, *PCCP* 2022](https://pubs.rsc.org/en/content/articlehtml/2022/cp/d2cp03938a)). Recommendation: full CP at double zeta, half-CP at triple zeta and above, CP-free with F12.
3. **Do not counterpoise-correct the optimisation below triple zeta.** "For double-zeta basis sets, the counterpoise correction tends to degrade the quality of the optimized geometries, regardless of the method used" ([Glick, Kumawat & Sherrill 2025](https://pubs.aip.org/aip/jcp/article/162/17/174106/3345631/Evaluating-wavefunction-methods-the-counterpoise)).
4. **The magnitude by which CP-relaxed optimisation lengthens R could not be obtained at the requested precision from a primary source and is `n.a.`** The sign is unambiguous. The measurement that would settle it: a CP-corrected and an uncorrected optimisation of the same dimer at the same triple-zeta level with the corrected `%geom` block of Section 4.4, reporting ΔR_cm in pm.
5. **Never treat gCP and Boys–Bernardi CP as the same object.** gCP is an atom-pairwise geometry-only semi-empirical estimate of the BSSE energy with typically 10–30 % error on the BSSE itself; Boys–Bernardi CP is the ghost-orbital calculation. r²SCAN-3c already carries D4 + gCP; ωB97X-3c carries D4 and **no** gCP, its developers stating that "small residual BSSE effects are efficiently absorbed by the D4 damping scheme" ([ωB97X-3c README](https://github.com/grimme-lab/wB97X-3c/blob/main/README.md)). Never add either to a composite that already contains one.
6. **Do not rely on cancellation between diffuse-function and counterpoise contributions.** For N₂O⋯CO the two "were not additive; did not compensate each other; had almost the same value but opposite signs" ([Demaison *et al.* 2021](https://pubs.aip.org/aip/jcp/article/154/19/194302/565922/How-accurate-is-the-determination-of-equilibrium)).

### 5.8 Cost constants

- **MPQC CCSD(T)-F12/TightPNO/cc-pVDZ-F12 (paired with CABS), ~10 atoms, 8–16 cores: 5–40 min, plan on 15 min, uncertainty ×/÷ 3, measure locally before scheduling.** Published anchors, all MPQC CCSD(T)-F12 with the improved (T): diclofenac/def2-SVP (329 basis functions) 10.2 min on 4 cores; def2-TZVP (667 BF) 71.8 min; def2-QZVPP (1439 BF) 1203 min; penicillin 430 BF → 10.3 min, 858 BF → 64.5 min, 1921 BF → 614.7 min; vancomycin 1797 BF → 147.7 min ([Guo, Riplinger, Becker, Liakos, Minenkov, Cavallo & Neese, *JCP* 148, 011101 (2018)](https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/1.5011798/13764376/011101_1_online.pdf)). A 10-atom complex in cc-pVDZ-F12 (paired with CABS) is about 380–450 basis functions. TightPNO tightens `T_CutPNO` from 3.33e-7 to 1e-7, `T_CutPairs` to 1e-5 and `T_CutMKN` to 1e-4, typically a factor of 2–4 for small systems. An independent anchor of 63 atoms / 2203 AOs / 16 cores / 7.7 h for LNO-CCSD(T) is consistent ([Nagy *et al.*, *Chem. Sci.*](https://real.mtak.hu/205919/1/state-of-the-art-local-correlation-methods-enable-affordable-gold-standard-quantum-chemistry-for-up-to-hundreds-of-atoms.pdf)). Estimates near 30 s per point are excluded: the SCF plus RI-MP2 guess alone exceeds that.
- **Consequence: at 15 min per point a 12-hour tier buys about 48 points on 16 cores**, not thousands. This alone removes any tier that promised a MPQC CCSD(T)-F12 surface scan below the one-week mark. Every tier cell states its uncertainty as `15 min (×/÷ 3)`, and a **mandatory local calibration** — one single point on the actual complex, wall time recorded, all tier point counts rescaled — precedes any use of these tiers as a schedule.
- **`ExtOpt` numerical gradients multiply the gradient cost by 6N, i.e. 60× at N = 10.**
- **Hardware constants** are given in full in Section 8: i7-13700K = 8 P-cores + 8 E-cores, 24 threads, AVX2 only, DDR5 officially 5600 with 6400 an XMP setting, 102.4 GB/s dual channel at 6400; run **8 MPI ranks on the P-cores with `%maxcore 3000`**. RTX 3090 = 10,496 CUDA cores, 24 GB, 936 GB/s, 35.6 TFLOPS FP32, FP64 = 35.6/64 = **0.556 TFLOPS**, against roughly 0.61–0.64 TFLOPS AVX2 FP64 from the eight P-cores — **the GPU is no faster than its host CPU in double precision.**

---

## 6. Observables the previous versions underweighted

Version 2's microwave table scored B_e and B₀ and nothing else. Eight observables that change what an experimentalist does are added here, five of them at zero or near-zero marginal cost.

### 6.1 Dipole components μ_a, μ_b, μ_c — signed, in the principal axis system

**Requirement: ±0.1 D absolute on each component, signed, in the principal axis system of the same geometry that produced A, B, C.** Not the magnitude. The question an experimentalist asks is binary: does this species have an observable b-type or c-type spectrum? Camphor has |μ_c| = 0.0804(7) D ([Kisiel *et al.*, *PCCP* 2003](https://pubs.rsc.org/en/content/articlelanding/2003/cp/b212029a)); Ar–ketene has μ_a = 0.125(3) D against μ_b = 1.369(2) D ([NIST / Gillies *et al.*, Ar–ketene](https://www.nist.gov/publications/rotational-spectra-structure-internal-dynamics-and-electric-dipole-moment-argon-0)). For Ar–oxazole the search was built on an assumed μ_a = 0, μ_b = 1.34, μ_c = 0.66 D, "which should cause a b- and C-type spectrum instead of an a- and b-type spectrum", and indeed "all measured lines are of b- or c-type" and "a-type lines were not found" ([Kraka *et al.* 1995](https://s3.smu.edu/dedman/catco/publications/pdf/JPhysChem_99_12466_1995.pdf)). That prediction was worth more than another decimal place on B.

Intensity scales as μ², so a 30 % component error is a factor-1.7 intensity error, well inside the factor 0.3–3 intensity scatter that automated tools already tolerate ([RAARR, arXiv:1812.06221](https://arxiv.org/pdf/1812.06221.pdf)). Relative accuracy is not the constraint; absolute accuracy is.

**The document does not claim that hybrid DFT meets the ±0.1 D specification.** Statistically, hybrid functionals give regularised RMS errors of 5–6 % and double hybrids 3.6–4.5 %, comparable with CCSD's 4 %, against CCSD(T)/CBS references ([Hait & Head-Gordon, arXiv:1709.05075](https://arxiv.org/abs/1709.05075)). Against jet Stark measurements the per-component errors are larger: for n-butanol *TGt* the cam-B3LYP/6-311++G(d,p) values μ_a 0.89, μ_b 1.05, μ_c 0.94, μ_tot 1.67 D compare with measured 0.7137(10), 0.8989(8), 0.8071(8), 1.4032(11) D — a **0.27 D error in μ_tot and 0.15–0.18 D per component**; for *GTg′* the computed 2.01 D against a measured 1.6532(16) D is a **0.36 D error** ([Fatima *et al.*, electric dipole moments from Stark effect in supersonic expansion](https://pmc.ncbi.nlm.nih.gov/articles/PMC9961461/)). **So the ±0.1 D requirement stands, no cheap method currently meets it per component for flexible or weakly bound species, and the practical use of a computed dipole is the presence or absence of a branch, not a quantitative intensity.**

The measurement protocol matters for the comparison. Experimental dipoles come from second-order Stark shifts of individual M-components in a calibrated field, per component and per transition, and jet measurements are far more precise than room-temperature ones: n-propanol *Ga* gives μ_a = 0.32(6) D from a room-temperature spectrum against 0.4914(4) D from supersonic expansion, and the *Aa* conformer gives μ_a = 0.3589(7), μ_b = 1.2820(13), μ_c = 0 by symmetry, μ_tot = 1.3312(13) D from 37 lines at σ_fit = 2.29 kHz ([Fatima *et al.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC9961461/)). Components can sometimes be extracted from a-type transitions alone. **A row comparing a computed dipole to "the experimental dipole" must state which kind of Stark measurement it is comparing against.**

**Dark conformers.** Any conformer with |μ_α| < 0.1 D on an axis is flagged **dark** in that selection rule and is excluded from the predicted line list while being retained in the thermodynamic ensemble. A conformer that is spectroscopically dark is not a search target regardless of its energy.

### 6.2 Nuclear quadrupole coupling χ_aa, χ_bb − χ_cc

For any ¹⁴N, ³⁵/³⁷Cl, ⁷⁹/⁸¹Br or ¹²⁷I in the complex, hyperfine structure is the strongest topology discriminator available, because χ is a tensor and therefore reports the **orientation** of the nucleus in the principal axis frame, which A, B, C do not. Magnitudes: Ar–oxazole χ_aa = 2.3032(6), χ_bb = −4.0526(8), χ_cc = 1.7494(4) MHz ([Kraka *et al.* 1995](https://s3.smu.edu/dedman/catco/publications/pdf/JPhysChem_99_12466_1995.pdf)); H₂CO–H³⁵Cl eQq_aa = −41.424(14), eQq_bb = 14.106(19), eQq_cc = 27.318(19) MHz, with the ³⁷Cl isotopologue at −32.678(11) MHz ([Fraser, Gillies, Zozom, Lovas & Suenram 1987](https://www.sciencedirect.com/science/article/pii/0022285287900919/pdf)); the ¹⁴N coupling projected on the figure axis is −4.223(4) MHz in C₆H₆–HCN against −1.143(2) MHz in Ar₃–HCN ([Gutowsky, Arunan *et al.* 1995](https://pubs.aip.org/aip/jcp/article/103/10/3917/481163/Rotational-spectra-and-structures-of-the-C6H6-HCN)).

Target 5 %. A 20-complex benchmark settles the method: "MP2/6-311++G(2d,2p) as an ab initio method outperformed all DFT methods when requiring both reliable geometry and NQCC predictions with a single calculation"; B1LYP/TZV(3df,2p) is best for χ_zz, CCSD(T)/cc-pVQZ best for (χ_xx − χ_yy), and B3LYP-D3/def2-TZVPD or B3LYP-D4/def2-TZVPD best for rotational constants ([Dohmen, Fedosov & Obenchain, *PCCP* 2023](https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04067k)). Two warnings from the same work are carried into the tables: "simply using Bailey's Qeff determined from monomers for complexes is a fallacy" — their complex-derived eQ_eff/h(³⁵Cl) = −17.3 ± 1.4 MHz a.u.⁻¹ against Bailey's monomer value of −19.166 ± 0.021 — and "flexible systems with large amplitude motions or energetically close conformers require computational treatment of these phenomena to improve the NQCC predictions".

### 6.3 Deuterium quadrupole coupling χ(D)

Isotopic substitution with deuterium is the workhorse of complex structure determination, and every deuteration introduces a new hyperfine pattern that must be predicted or the spectrum will not fit. For D₂¹⁶O, χ_aa(D) = 152.55(88) kHz experimental against 153.7 kHz computed, and (χ_bb − χ_cc)(D) = 198.03(76) against 198.6 kHz, at a fit σ of 0.63 kHz — agreement below 1 %, far better than the ~5 % typical of ¹⁴N ([Cazzoli *et al.*](https://hal.science/hal-00604410/document)).

That accuracy is earned and the document states what it costs. The CCSD(T) equilibrium χ_aa converges 166.277 (aug-cc-pCVTZ) → 157.806 (pCVQZ) → 156.813 (pCV5Z) → 156.373 kHz (pCV6Z), and the vibrational correction runs −3.262 → −2.711 → −2.689 kHz over the same series ([Cazzoli *et al.*](https://hal.science/hal-00604410/document)). **The TZ-to-6Z basis effect is 9.9 kHz, 6.3 %, and the vibrational correction is 1.7 %: a double-zeta χ(D) is worthless.** In complexes the magnitudes are similar and the information is directional — SO₂⋯DF gives χ_cc = 227(3) kHz and (χ_aa − χ_bb) = 68(13) kHz, used to confirm the hydrogen-bond geometry ([Andrews *et al.*, *JCP* 85, 3180](https://pubs.aip.org/aip/jcp/article/85/6/3180/219289/Rotational-spectrum-H-19F-spin-spin-and-D-nuclear)).

### 6.4 The inertial defect Δ = I_c − I_a − I_b = −2P_cc, and the planar moments

**This is the headline free output of the document.** It costs nothing: every method row already produces A, B, C, and Δ is three subtractions using [A(MHz)][I_a(amu Å²)] = 505 379.005(36) ([Groner](https://www.sciencedirect.com/science/article/abs/pii/S0022285216300959); [Oka](https://okaionfactory.web.illinois.edu/publications/PDF/oka194.pdf)).

Four reasons it is promoted from absent to headline.

1. **It is a differential quantity**, so systematic geometry error largely cancels — the concrete instance of the Product C principle.
2. **It is diagnostically sharp where A, B, C are blunt.** Planar species cluster near zero: formamide isotopologues span +0.018 to −0.022 amu Å², phenol sits at −0.032, malonaldehyde at +0.1026. Genuine non-planarity is unmistakable: 2-fluoronitrobenzene has Δ = −17.8349(16) amu Å², "clearly indicating that it is non-planar (32°)" ([Oka](https://okaionfactory.web.illinois.edu/publications/PDF/oka194.pdf)).
3. **It measures the low-frequency out-of-plane force field** — the quantity this document elsewhere concedes is hard to compute. Oka reports the empirical semi-classical relation between Δ₀ and the lowest out-of-plane wavenumber, with h/π²c = 134.861 amu Å² cm⁻¹, and for nitrosobenzene the inertial defect "decreases drastically by ΔA_t ≃ −0.82 for each excitation of the vibration" ([Oka](https://okaionfactory.web.illinois.edu/publications/PDF/oka194.pdf)). Δ read off successive vibrational satellites is therefore a direct experimental measurement of the intermolecular mode.
4. **It has a predictive literature for the floppy aromatic class**, in the form of an empirical formula for inertial defects of aromatic ring systems derived from measured negative defects attributed to low-energy out-of-plane zero-point vibrations ([McNaughton *et al.*, *PCCP* 2017](https://pubs.rsc.org/en/content/articlelanding/2017/cp/c6cp07487a)). For complexes it works directly: formamidinium formate has Δ = −0.243 amu Å² "consistent with a planar structure", and the tropolone–formic acid complex Δ = −1.46 amu Å² ([Zhou *et al.*](https://research-repository.st-andrews.ac.uk/bitstream/handle/10023/16997/Zhou_2019_JCP_Synthesismicrowavespectra_AAM.pdf)).

**Validation rule, binding: if the computed and experimental inertial defects differ in *sign*, the structure is wrong, regardless of how well A, B and C agree.**

The planar moments P_aa = Σm_i a_i², P_bb, P_cc are reported on the same row at the same zero cost. P_aa "measures the extension of masses along the molecule's a axis (or out of the bc plane)", with the ordering P_aa > P_bb > P_cc, and second moments are "often the simplest parameters to interpret" because each isolates mass distribution along one axis ([Groner](https://www.sciencedirect.com/science/article/abs/pii/S0022285216300959)). For a rare-gas complex the change in P_cc on complexation says immediately whether the rare gas sits in a symmetry plane.

### 6.5 Vibrational satellites and excited-state constants

Real jet spectra of complexes are full of satellites, and for a van der Waals complex the low-frequency intermolecular modes are thermally populated even in a jet, so satellites are the second-strongest features in the spectrum, not exotica. In a microwave taxonomy study of C₂S/C₃S/C₄S, "more than 25 new vibrationally excited states" were detected across a survey of ">60 distinct chemical species", with assignment relying on "published vibration–rotation interaction constants for C₃S, or newly calculated ones for C₂S and C₄S", producing catalogues for radio-astronomy searches in the 253–280 GHz range ([Gordon *et al.*, *PCCP* 2018](https://pubs.rsc.org/en/content/articlelanding/2018/cp/c8cp01102h)).

**The α_r are already computed for ΔB_vib. Reporting B_v for v = 1 of the three lowest modes, and Δ for each, is free.** The caveat that must be printed alongside: the α_r for intermolecular modes are the ones VPT2 gets wrong, so these satellites carry a much larger error bar than the ground state and are flagged as order-of-magnitude guides, mode by mode.

### 6.6 Direct dipolar spin–spin coupling D

A pure geometry quantity, D ∝ 1/r³, with no electronic structure in it at all, and therefore a direct model-free measurement of an internuclear distance in a weakly bound complex. For SO₂⋯HF, D_HF = −190(2) kHz and (ᵃD_HF − ᵇD_HF) = −57(4) kHz, used together with the quadrupole constants to confirm a planar *cis* hydrogen bond with r(O⋯F) = 2.818(8) Å ([Andrews, Taleb-Bendiab, LaBarge, Hillig & Kuczkowski, *JCP* 85, 3180](https://pubs.aip.org/aip/jcp/article/85/6/3180/219289/Rotational-spectrum-H-19F-spin-spin-and-D-nuclear)). For D₂O the D–D coupling is D_aa = −1.57(41) kHz against −1.61 computed ([Cazzoli *et al.*](https://hal.science/hal-00604410/document)).

**This is the highest information-per-CPU-second entry in the document.** Cost: seconds. Deliverable: D_gg in kHz for every spin-½ pair from the trial geometry, with pairs above |D| = 50 kHz flagged as resolvable.

### 6.7 Nuclear spin–rotation C_aa, C_bb, C_cc

Reported, but not targeted. For D₂¹⁶O, C_aa(D) = −2.89(26) kHz experimental against −2.98 computed, C_bb −2.43(18) against −2.41, C_cc −2.62(15) against −2.62, from CCSD(T)/aug-cc-pCV6Z equilibrium values plus CCSD(T)/aug-cc-pCV5Z vibrational corrections ([Cazzoli *et al.*](https://hal.science/hal-00604410/document)). **Theory is at experimental accuracy here — the only observable in this document of which that is true** — and the document says so. It matters practically because unresolved spin–rotation broadens lines and biases fitted centre frequencies, and because spin–rotation is how deuterated species get their hyperfine patterns right. Target 10 %; cost, hours for a property job on an existing geometry.

### 6.8 Internal rotation, V₃ and A/E splittings

A methyl rotor splits every line into A and E components, doubling the line count and defeating naive pattern matching unless it is anticipated. V₃ is obtained from a relaxed torsional scan on a 10° grid with all other coordinates optimised, fitted to V = Σ_n (V_3n/2)(1 − cos 3nτ), and combined with the internal-rotation constant F (F₀ ≈ 158–161 GHz for a methyl top; 159.808(80) GHz fitted for 2-methylthiophene) to form the reduced barrier s = 4V₃/(9F).

Achievable accuracy, from paired experiment and computation:

| System | Experimental V₃ | Computed V₃ | Deviation |
|---|---|---|---|
| 2-methylthiophene | 197.7324(18) cm⁻¹ (XIAM) | MP2/6-311++G(d,p) 183.02 cm⁻¹ | −7.4 % |
| 2-methylthiophene | 197.7324(18) cm⁻¹ | B3LYP-D3BJ/6-311++G(d,p) 236.31 cm⁻¹ | +19.5 % |
| *cis*-m-methylanisole | 55.7693(90) cm⁻¹ (aixPAM Fit III) | B3LYP/6-311++G(d,p) 50.128 cm⁻¹ | −10.1 % |
| *trans*-m-methylanisole | 36.6342(84) cm⁻¹ | B3LYP/6-311++G(d,p) 32.768 cm⁻¹ | −10.6 % |
| NH₃⋯HCOOH | 195.18(7) cm⁻¹ | calculated values span 168.3–212.8 cm⁻¹ | ±14 % |

Values from the [2-methylthiophene study, *Spectroscopy Journal* 1(1):5](https://www.mdpi.com/2813-446X/1/1/5), the [m-methylanisole study](https://hal.science/hal-03183072/document) and [Roehling *et al.* 2024](https://experts.arizona.edu/en/publications/ammonia-formic-acid-complex-internal-rotation-analysis-calculatio/); the 2-methylthiophene authors summarise the position fairly as "many levels of theory yield values with less than 10 cm⁻¹ deviation".

**Verdict: ±10 cm⁻¹ or roughly 10–14 % is what a cheap DFT scan buys.** That is ample for identifying which torsional regime you are in and for starting a fit, and it is not enough to predict a splitting pattern blind, because near-degenerate low barriers amplify a 10 % error in V₃ into an order-of-magnitude error in the splitting. Anchors across the regime: V₃ = 11.21745(2) cm⁻¹ for a very low barrier fitted with XIAM ([Obenchain *et al.*](https://bib-pubdb1.desy.de/record/454603/files/The%20low%20barrier%20methyl%20internal%20rotation%20in%20the_Obenchain_submission.pdf)); 49.374548(1) cm⁻¹ for p-methylanisole ([HAL hal-03183097](https://hal.science/hal-03183097v1/document)); about 439.15 cm⁻¹ for the two equivalent tops of 2,5-dimethylfuran ([HAL hal-03183074](https://hal.science/hal-03183074/document)); and at V₃ = 565.1(5) cm⁻¹, splittings of only about 51 MHz ([Semantic Scholar PDF](https://pdfs.semanticscholar.org/0be3/8225f41fd1700efaa6b69c19ef4e763d2784.pdf)).

**Protocol: compute V₃ from a relaxed scan at the best affordable level, convert to a predicted splitting, then fit V₃ to the observed splittings with XIAM (or BELGI/RAM36 if s is small).** The computed number's job is to start the fit and to distinguish conformers, not to predict the spectrum.

### 6.9 Tunnelling splittings — the honest entry

**No tier of this matrix reliably delivers a tunnelling splitting.** The required accuracy is only a factor of 3, which sounds weak and is not routinely met. The dynamic range is the reason: within the water dimer alone the largest 1→4 tunnelling splitting is −70,128.436 ± 0.10 MHz, the ground-state acceptor-switching splitting is 279,650 MHz for (H₂O)₂ and 53,000 MHz for (D₂O)₂, the K = 0 donor–acceptor interchange splitting is 19,526.73 MHz for (H₂O)₂ against 1,172.23 MHz for (D₂O)₂, and the K = 0 donor tunnelling is 6 MHz ([Mukhopadhyay, Cole & Saykally, *Chem. Phys. Lett.* 633, 13 (2015)](https://escholarship.org/content/qt1j70w8wt/qt1j70w8wt.pdf)) — nine orders of magnitude in one molecule.

**Deliverable for every row: the barrier, the reduced mass, the tunnelling path, and the estimated splitting explicitly flagged as an estimate.** Two routes do better than that and both are expensive. Full-dimensional variational vibration–rotation–tunnelling treatment on a well-fitted surface reproduces the observed splittings: a 2025 water-dimer two-body surface with an RMS fitting error of 0.70 cm⁻¹ gave levels agreeing with all experimental terahertz origins and tunnelling splittings to within 0.44 cm⁻¹ ([Wang, Yang, Carrington & Zhang, *J. Chem. Phys.* 163, 144308 (2025)](https://pubmed.ncbi.nlm.nih.gov/41070798/); companion surface reducing discrepancies "from about 1 cm⁻¹ to 0.36 cm⁻¹", [PubMed 41081418](https://pubmed.ncbi.nlm.nih.gov/41081418/)). Path-integral methods reproduce tunnelling matrix elements well — 3.0(1) cm⁻¹ for the water-dimer acceptor pathway against a reference of 3.0, and 26(2) cm⁻¹ for the trimer flip against an experimental 21.76 cm⁻¹, both better than instanton theory on the same surface ([Tunnelling splittings in water clusters from PIMD, EPFL Infoscience](https://infoscience.epfl.ch/server/api/core/bitstreams/cc42831e-5aa1-461b-8e00-0d72817f4a4d/content)).

Note also that a tunnelling splitting is a statement about a delocalised state, so **no molecular-dynamics averaging protocol can produce one, even in principle** — standard path-integral MD recovers quantum statistics, not quantum tunnelling dynamics, and needs instanton or ring-polymer-instanton machinery for the splitting itself.

### 6.10 Isotopologue shifts and campaign planning

For natural-abundance work theory is not asked for absolute isotopologue constants; it is asked for the **shift**, which is predicted far better because geometry error largely cancels. Mass differences: ¹²C→¹³C is 1.0033548 u, ³⁵Cl→³⁷Cl is 1.9970499 u, ⁷⁹Br→⁸¹Br is 1.9979535 u ([Kisiel, structural programs](http://info.ifpan.edu.pl/~kisiel/struct/struct.htm)). **The workflow is to scale a trial geometry so that it reproduces the experimental parent A, B, C, then predict the isotopologue spectrum from that scaled geometry — which is what the EVAL program does** ([Kisiel](http://info.ifpan.edu.pl/~kisiel/struct/struct.htm)). Target 0.02 % on the shift; cost, seconds once a geometry exists. A dedicated program exists for fitting the rare-gas coordinate of a rare-gas complex by treating it as an added point mass.

Two warnings on the scaling route. Empirical rotational-constant scaling factors reduce rRMSD from 1.0 % to 0.3 % for linear molecules and from 1.4 % to 0.4 % for isotopologues (MAD 110 → 26 MHz), but on the HCl(H₂O) hydrogen-bonded set only from **7.9 % to 6.8 %** ([Vogt *et al.*, *Molecules* 29, 5874](https://www.mdpi.com/1420-3049/29/24/5874)). Where the error is a uniform systematic bond-length bias a differential treatment removes it almost entirely; where the error is in the intermolecular coordinate itself it does not. Scaling factors calibrated on covalent molecules do not transfer to weakly bound complexes, and this document does not claim they do.

---

## 7. Nuclear spin statistics and permutation-inversion symmetry

Getting the molecular symmetry group wrong means predicting lines that do not exist, failing to predict lines that do, and getting the relative intensities wrong — which is what makes a crowded CP-FTMW spectrum unassignable.

The relevant object is the molecular symmetry (MS) group in the sense of [Longuet-Higgins, *Mol. Phys.* 6, 445 (1963)](https://ui.adsabs.harvard.edu/abs/1963MolPh...6..445L/abstract): "the set of (i) all feasible permutations of the positions and spins of identical nuclei and (ii) all feasible permutation-inversions". Longuet-Higgins states the payoff directly — by the representations of this group "one can classify not only the spin states and states of motions of the nuclei, but even the electronic states", with examples that "illustrate the use of this concept in determining the statistical weights of individual levels and selection rules". Bunker traces the lineage from Hougen's full point group to the permutation-inversion group and stresses that the MS group "applies to all molecules, rigid or nonrigid" ([Bunker, *The Molecular Symmetry Group: a personal view*](https://d197for5662m48.cloudfront.net/documents/publicationstatus/231737/preprint_pdf/3347adf8e8b88c35e3b80cce4047b4fd.pdf)).

**Feasibility is a dynamics judgement, not a software output.** Whether a permutation is feasible depends on whether the corresponding barrier is surmountable on the experimental timescale. The groups can be large: the water pentamer's MS group is the wreath product S₅[S₂], with **3840 operations** ([Wales and co-workers, OSTI UCRL-JRNL-202191](https://www.osti.gov/servlets/purl/15013980); published in [*J. Phys. Chem. A*](https://pubs.acs.org/doi/10.1021/jp049955k)). The subtleties of non-rigid group theory remain live literature ([arXiv:1704.02697](https://arxiv.org/pdf/1704.02697)); worked spin-weight derivations for methylamine via permutation-inversion are available ([Periodica Polytechnica](https://pp.bme.hu/ch/article/download/2806/1911/6564)), as is a clean pedagogical treatment ([UWO, *Spin statistics*, chapter 8](https://physics.uwo.ca/~mhoude2/courses/astro9701/Spin_statistics.pdf)).

**Two software corrections, both binding.**

- **`molsym` does point groups, not molecular symmetry groups.** Its repository advertises point-group detection, symmetry-element generation, character-table generation and SALC generation, and mentions neither permutation-inversion groups, nor non-rigid molecular symmetry groups, nor nuclear spin statistics ([NASymmetry/MolSym](https://github.com/NASymmetry/MolSym); [Goodlett and Kitzmiller, *J. Chem. Phys.* 161, 024107 (2024)](https://pubs.aip.org/aip/jcp/article/161/2/024107/3302915/MolSym-A-Python-package-for-handling-symmetry-in)). It is a good tool for rigid-molecule point groups and the wrong tool for a tunnelling dimer. The LAM document's proposal to obtain spin statistics by querying `molsym` is deleted.
- **PGOPHER requires user-supplied weights.** Statistical weights "must include equivalent spins for correct absolute intensities", though a constant factor from non-equivalent spins may be omitted, affecting S, S_pol and partition functions but not Einstein A coefficients or absorption coefficients ([PGOPHER paper](https://pgopher.chm.bris.ac.uk/Help/PGOPHERaccepted.pdf)). The canonical illustration is H₂'s 3:1 ortho:para alternation, which "must be set up in the statistical weights". PGOPHER's hyperfine machinery uses F₁ = J + I₁, F₂ = F₁ + I₂, with an alternative coupling I₁₂ = I₁ + I₂, F = J + I₁₂ selected by `AsNext` for equivalent pairs; **three or more equivalent nuclei are not implemented** in the version described. Its documentation warns that "only nuclei not explicitly simulated with nucleus objects should be included in calculating the SymWt and AsymWt statistical weights" ([PGOPHER linear-molecule nucleus help](https://pgopher.chm.bris.ac.uk/Help/linearnucleus.htm)), and there are documented limits on symmetric-top spin handling ([PGOPHER release notes](https://pgopher.chm.bris.ac.uk/download/old/9.0.101/Help/bugs.htm)).

**Required manual step in every workflow that involves equivalent nuclei:** determine the MS group from the set of feasible permutation-inversion operations, derive the spin statistical weights by hand, enter them into PGOPHER or SPCAT explicitly, and record the feasibility argument in the output. No package in this document's stack automates this.

**Consequence for Eckart alignment.** The Eckart conditions fix the body-fixed frame by requiring the vibrational displacements to carry zero net linear and angular momentum relative to a reference structure ([Eckart conditions](https://en.wikipedia.org/wiki/Eckart_conditions)), and they "can only be formulated for a semi-rigid molecule" ([chemeurope, Eckart conditions](https://www.chemeurope.com/en/encyclopedia/Eckart_conditions.html)). For a complex tunnelling among *n* permutationally equivalent minima there are *n* equally valid references and the rotation matrix becomes discontinuous as a trajectory crosses between them. Multi-reference Eckart formulations exist for exactly this situation ([Numerical and exact kinetic energy operator using Eckart conditions with one or several reference geometries: HONO](https://dugi-doc.udg.edu/bitstream/handle/10256/16593/026355.pdf?sequence=1&isAllowed=y)), but no shipped tool in this document's software stack implements one. **Binding gate: count the equivalent minima. If more than one is sampled, either restrict the ensemble to one symmetry-distinct basin, align to that basin's reference, and report a single-well average with an explicit note that tunnelling contributions are excluded, or mark the averaged constants `n.a.` Single-reference Eckart alignment must not be applied silently across equivalent minima.**

---

## 8. Hardware, and the routing decision procedure

### 8.0 Setup numbering, fixed

The position papers disagreed about which setup is called Setup 1. **This document adopts the common brief's numbering and applies it throughout:**

| Label | Machine | Role |
|---|---|---|
| **Setup 1** | GitHub Actions, GitHub Classroom (50 seats), GitHub Codespaces. CPU only. | Teaching |
| **Setup 2** | Workstation: Intel Core i7-13700K, NVIDIA RTX 3090, 64 GB DDR5, NVMe | Production |
| **Setup 3** | HPC cluster, multi-node, batch scheduler | Reference campaigns |

The tier tables carry a **"Setup 1 feasible?"** column, which is a teaching-deployment flag: `Yes`, `Yes (at risk)`, or `n.a.` with the reason. A reader coming from version 2, where the workstation was Setup 1, should re-read every setup reference in this document rather than assume continuity.

### 8.1 Setup 2 — the workstation, in detail

**Verified CPU specification.** Intel's datasheet gives 16 total cores = **8 Performance-cores + 8 Efficient-cores**, 24 threads, P-core base 3.40 GHz / turbo 5.30 GHz, E-core base 2.50 GHz / turbo 4.20 GHz, 30 MB L3 Smart Cache and 24 MB total L2 (2 MB per P-core, 4 MB per E-core module), base power 125 W and maximum turbo power 253 W, two memory channels, DDR5 up to **5600 MT/s**, maximum memory bandwidth **89.6 GB/s**, and instruction-set extensions **SSE4.1, SSE4.2, AVX2** ([Intel Core i7-13700K datasheet](https://media.distrelec.com/Web/Downloads/_t/ds/BX8071513700K_eng_tds.pdf); corroborated by [TechPowerUp](https://www.techpowerup.com/cpu-specs/core-i7-13700k.c2850)). **DDR5-6400 is an XMP setting outside Intel's validated envelope**; 6400 MT/s × 8 bytes × 2 channels = **102.4 GB/s**, a 14 % uplift over the supported 5600.

**AVX-512 is fused off on this generation.** Intel states that AVX-512 "will be fused off on Alder Lake mobile products and most desktop products … Intel plans to fuse off AVX-512 on Alder Lake products going forward" ([Intel support article 000089918](https://www.intel.com/content/www/us/en/support/articles/000089918/processors.html); corroborated by [Tom's Hardware](https://www.tomshardware.com/news/intel-reportedly-kills-avx-512-alder-lake-cpus)), and the 13700K datasheet's instruction list confirms the outcome. The widely deployed ORCA module at Texas A&M is `ORCA/6.1.1-avx2` ([Texas A&M HPRC](https://hprc.tamu.edu/kb/Software/ORCA/)), so this box is on the same instruction-set tier as a much older server part, with only clock and cache in its favour.

**Run 8 MPI ranks bound to the P-cores. Do not run 24.** Three independent lines of evidence support this. Intel's own oneMKL guidance for hybrid parts recommends "running threads on the P-cores only", suggests `KMP_HW_SUBSET=8c:intel_core`, notes that "for higher performance, Intel Hyper-Threading Technology on P-cores must be disabled", and explains that with statically load-balanced work "E-cores will take longer to complete the work items assigned to them" ([Intel oneMKL Developer Guide](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2023-0/managing-performance-with-heterogeneous-cores.html)). Open MPI's maintainers document the topology and state bluntly that "these processors were not designed for HPC", recommending an explicit `pe-list` selecting only P-cores ([Open MPI issue #11345](https://github.com/open-mpi/ompi/issues/11345)). Field reports on commercial MPI solvers describe the OS scheduler parking work on E-cores with "significantly longer solve times" ([Maya HTT knowledge base](https://help.mayahtt.com/kb/topics/how_to_resolve_intel_mpi_performance_issues_on_windows_with_hybrid_cpu_architectures.html)), and Intel concedes that "some legacy software can experience performance inversions as core counts increase" ([PCWorld, quoting Intel](https://www.pcworld.com/article/545013/intel-alder-lake-to-offer-8-p-core-only-model-and-have-avx512-too.html)).

The mechanism is specific to ORCA. ORCA distributes SCF integral batches, DFT grid batches and DLPNO pair lists across ranks and **synchronises every iteration**, so a synchronous iteration costs the time of the slowest rank. Hyper-threaded P-core siblings share one set of AVX2 FMA ports, so oversubscribing them degrades a floating-point-bound code rather than helping.

Canonical invocation:

```
! ... PAL8            # or  %pal nprocs 8 end
%maxcore 3000         # MB, PER RANK
```
```bash
export OMP_NUM_THREADS=1
export KMP_HW_SUBSET=8c:intel_core,1t     # P-cores only, one thread each
/full/path/to/orca job.inp > job.out       # never launched with mpirun
```

ORCA must not be started with `mpirun` and must be called with its full path ([ORCA Input Library, setting up ORCA](https://sites.google.com/site/orcainputlibrary/setting-up-orca)). Texas A&M's guidance is that "ORCA scales best when using up to 16 cores" and recommends no more than 16 ([Texas A&M HPRC](https://hprc.tamu.edu/kb/Software/ORCA/)), measured on homogeneous server cores, so 16 is an upper bound rather than a target on a hybrid desktop part; local correlation scales worse, with "appreciable" speedup only "up to 1–2 dozen processor cores" ([Nagy *et al.*, *Chem. Sci.*](https://real.mtak.hu/205919/1/state-of-the-art-local-correlation-methods-enable-affordable-gold-standard-quantum-chemistry-for-up-to-hundreds-of-atoms.pdf)).

**`%maxcore` is per rank, not per job.** The manual defines `%maxcore 8192` as "8 Gb of memory per processor" ([ORCA 6.1 frequencies](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/frequencies.html)). With 64 GB physical and 8 ranks the arithmetic ceiling is 8 GB per rank, but ORCA routinely exceeds `%maxcore` by 20–50 % for correlated modules and the OS plus scratch page cache need headroom. **`%maxcore 3000` allocates 24 GB and leaves about 40 GB free.** Raise to 6000 only for a single-rank job; never set ranks × maxcore above half of physical RAM. Note the compounding trap: going from 8 to 16 ranks halves the memory each MDCI rank can use, pushing `InCore AUTO` down a level and moving integrals from RAM to disk, so on a two-channel box doubling ranks can make a coupled-cluster job slower.

For DLPNO on a single node, `%mdci StorageType Shared` lets one shared copy of frequently used integrals serve all ranks and falls back to disk if memory is short; it works only when all processes are on one node ([ORCA 6.1 MDCI](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/mdci.html)), which is Setup 2's situation. **A RAM disk is the wrong answer on 64 GB:** `%maxcore 3000` × 8 already commits 24 GB, and concurrent DLPNO scratch is estimated at 40–160 GB, so a tmpfs large enough to matter would have to be carved out of memory the calculation needs.

**Memory bandwidth is the binding constraint for correlated work.** The eight P-cores at AVX2 can issue roughly 0.61–0.64 TFLOPS FP64 (8 cores × 2 FMA ports × 4 doubles per 256-bit vector × 2 flops × ≈4.8–5.0 GHz). At 102.4 GB/s that is a machine balance of about 0.16 bytes per FLOP: any kernel achieving fewer than about 6 FLOPs per byte moved is bandwidth-limited. Compute-bound on this box: two-electron integral evaluation, and large dense multiplications inside a canonical CCSD ladder when the blocks fit in the 30 MB L3. Bandwidth-bound: the four-index integral transformation, DFT exchange–correlation grid evaluation, DLPNO pair-list and PNO-construction phases, and the chain-of-spheres exchange step. Adding ranks adds no bandwidth; two channels is two channels.

**The one exception where all sixteen cores earn their keep.** Embarrassingly parallel work with no synchronisation — GOAT structure batches, QCxMS trajectory ensembles, NumGrad displacement sweeps, independent PES grid points, conformer single points — **should be run as 16 concurrent single-rank jobs, not as one 16-rank job.** This is the single most useful hardware optimisation available on Setup 2 and it appears as an explicit note on every numerically differentiated and every ensemble row.

### 8.2 The RTX 3090: what it is for, and the ruling v3 got wrong

**v3 concluded that this card has no legitimate role in electronic structure. That conclusion is withdrawn.** The front-matter subsection "Corrections to v3's GPU ruling" states the reversal and the reasoning failure; this section states the replacement.

**The specification, unchanged and still correct.** RTX 3090 (GA102, compute capability 8.6): 10,496 CUDA cores across 82 streaming multiprocessors, 24 GB GDDR6X, **936 GB/s** memory bandwidth, 35.6 TFLOPS FP32, and an FP64 rate of 1/64 of FP32 = **0.556 TFLOPS [D]** ([GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf); [GeForce RTX 30 series specifications](https://en.wikipedia.org/wiki/GeForce_RTX_30_series)). The eight P-cores issue roughly 0.61–0.64 TFLOPS FP64 at AVX2 `[D]`.

**Why that comparison does not predict wall-clock, and must not be used as a routing rule.** Peak ratios order two machines only if both realise comparable fractions of peak. In Gaussian-basis electronic structure they do not:

- gpu4pyscf's Rys-quadrature kernels have an uncached arithmetic intensity of ~3/16 FLOP/byte against an A100 machine balance of 6.1 FLOP/byte, i.e. **memory-bound**; the named limiters are the register file ((pp|pp) needs 129 FP64 words of a 255-word budget; (dp|pp) exceeds it by nearly a factor of two and spills), local memory ((ii|ii) needs 731 KB against a 512 KB limit — this is *why* the basis ceiling is g), and occupancy ((ff|ff) is limited by "insufficient workload to fully occupy the streaming multiprocessors"). Delivered FP64 is 2–5 TFLOP/s of a 9.7 TFLOP/s peak at low angular momentum and 0.8 TFLOP/s at N = 7 `[M]` ([Li, Zhang, Sun & Chan](https://arxiv.org/html/2407.09700v1)).
- LibintX evaluates integrals **in double precision** at 25–70 % of hardware peak and states that "the ratio of peak FLOP rate of the V100 GPU to that of 1 Xeon Gold 6136 Skylake core is 73:1" — yet measures **107–1171× [M]** across integral classes, "higher than the 73:1 GPU:CPU performance ratio" ([LibintX](https://arxiv.org/html/2405.01834v2)). A measured ratio above the peak ratio proves the CPU realises a smaller fraction of its own peak.
- TeraChem's authors classify "all of the HF exchange matrix kernels" as memory-bound with unavoidably uncoalesced density-matrix access ([TeraChem](https://arxiv.org/html/2406.14920v3)); QUICK attributes residual CPU–GPU deviations to "differences in the exchange correlation quadrature grid", not arithmetic ([QUICK docs](https://quick-docs.readthedocs.io/en/latest/performance.html)).

**The operative figures of merit are therefore bandwidth, register and local-memory pressure, and occupancy.** On bandwidth the card's advantage is **9.1–10.4× [D]**: 936 GB/s against 89.6 GB/s at Intel-specified DDR5-5600 and 102.4 GB/s at DDR5-6400 XMP ([Intel ARK, i7-13700K](https://ark.intel.com/content/www/us/en/ark/products/230500/intel-core-i7-13700k-processor-30m-cache-up-to-5-40-ghz.html)). Quote the range, not a single number — v3's bare "102.4 GB/s, a factor of 9.1" used the XMP figure without saying so.

**Precision: there is no penalty.** gpu4pyscf's production path is double precision throughout; mixed precision appears in the v1.0 paper only as future work.

| Quantity | Documented agreement | Source |
|---|---|---|
| Energy vs CPU PySCF | **< 10⁻¹¹ Ha** — below the SCF convergence threshold of 10⁻¹⁰ | [Wu *et al.*](https://arxiv.org/html/2404.09452v2) |
| Gradient 2-norm vs CPU PySCF | **< 10⁻⁷ Ha/bohr** | [Wu *et al.*](https://arxiv.org/html/2404.09452v2) |
| Hessian 2-norm vs CPU PySCF | **< 10⁻⁶ Ha/bohr²** | [Wu *et al.*](https://arxiv.org/html/2404.09452v2) |
| Direct-SCF HF energy vs Q-Chem 6.1, (H₂O)₂–(H₂O)₁₀ / def2-TZVPP | 4.6 × 10⁻¹¹ to 9.6 × 10⁻⁸ Ha | [scf_pyscf_qchem](https://raw.githubusercontent.com/pyscf/gpu4pyscf/master/benchmarks/scf/scf_pyscf_qchem.md) |
| DF-SCF energy vs Q-Chem 6.1 | ~10⁻⁶–10⁻⁵ Ha | [df_pyscf_qchem](https://raw.githubusercontent.com/pyscf/gpu4pyscf/master/benchmarks/df/df_pyscf_qchem.md) |
| xQC in double precision vs gpu4pyscf | "reproduces GPU4PySCF's results exactly" | [xQC](https://arxiv.org/html/2507.09772v1) |

All `[M]`. The residual B3LYP discrepancies of 10⁻⁸–10⁻⁴ Ha that grow with system size in the direct-SCF table are **exchange–correlation quadrature-grid differences against Q-Chem, not precision loss** — the Hartree–Fock column of the same table, which has no grid, sits at 10⁻¹¹–10⁻⁸ Ha.

**v3's precision worry was real, but for other codes.** TeraChem defaults to mixed precision (integrals above a density-weighted Schwarz bound of 10⁻⁵ in FP64, the rest FP32) ([TeraChem](https://arxiv.org/html/2406.14920v3)), and an all-FP32 integral path costs about 1 mHa — the measured gly30/def2-TZVPP FP64−FP32 difference is **1.95 mHa = 1.22 kcal/mol ≈ 428 cm⁻¹ [M/D]** ([xQC](https://arxiv.org/html/2507.09772v1)). **v3 applied TeraChem's precision model to gpu4pyscf. That was the specific factual substitution that produced the wrong answer.**

**Capabilities, at v1.8.0 (2026-07-23)** ([CHANGELOG](https://raw.githubusercontent.com/pyscf/gpu4pyscf/master/CHANGELOG); [gpu4pyscf README](https://github.com/pyscf/gpu4pyscf)). v3 described v0.6.0 and several of its "hard limits" are no longer true.

*On the GPU today:* direct SCF and density fitting; SCF, analytic gradient and **analytic Hessian** for HF and DFT (RHF/RKS and UHF/UKS); LDA, GGA, meta-GGA, hybrid and range-separated functionals via libXC; spin-conserved and spin-flip TDA/TDDFT; geometry optimisation and transition-state search via geomeTRIC; ASE interface; **DFT-D3 and DFT-D4 dispersion**; VV10 with gradient and Hessian; **GPU-accelerated ECP integrals and their first and second derivatives** (v1.4.0, 2025-03-27; ECP gradient kernels v1.4.1); PCM models with analytic gradient and Hessian; SMD; CHELPG/ESP/RESP; multi-GPU DF energies, gradients and Hessians (v1.2.0–v1.2.1, experimental). Experimental: MP2/DF-MP2, CCSD, polarizability/IR/NMR, Raman, QM/MM with periodic boundary conditions, periodic SCF/DFT.

*Still genuine limits:* atomic basis up to **g**, auxiliary up to **i**; **no double hybrids**; no TDDFT Hessian; meta-GGA requires the density Laplacian; DF bounded "up to ~168 atoms with def2-tzvpd, bounded by CPU memory"; MP2 gradients and Hessians remain on the CPU and CCSD is experimental and energy-only. Binary packages target compute capability ≥ 7.0 — "Volta and later, such as Tesla V100, RTX 20 series and later" — so **the RTX 3090 is explicitly a supported target**.

*Documentation lag, worth stating.* [PySCF's GPU page](https://pyscf.org/user/gpu.html) still lists direct-SCF Hessians as CPU-only and ECP and dispersion as CPU. The repository README supersedes it. v3 cited the stale page.

**The card's legitimate roles in this document.** (a) **Reference-accuracy DFT via gpu4pyscf** — energies, analytic gradients and analytic Hessians, TDA/TDDFT, PCM/SMD, geometry and transition-state optimisation, in full double precision, for systems above roughly 100 basis functions (§8.3). (b) Machine-learned force-field inference and molecular dynamics. (c) GPU dense or matrix-free diagonalisation of small discrete-variable-representation Hamiltonians, with the caveats of §13.2. (d) Screening-grade mixed-precision SCF in TeraChem where ~1 mHa is tolerable. **Not for:** double hybrids, any coupled cluster of reference quality, DLPNO, CASSCF/NEVPT2, basis beyond g, or working sets above 24 GB.

**"No ORCA row anywhere in this document runs its own electronic structure on the GPU"** — correct, because ORCA has no GPU path ([PERUN HPC](https://wiki.perun.tuke.sk/env/orca/)). But v3's phrasing was misleading in the other direction: **several ORCA rows are GPU-*fed* through `!ExtOpt`**, and ORCA's own tutorial names `Opt`, `NEB-TS` and `GOAT` as supported consumers ([ORCA external-methods tutorial](https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extopt.html)).

**Reproducibility caveat, retained from v3.** GPU atomics are non-associative and run-to-run bitwise reproducibility is not assured. This matters at the 10⁻⁹ Ha level; the determinism guard of §20.2 stands, and a rotational constant is re-derived from a CPU re-optimisation before it is quoted.

### 8.3 The crossover: where the GPU starts and stops paying

The GPU's advantage is a function of occupancy, hence of size. gpu4pyscf's own direct-SCF benchmark, B3LYP/def2-TZVPP on water clusters, A100-80G against Q-Chem 6.1 on 32 cores ([scf_pyscf_qchem](https://raw.githubusercontent.com/pyscf/gpu4pyscf/master/benchmarks/scf/scf_pyscf_qchem.md)), all `[M]`:

| System | def2-TZVPP basis functions | Direct-SCF speedup vs 32-core Q-Chem |
|---|---:|---:|
| (H₂O)₂ | ~118 | **0.182 — 5.5× slower** |
| (H₂O)₃ | ~177 | 1.368 |
| (H₂O)₄ | ~236 | 2.673 |
| (H₂O)₅ | ~295 | 4.777 |
| (H₂O)₁₀ | ~590 | 8.033 |

Density fitting has a matching floor: vitamin C at STO-3G is **0.879×**, a loss ([df_pyscf_qchem](https://raw.githubusercontent.com/pyscf/gpu4pyscf/master/benchmarks/df/df_pyscf_qchem.md)).

**Crossover ≈ 150–170 basis functions against 32 Xeon cores `[D]`, hence roughly 50–90 basis functions against 8 P-cores `[D]`** — that is, essentially any molecule larger than a few heavy atoms in a triple-zeta basis. Li and co-workers give the mechanism: "larger molecules typically experience a more substantial speedup … due to the high GPU occupancy of a large molecule" ([Wu *et al.*](https://arxiv.org/html/2404.09452v2)); LibintX puts it as "small input or very lopsided ij,kl inputs will result in poor performance" ([LibintX](https://arxiv.org/html/2405.01834v2)). Rowan states plainly that "CPU-based calculations are competitive with small GPUs for smaller molecules" ([Rowan](https://www.rowansci.com/blog/gpu4pyscf)).

**This 50–90 bf figure is `[D]`, not `[M]`, and under the standing rule of §17.4 it may not on its own gate a routing decision.** It is marked as **requiring local measurement**, and §8.4 is the measurement.

**Where the advantage is largest, and where it is not**, all A100 figures against 32 cores, all `[M]`:

| Quantity | Speedup | Reading |
|---|---|---|
| DF Hessian (vitamin C / inosine) | **54.2× / 71.8–82.2×** | the GPU's best case; also the first thing to exceed 24 GB |
| DF gradient (raffinose 66 atoms / sphingomyelin 84 atoms) | 27.9× / 26.3× | use DF, always |
| DF SCF (raffinose) | 20.1× | |
| Direct-SCF gradient at def2-TZVPP, (H₂O)₁₀ | **2.0×** | direct SCF is the wrong algorithm for gradient-driven work |
| Direct-SCF gradient at STO-3G, (H₂O)₁₀ | 85.4× | small basis inflates the ratio; not the operating regime |
| Functional dependence at def2-TZVPP, (H₂O)₁₀ | LDA 3.58, PBE 3.60, HF 5.28, ωB97m-v 6.56, B3LYP 8.03, M06 8.73 | **the GPU wins most where exchange dominates** — the opposite of the FP64-GEMM intuition |

**Projection to this box.** To carry an A100 figure to an RTX 3090, de-rate by a factor between the bandwidth ratio (A100-80G 2039 GB/s ÷ 3090 936 GB/s ≈ **2.2×**) and the FP64-peak ratio (9.7 ÷ 0.556 ≈ **17.5×**), kernel by kernel — **not** at the FP64 ratio, because the kernels are not FP64-peak-limited — then multiply back up by **3–4×** because the published baseline is 32 Xeon cores, roughly four times this workstation's eight P-cores. For DF-SCF/def2-TZVPP on a 40–80 atom organic that lands at roughly **5–30× versus this CPU `[E]`**, consistent with what practitioners report. **That is an estimate, not a measurement, and it is the reason §8.4 exists.**

**Consumer-class evidence, since no peer-reviewed gpu4pyscf benchmark on an RTX 3090 or 4090 was found.** The honest position is bracketing, not a measured number:

| Evidence | What it shows |
|---|---|
| Rowan's GPU sweep across T4 (0.254 TFLOPS FP64), L4 (0.473), A10 (0.976), L40S, A100, H100, H200 versus Psi4/PySCF on 16 vCPU — **the 3090's 0.556 sits between L4 and A10** | 13× on 78-atom maraviroc; ">50×" on a 95-atom transition state; "even the 5-year-old A100 GPUs show significant speedup" ([Rowan](https://www.rowansci.com/blog/gpu4pyscf)) |
| BrianQC's entire vendor benchmark suite runs on a **GTX 1080 Ti** (FP64 ≈ 0.33 TFLOPS, *worse* than the 3090) against 8- and 16-core Xeons | a commercial vendor does not benchmark on a card where it loses ([BrianQC](https://www.brianqc.com/benchmarks), [metal complexes](https://www.brianqc.com/metal-complexes)); numeric speedups are in figures only — **`n.a.`** |
| TeraChem's own page: GeForce 1080 Ti **34.79 s** vs Tesla K80 **56.01 s** (RHF/6-31G*) despite the K80's ~9× FP64 advantage | FP64 peak does not order these devices ([PetaChem](http://www.petachem.com/performance.html)) |
| One user report of Q-Chem 5.4.2 + BrianQC on **4 × RTX 3090**: SCF 27 s → 5 s, results "the same within the difference one would expect" | weak evidence — four cards, undisclosed molecule and CPU — but evidence of the *sign*, on the exact card ([Q-Chem forum](https://talk.q-chem.com/t/q-chem-5-4-2-and-brianqc-1-2-1-wall-clock-time-discrepancy/538)) |
| Peer-reviewed endorsement of the hardware class: ωB97X-3c on consumer GPUs is "the most efficient implementation accessible with consumer-grade hardware" | ([Steinbach & Bannwarth, *PCCP* 26, 16567](https://pubmed.ncbi.nlm.nih.gov/38829649/)) — but TeraChem's GPU integral library is limited to l < 3, no f functions |

**Where v3's argument retains force, stated honestly.** Three counterweights survive. (i) xQC's authors write that "since A10-24G has 1/32 hardware units for FP64 compared to FP32, FP64 algorithms are always bounded by the limited compute units" ([xQC](https://arxiv.org/html/2507.09772v1)); the 3090 is 1/64, worse, so for the low-angular-momentum kernels where an A100 reaches 2–5 TFLOP/s the 3090 is hard-capped at 0.556 and **will** be several-fold slower. (ii) gpu4pyscf's multigrid Gaussian-plane-wave module reaches "approximately 80 % of the peak FP64 throughput" on A100 Coulomb kernels ([GPW GPU implementation](https://arxiv.org/abs/2603.24881)); where a kernel really is near-peak FP64 the 3090 *is* de-rated by nearly the full 17.5×. That module is periodic, not this document's workload, but it is the strongest surviving fragment of the old argument. (iii) Reported fractions of peak vary by an order of magnitude across integral classes — Barca and co-workers reach only 2 % of FP64 peak for [dd|dd] ([LibintX](https://arxiv.org/html/2405.01834v2)) — so **no single de-rating factor is correct**.

**The honest limits, in one list.** No GPU MPQC CCSD(T)-F12 exists in any code. GPU CCSD(T) exists in TeraChem, not in any free code, and even there a 63-atom, >1000-basis-function (T) correction takes "a little under 8 hours on a single node" ([Fajen *et al.*](https://www.arxiv.org/abs/2512.01055)). No double hybrids in gpu4pyscf. Basis functions beyond g are unavailable. 24 GB is a wall with no graceful spill; the A100 Hessian for 84-atom sphingomyelin needed 30 min on an 80 GB card, so **Hessians are the first thing to fall off a 24 GB board**. Anything input/output-bound — SPFIT/SPCAT fitting, QCxMS trajectory farms, file-heavy scans — gets nothing. And CPU fallback in gpu4pyscf is automatic and silent for any module not on the GPU list, which is a *timing* hazard: **always check the log for which module ran where.**

### 8.4 The fair-comparison protocol: settle it on your own machine

An out-of-the-box gpu4pyscf-versus-ORCA comparison is not a clean GPU-versus-CPU measurement. Five defaults differ, and three of them favour ORCA:

| # | Confound | ORCA 6.1 default | gpu4pyscf / PySCF default | Direction |
|---|---|---|---|---|
| a | GPU acceleration | none — ORCA has no GPU path | full | favours GPU (the real effect) |
| b | Parallel width | 8 MPI ranks on 8 P-cores | 10,496 CUDA cores / 82 SMs; published baselines are **32 Xeon cores**, ~4× this CPU | favours GPU |
| c | Exchange algorithm | **COSX is the default whenever HF exchange is needed** ([ORCA numerical integration](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/numericalintegration.html)) — a semi-numerical approximation | exact DF-K with `def2-universal-jkfit`, or full direct K | **favours ORCA** |
| d | Exchange–correlation grid | **DEFGRID2**: SCF AngularGrid 4 (Lebedev 302, pruned) | PySCF `level = 3`; the *published benchmarks* use **(99,590)**, finer than either | **favours ORCA** at defaults, more so against a published number |
| e | SCF convergence | between medium and strong: TolE 1e-6 → 3e-7, Thresh 1e-10 | `conv_tol = 1e-9` Eh | **favours ORCA** |

**This strengthens rather than weakens the case for the card.** If gpu4pyscf is much faster while doing *exact* DF exchange, on a *finer* grid, to a *tighter* threshold, the underlying advantage is larger than the raw ratio. A fair protocol will probably *reduce* the measured ratio while *validating* it. gpu4pyscf's own maintainers say so: "The default settings of quantum chemistry package can be significantly different. With different settings, the performances are not comparable" ([benchmarks README](https://raw.githubusercontent.com/pyscf/gpu4pyscf/master/benchmarks/README.md)).

Run this on three systems spanning the crossover — a water dimer (~118 bf), caffeine (24 atoms), and a 50–80 atom drug-like molecule.

**ORCA 6.1 input, 8 P-cores:**

```
! B3LYP def2-TZVPP def2/JK RIJK NOCOSX DEFGRID3 TightSCF NoUseSym NoPop
%pal nprocs 8 end
%maxcore 3000
%scf
  ConvForced 1
  TolE     1e-9
  TolErr   1e-7
  TolMaxP  1e-7
  TolRMSP  5e-9
  Thresh   1e-11
  DIISMaxEq 15
end
%method
  RunTyp Energy
end
* xyzfile 0 1 molecule.xyz
```

Key choices: **`RIJK` + `def2/JK` with explicit `NOCOSX`** matches gpu4pyscf's `density_fit()`, which uses one auxiliary basis for both J and K — leaving ORCA's default COSX in place is the single largest confound. **`DEFGRID3`** (AngularGrid 6 = Lebedev 590) is the closest stock ORCA grid to the (99,590) of the published benchmarks. **`TolE 1e-9`** matches PySCF's default. Pin to physical cores: `taskset -c 0-15 orca job.inp > job.out`, or bind eight ranks to the eight P-cores with `numactl`/`OMP_PLACES`.

**gpu4pyscf input, RTX 3090:**

```python
import time, pyscf, cupy
from pyscf import gto
from gpu4pyscf.dft import rks
from gpu4pyscf.drivers.dft_driver import warmup

warmup()                      # burn CUDA/JIT start-up outside the timer

mol = gto.M(atom='molecule.xyz', basis='def2-tzvpp',
            cart=False,       # spherical, matching ORCA
            max_memory=32000, verbose=4)

mf = rks.RKS(mol, xc='b3lyp').density_fit(auxbasis='def2-universal-jkfit')
mf.grids.atom_grid  = (99, 590)   # match DEFGRID3-class quadrature
mf.grids.prune      = None        # ORCA prunes; also run prune=True as a variant
mf.conv_tol         = 1e-9
mf.conv_tol_grad    = 1e-6
mf.direct_scf_tol   = 1e-11       # match ORCA Thresh
mf.max_cycle        = 100
mf.init_guess       = 'minao'

t0 = time.perf_counter()
e  = mf.kernel()
cupy.cuda.Stream.null.synchronize()
print('E =', e, 'SCF wall =', time.perf_counter()-t0, 'iters =', mf.cycles)
```

**Acceptance and reporting.**

1. **Correctness gate first.** |E(ORCA) − E(gpu4pyscf)| must be **< 1 mHa**; if it is larger the grids or auxiliary bases are not matched and the timing is meaningless. Expect ~10⁻⁵–10⁻⁴ Ha of residual grid difference, consistent with the published gpu4pyscf-versus-Q-Chem DF agreement.
2. **Report iteration counts.** A 12-versus-18-iteration difference is a 1.5× "speedup" that is not a speedup. Initial guesses differ (ORCA PModel, PySCF MINAO).
3. **Time the SCF loop only**, excluding start-up; discard the first run on each side.
4. **Run the default-versus-default comparison too, and report both.** `! B3LYP def2-TZVPP def2/J RIJCOSX` at DEFGRID2 against `mf.grids.level = 3` is the *practically relevant* number — it is what you will actually type — and the matched run is the *scientifically clean* one. Both belong in your notes.
5. **Repeat for gradient and Hessian.** Hessians are where the GPU advantage is largest and where 24 GB is most likely to bite.
6. **Log `nvidia-smi dmon` during the run.** Low SM occupancy and memory throughput far below 936 GB/s mean the job is below the crossover, not that the GPU is unsuited.
7. **Run every timing twice, once with the GPU idle and once with it loaded** (§9.1), or every pipelined tier estimate in §18 is unanchored.

**Record the result in the calibration file of §20.1.** Once measured, replace the `[E]` 5–30× projection in this document with your `[M]` number and re-derive the affected tier rows.

### 8.4a Setup 3 — HPC, and why "1 mo" is a campaign and not a job

Real academic clusters cap walltime at 48 hours: NERSC's `regular` QOS caps at 48 h, with `debug` at 0.5 h and `interactive` at 4 h, and the only multi-week QOS is `workflow` at 2,160 h on a quarter of a login node ([NERSC job policy](https://docs.nersc.gov/jobs/policy/)); TACC Stampede3 states that "the maximum runtime for any individual job is 48 hours" across every production queue, that nodes are never shared, and that "Jobs can be chained using checkpointing so that outputs from one job become inputs to the next" ([TACC Stampede3](https://docs.tacc.utexas.edu/hpc/stampede3/)).

**There is therefore no such thing as a one-month job.** The 3 d, 1 w and 1 mo tiers are campaigns of sub-48-hour units, and every such row in this document publishes its decomposition.

Restartability governs how a campaign is built:

| Calculation | Restartable? | Mechanism |
|---|---|---|
| SCF | effectively yes, via orbital reuse | `OptGuess MORead` "uses MOs of the previous point" and is the optimisation default ([ORCA 6.1 optimizations](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations.html)) |
| Geometry optimisation | **partly** — the accumulated Hessian survives, the trajectory does not | "the model Hessian updated during the previous calculation can be reused by passing a `basename.opt` file or a `basename.carthess` file", via `%geom InHess READ; InHessName` ([ORCA 6.1 optimizations](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations.html)) |
| Numerical frequencies | **yes** | `%freq Restart true` with the `basename.res.*` files, at identical level, basis and geometry ([ORCA 6.1 frequencies](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/frequencies.html)) |
| **Analytic frequencies** | **no** | explicitly not restartable ([ORCA 6.1 frequencies](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/frequencies.html)) |
| **MDCI / DLPNO / canonical CC single point** | **not documented** | the MDCI chapter describes no restart mechanism ([ORCA 6.1 MDCI](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/mdci.html)); treat as non-restartable |

Protocol for the upper tiers: decompose into units that finish in under 40 hours, leaving 20 % margin; never submit a coupled-cluster single point that has not been benchmarked to finish inside that, because it cannot be resumed; chain the resumable parts with `InHess READ` and `%freq Restart true`; and **prefer 6N-displacement parallelism over restart** — 60 displaced gradients per NumGrad cycle are independent, so running them as 60 separate array tasks converts a non-restartable multi-day job into a fully checkpointed campaign and exploits the fact that ORCA does not scale past about 16 cores anyway. Budget for whole-node charging: Stampede3 does not share nodes and charges for all cores on the node with a 15-minute minimum ([TACC Stampede3](https://docs.tacc.utexas.edu/hpc/stampede3/)), so packing several independent ORCA jobs onto one node is a requirement rather than an optimisation.

**Storage.** Canonical CCSD(T)/cc-pVTZ-F12 (paired with CABS: OptRI [Yousaf & Peterson, J. Chem. Phys. 129, 184108 (2008)], JKFIT, MP2FIT) on a 10-atom complex (~596 basis functions, ~35 correlated occupied orbitals, N_virt ≈ 561) needs a four-external integral set of about N_v⁴/8 doubles ≈ **99 GB**, plus a three-external set of about **25 GB**, if run integral-conventional. ORCA warns that "AOX-based calculations take four times as much disk space as AO-based calculations", recommends `KCOpt KC_AOBLAS`/`KC_AO` above about 300 basis functions, and reserves the full transformation for "up to about 300 basis functions" ([ORCA 6.1 MDCI](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/mdci.html)). **This job is not runnable integral-conventional on Setup 2 and must be AO-direct on Setup 3 with at least 150 GB of node-local scratch.** For DLPNO, the nearest measured anchor is about 8.40 GiB of scratch for a 24-atom uracil-dimer DLPNO-CCSD/cc-pVDZ TightPNO run ([OSTI report](https://www.osti.gov/servlets/purl/2578595)), scaling to an estimated **5–20 GB per TightPNO/cc-pVDZ-F12 (paired with CABS) point per rank**, hence 40–160 GB concurrent across 8 ranks unless `StorageType Shared` is used. Provision at least 500 GB of free NVMe on Setup 2 and never run these jobs on the OS drive. The binding requirement is IOPS and queue depth rather than bandwidth, because DLPNO writes many small irregular blocks from concurrent ranks: a consumer PCIe 4.0 NVMe drive is adequate, a SATA SSD or a network filesystem is not.

### 8.4b Setup 1 — the GitHub teaching tier

**Actions limits, verified.** Workflow run time 35 days; **job execution on GitHub-hosted runners 6 hours**; self-hosted job execution 5 days; **job matrix cap 256 jobs per workflow run**; concurrent standard hosted jobs **20 (Free), 40 (Pro), 60 (Team), 500 (Enterprise)**; storage and minutes **Free 500 MB artifacts + 2,000 min + 10 GB cache**, Pro 1 GB + 3,000 min, Team 2 GB + 3,000 min ([GitHub Actions limits](https://docs.github.com/en/actions/reference/limits)). Artifacts and logs are retained 90 days by default, configurable per artifact ([artifact retention](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/remove-workflow-artifacts)).

**Runner hardware.** Public repositories get **4 CPU, 16 GB RAM, 14 GB SSD** free and unlimited; private repositories get 2 CPU, 7 GB RAM, 14 GB SSD ([GitHub-hosted runners](https://docs.github.com/en/actions/using-github-hosted-runners/using-github-hosted-runners/about-github-hosted-runners)). Larger runners are Team/Enterprise only, **never eligible for included minutes, and always billed per minute including on public repositories**; the only GPU option is 4 CPU + one Tesla T4 ([larger runners](https://docs.github.com/en/actions/using-github-hosted-runners/using-larger-runners/about-larger-runners)). **Do not plan on GPU runners for a class.**

**Codespaces.** Machine types run from 2 cores / 8 GB / 32 GB storage to 32 cores / 128 GB / 128 GB ([what are Codespaces](https://docs.github.com/en/codespaces/about-codespaces/what-are-codespaces)); default idle timeout is 30 minutes, configurable 5–240 ([Codespaces timeout](https://docs.github.com/en/codespaces/setting-your-user-preferences/setting-your-timeout-period-for-github-codespaces)); stopped codespaces are auto-deleted after 30 days of inactivity, configurable 0–30 ([Codespaces automatic deletion](https://docs.github.com/en/codespaces/setting-your-user-preferences/configuring-automatic-deletion-of-your-codespaces)); included usage is 120 core-hours + 15 GB-month on Free and 180 core-hours + 20 GB-month on Pro, with organisation plans including none ([Codespaces billing](https://docs.github.com/en/billing/managing-billing-for-your-products/about-billing-for-github-codespaces)).

**GitHub Classroom is better than version 2 assumed.** Verified students get up to **180 core-hours per month** on their personal accounts. Verified teachers get a Codespaces Education benefit sized by GitHub as "enough for a class of **50** with **5 assignments per month**, using a **2 core machine** with **1 codespace stored per student**"; when enabled through Classroom, usage in assignment repositories is charged to the organisation and does not touch the student's personal allowance, and GitHub **auto-applies a policy restricting all organisation codespaces to 2-core machines** ([Classroom + Codespaces](https://docs.github.com/en/education/manage-coursework-with-github-classroom/integrate-github-classroom-with-an-ide/using-github-codespaces-with-github-classroom)). The teacher's numerical monthly allowance is not published: **`n.a.`**

**The binding constraint is concurrency and licensing, not money and not walltime.** On a public repository Actions minutes are free and unlimited on a 4 CPU / 16 GB runner, which is *better* than the 2-core Codespaces machine that Classroom policy enforces. Fifty students submitting simultaneously queue behind 20 concurrent slots on the Free plan. And the ORCA licence forbids putting ORCA in any shared or public image (Section 11.1). Version 2's diagnosis — that the constraint was "core count and paid machine-hours" — is replaced by these three.

**Reachable tiers.** Taking the 4-CPU public runner as the teaching unit, it delivers roughly 0.3–0.4× the throughput of Setup 2's eight P-cores, so a 6-hour job cap buys about 2 hours of Setup-2-equivalent work. The **10 s, 1 min, 30 min, 1 h and 3 h tiers fit**; the **12 h tier fits only as three chained 6-hour jobs**, and only for engines that may legally be installed; **1 d and beyond are out of reach** except as ensembles of independent short jobs. With 20 concurrent jobs and a 256-job matrix cap, a single workflow can dispatch a 256-conformer sweep, which is an excellent teaching artefact.

Practical rules: set `retention-days` explicitly on every artifact, because the Free artifact allowance is 500 MB against a 90-day default retention; upload only `.out`, `.xyz`, `.hess` and a compressed properties file, never scratch and never `.gbw`; prebuild the dev container through a prebuild workflow so students do not burn core-hours on package installation; and set codespace retention to 7 days rather than 30 to stay inside the storage allowance.

### 8.5 The routing decision procedure

Three things changed from v3: the "FP64 gate" is replaced by a **GPU capability-and-size gate**; **restartability** becomes a routing input rather than a footnote; and a **heterogeneous co-scheduling step** is inserted.

```
route(observable, product_class, target_accuracy, setup, budget_tier,
      n_atoms, n_basis, restart_needed):

  # ---- STEP 0: the product question, before anything else ----
  if measured_parent_or_analogue_exists:
      product = B or C           # window 0.03-0.1 %
  else:
      product = A                # window 0.3-0.5 % semi-rigid, 1-2 % floppy in B_0
      # state this in the output. Do not print a 0.1 % window for product A.
  # and state WHICH quantity: B_e or B_0. They are different specifications.

  # ---- STEP 1: engine capability gates (hard) ----
  if observable in {analytic_Hessian_at_SCF, VPT2_at_DFT, dipole, polarizability,
                    TD-DFT, LED, NBO, implicit_solvation, DLPNO, F12_single_point,
                    canonical_CC_energy_or_gradient}:
      engine, device = ORCA, CPU              # ORCA has no GPU path at all
  elif observable in {CC_analytic_2nd_derivative, anharmonic_CCSD(T)_force_field,
                      sextic_distortion, spin_rotation, DBOC,
                      vibration_rotation_alpha_r_at_CC}:
      engine, device = CFOUR, CPU             # VIB=EXACT, ANHARM=VPT2 / VIBROT
  elif observable in {F12_gradient, junChS-F12_geometry}:
      engine, device = Molpro, CPU            # commercial; see 9.1
  elif observable in {focal_point_gradient}:
      engine, device = Psi4, CPU              # Psi4 has the driver; ORCA does not
  elif observable in {SAPT_decomposition, SAPT(DFT)}:
      engine, device = Psi4 | autoPES, CPU
  elif observable in {coupled_channel_bound_states}:
      engine, device = BOUND, CPU
  elif observable in {MD_trajectory, PIMD, large_ensemble_sampling, MLFF_scan,
                      MLFF_Hessian_seed}:
      engine, device = (MACE | AIMNet2 | UMA), GPU
  elif observable in {DFT_energy, DFT_gradient, DFT_analytic_Hessian,
                      TDA/TDDFT, PCM/SMD, TS_search, ESP/RESP}:
      engine, device = (ORCA, CPU) or (gpu4pyscf, GPU)   # -> STEP 2
  elif observable in {DVR_diagonalisation}:
      engine, device = SciPy/CuPy/JAX, CPU_FP64 default   # see 13.2
  else:
      engine, device = ORCA, CPU

  # ---- STEP 2: GPU capability and size gate (REPLACES v3's "FP64 gate") ----
  if device is GPU and engine is a quantum-chemistry code:
      # 2a. method gate
      assert method not in {double_hybrid, DLPNO, canonical_CCSD(T),
                            CASSCF, NEVPT2, MP2_gradient, MP2_Hessian,
                            TDDFT_Hessian}
      assert max_angular_momentum <= g            # auxiliary <= i
      # 2b. size gate: the crossover, NOT the FP64 peak
      if n_basis < 100:        # ~<10 heavy atoms at def2-TZVPP
          prefer CPU/ORCA      # competitive, and better tooling  [D - measure]
      elif n_basis <= 300:
          use GPU WITH density fitting; direct SCF may lose here
      else:
          use GPU decisively; the advantage grows with occupancy
      # 2c. algorithm gate
      assert density_fitting_enabled        # DF grad 26-28x, direct-SCF grad ~2x
      # 2d. memory gate: 24 GB is the real limit
      assert peak_working_set < 24 GB       # no graceful spill
      # 2e. precision: NO accuracy penalty in gpu4pyscf (FP64, <1e-11 Ha).
      #     The gate applies only to reduced-precision codes:
      #       TeraChem mixed precision -> screening/MD, state it
      #       any all-FP32 path        -> ~1 mHa = 0.6 kcal/mol; screening only
      retain the determinism guard: GPU atomics are non-associative;
      re-optimise on the CPU before quoting a rotational constant.

  # ---- STEP 3: geometry threshold gate (Section 4.4) ----
  if observable produces A, B, C and claimed accuracy <= 0.5 %:
      require the corrected %geom block, TightSCF or better, DEFGRID3
      require reporting of final MaxG and the softest force constant
      require basis >= augmented triple zeta, or BSSEOptimization.cmp,
              else cap the claim at 3 %                      # section 4.7
      require a core correction if fc is used                # section 4.8
      if geometry came from an MLFF and product == A:
          FAIL -> re-optimise in QM first
      # MLFF geometries ARE permitted for Product-C differences and for
      # qualitative inertial-defect / planar-moment checks.

  # ---- STEP 4: setup feasibility ----
  if setup is Setup1_github:
      assert budget_tier <= 3h                 # 6 h job cap
      assert engine is not ORCA                # EULA: no ORCA in a shared image
      assert concurrent_jobs <= 20             # Free plan
      # ChemCompute is the better route for a class; see section 19.
  if setup is Setup2_workstation:
      n_ranks, maxcore = 8, 3000               # P-cores only; never 24
      if engine is ORCA and observable is canonical_CC and n_basis > 400:
          escalate(Setup3_hpc)                 # ~99 GB four-external storage
      if embarrassingly_parallel(observable):
          n_jobs, n_ranks = 16, 1              # the only case for the E-cores
  if setup is Setup3_hpc:
      assert per_job_walltime <= 40h

  # ---- STEP 4.5: heterogeneous co-scheduling (NEW) ----
  if setup is Setup2_workstation and device is CPU:
      if a GPU-eligible companion stage exists for this row:
          n_ranks, maxcore = 7, 3400           # reserve P-core 7 for the feeder
          launch the companion concurrently on executor 'gpu'
          assert companion.authority == "advisory_only"     # guard G1
          budget the CPU stream at 1.20x its solo wall time # section 9.1
      else:
          n_ranks, maxcore = 8, 3000           # nothing to feed; take it back

  # ---- STEP 5: restartability (NEW - a routing input, not a footnote) ----
  if not restartable(engine, observable):
      # ORCA analytic frequencies: NOT restartable
      # ORCA MDCI / DLPNO / canonical CC: no documented restart
      if estimated_runtime > wall_cap:
          if decomposable(observable):
              emit independent displacement / grid-point jobs   # PREFERRED
          elif CFOUR can do it:
              route to CFOUR (JOBARC + JAINDX + MOINTS + MOABCD restart)
          else:
              FAIL -> escalate setup or reduce the method
  else:
      emit_chained_jobs(InHess READ | %freq Restart true | %md Restart IfExists)

  # ---- STEP 6: emit a window, never a number ----
  return (centre, half_width = conformal_q90 x centre,   # section 17.5
          quantity in {B_e, B_0}, product_class, frozen_monomer_flag,
          engine, device, n_ranks, maxcore, concurrency_tag,
          state_in, state_out, scratch_estimate)
```

Four rules invert intuition and are stated separately.

- **Step 1 is a capability gate, not a performance gate:** no amount of GPU makes ORCA run on a GPU. The converse also holds — **no amount of FP64 peak arithmetic tells you whether gpu4pyscf will beat ORCA on this box; only the matched protocol of §8.4 does.**
- **Step 2b's threshold is derived, not measured on this hardware.** It is flagged accordingly and must be replaced by a local measurement before it is used to exclude anything.
- **Step 4's embarrassingly-parallel branch is the only configuration in which all sixteen cores of the 13700K should be used** — and note that a running GPU feeder competes with it, so drop to 15 concurrent jobs when one is active.
- **Step 6 emits a window and a quantity label.** A tier row that emits a number without saying whether it is B_e or B₀ has not answered the question.

**One throughput rule that the routing procedure cannot express, so it is stated here.** A potential-energy-surface scan is N independent SCFs, not one large one, and the GPU serialises them. For points *below* the crossover, run 8–16 single-rank ORCA or single-thread PySCF jobs in parallel on the CPU: **the CPU wins on throughput even where the GPU wins on latency.** For points above the crossover, run them sequentially on the GPU — anchor: 32 s per point for ωB97M-V/def2-TZVPP on an A100 in a 1,089-point enzalutamide torsion scan `[M]` ([Wu *et al.*](https://arxiv.org/html/2404.09452v2)), de-rated for a 3090. Best of both: GPU for the scan, CPU simultaneously for an independent second workstream (§9).

---

## 8A. Concurrency and the scout-and-anchor heterogeneous pipeline

v3 costed every tier as though it owned the machine. Two rows scheduled together do not cost the sum of their times, and a ten-tier wall-clock table cannot express that. This section supplies the missing model.

### 8A.1 Is the premise true? The contention budget

"The CPU and the GPU do not directly limit each other" is true at the level of arithmetic units and false at the level of four shared resources: host cores, memory bandwidth, the power and thermal envelope, and the operating-system scheduler. PCIe, which is the resource people usually worry about, turns out not to matter.

Per **one** concurrent GPU worker alongside an ORCA job on Setup 2:

| Resource | Capacity | ORCA demand | GPU-worker demand | Contention | Action |
|---|---|---|---|---|---|
| **P-cores** | 8 | 8 ranks | **1 full core** — a profiled MACE step is 31.9 ms wall of which 18.1 ms (57 %) is host-side `[M]` ([MACE profiling](https://arxiv.org/html/2510.23621v1)) | **direct, 12.5 %** | **Reserve one P-core. ORCA at 7 ranks, `%maxcore 3400`.** The binding constraint |
| E-cores | 8 | 0 (forbidden by §8.1) | suitable only for non-latency helpers | none | orchestrator, parsers, deduplication, I/O go here |
| **Memory bandwidth** | 89.6 GB/s at Intel spec, 102.4 GB/s at XMP | high during the four-index transform, COSX and DLPNO pair/PNO construction | MLFF: peak host RAM ≲0.1 GB, dispatch-bound → **<5 % `[E]`**; gpu4pyscf: **5–15 % `[E]`** | small for MLFF, moderate for GPU-DFT | prefer an MLFF companion to a correlated ORCA job. **Measure locally — currently `n.a.`** |
| PCIe 4.0 ×16 | **63 GB/s duplex** ([Rambus](https://www.rambus.com/blogs/pci-express-4/)) | ~0 | ~36 MB once at model load; ~kB per step | **<0.001 % `[D]`** | Non-issue. Stop citing it as a reason not to pipeline |
| Host RAM | 64 GB | 7 × 3.4 GB = 23.8 GB + scratch cache | ≲1 GB per worker plus a CUDA context | ~40 GB free | comfortable; do not raise `%maxcore` to consume it |
| VRAM | 24 GB | 0 | 1–2 GB per MLFF worker `[E]` | none at ≤3 workers | cap with `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT='0=6G'` |
| GPU SMs | 82 SMs / 10,496 cores | 0 | one MLFF worker occupies the GPU **43 %** of wall time `[D]` | under-use, not contention | 3 workers under MPS at `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=33` |
| MPS client contexts | **48** (CUDA ≤13.0), 60 on r590 ([MPS overview](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)) | 0 | 1 per worker | none | never binding here |
| MIG partitions | **0 — unsupported on GeForce** ([MIG supported GPUs](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html)) | — | — | — | use MPS; do not plan for MIG |
| **Power** | CPU 253 W maximum turbo ([Intel datasheet](https://media.distrelec.com/Web/Downloads/_t/ds/BX8071513700K_eng_tds.pdf)) + GPU 350 W board ([GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)) = **603 W** at two components | 253 W peak | 350 W peak | **real** | ≥850 W supply; `nvidia-smi -pl 280` during pipelined campaigns. NVIDIA publishes no wattage recommendation — **`n.a.`** |
| Thermals | one case, one CPU cooler | — | 350 W into shared air | **5–10 % CPU clock loss `[E]`, unverified** | measure clocks with the GPU idle and loaded before trusting any tier time |
| OS scheduler | Linux CFS | needs pinning | needs pinning | migration hazard | `cpu_affinity='block'` / `'block-reverse'`; never oversubscribe GPU clients past physical cores |

**Composite verdict: free parallelism is ~85 % real `[D]`, not 100 %.** Running one GPU worker alongside ORCA costs one P-core (12.5 % of the rank budget) plus an estimated 5 % bandwidth and 5–10 % thermal clock loss: 8/7 × 1.05 × 1.05 ≈ 1.26, less superlinear cache effects, so **budget a 1.20× slowdown on the CPU stream `[E]`** in exchange for a full concurrent GPU stream. That trade is strongly positive whenever the GPU stream does anything useful at all. It is negative only when the GPU stream idle-waits — which is the failure mode the architecture below exists to prevent.

The authors of the MACE profiling work characterise the workload as "dispatcher/launch-bound … CPU/launch-bound rather than GPU-bound", naming synchronisations, dtype and device casts, many small kernel launches, Python dispatch, indexing and gather–scatter ([MACE profiling](https://arxiv.org/html/2510.23621v1)). **The feeder is a core-hog, not a RAM-hog, and it must sit on a P-core**: E-cores top out at 4.20 GHz against the P-cores' 5.30 GHz, and a launch-bound loop on an E-core would lose an estimated 20–30 % `[E]`.

For gpu4pyscf specifically the host stays involved for a documented list — MP2 gradients and Hessians remain on the CPU, and any module not on the GPU list falls back silently. A job therefore always has a CPU tail; check the log.

### 8A.2 Scout and anchor: the architecture

The user's two proposals both have partial names in the literature and neither was in v3.

**The GOAT → GPU → back loop is multi-fidelity / hierarchical ensemble refinement.** ORCA already ships the plumbing: `!ExtOpt` with `ProgExt` lets an external program serve energies and gradients to ORCA's optimiser, GOAT included, and the FACCTs wrapper repository supplies a server/client pair precisely because the standalone script "might slow down your calculation due to heavy imports" ([ORCA external-methods tutorial](https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extopt.html)).

**The CPU-optimise / read-state / GPU-PES-map / feed-back idea is an MLFF-preconditioned, uncertainty-gated hybrid optimiser, and it is published.** An ML+UQ optimiser that trains on the fly, steps on the ML surface, escalates to DFT when committee uncertainty exceeds a threshold, and then **computes the ML Hessian and uses it to precondition a BFGS optimiser driven by accurate DFT forces** cut DFT force calls by **2–3× `[M]`**: Au(643) 17 → 7 (59 %), Pd₁₃H₂ 38 → 18 (53 %), Rh(211) 11 → 4.5, Pd₂₀O₄ 23 → 17.1 ([Singh & Henkelman](https://theory.cm.utexas.edu/henkelman/pubs/singh24_10022.pdf)). Their operational parameters are worth copying: ML step target F_max,ML < 0.01 eV/Å, loose DFT gate F_max,DFT < 0.05 eV/Å, final F_max,DFT < 0.01 eV/Å, uncertainty coefficient ε = 1.50, and the guide declared unreliable after n_th = 5 failures.

**What is genuinely new, and must be labelled as engineering rather than theory.** Everything published is synchronous and single-device: the surrogate is consulted *between* high-level steps, on the same machine, blocking. Three extensions are not published as a named method: **device-disjoint execution** (the surrogate runs on the GPU while the DFT optimiser runs on the CPU, so it is free in wall clock rather than merely cheap in FLOPs); **reading the live state of the running CPU calculation** to decide where to map next; and **steering** — using the concurrent map to detect that the CPU job is walking toward a saddle or a spurious minimum, and to discover minima the CPU job will never visit. The name adopted here is **scout and anchor**. The *anchor* is the authoritative high-level CPU calculation that alone determines reported numbers. The *scout* is a concurrent cheap surface on the GPU that may supply starting points, Hessians, basins to check and warnings, and nothing else.

```
                    ┌────────────── ORCHESTRATOR (Parsl DFK, E-cores) ────────────┐
                    │  provenance.jsonl  ·  gates G1-G7  ·  stage scheduler        │
                    └───────┬──────────────────────────────────────┬──────────────┘
                            │ executor='cpu'                       │ executor='gpu'
                            │ 7 P-cores, 1 worker                  │ 1 P-core x 3 workers, MPS
                            ▼                                      ▼
   ┌────────── ANCHOR STREAM (authoritative) ─────┐   ┌──── SCOUT STREAM (advisory only) ────┐
   │ S1  GOAT / GFN2-xTB      -> ensemble_cpu.xyz │   │ T1 GOAT !ExtOpt + oet_server(AIMNet2)│
   │ S2  DFT opt (MLFF-seeded)-> iso_NNN.xyz      │<--│ T2 MLFF relax + MLFF Hessian         │
   │ S3  MPQC CCSD(T)-F12 re-rank-> ranked_hi.csv    │   │ T3 gpu4pyscf DF screen -> cull list  │
   │ S4  Analytic Hessian, top 3 -> B0, imag freq │   │ T4 committee UQ + basin re-check     │
   └──────────────────────────────────────────────┘   │ T5 live PES map around the anchors   │
                            │                          └──────────────────────────────────────┘
                            ▼        feedback arrows carry ONLY: xyz seeds, .carthess,
                     reported values                   cull lists, and warnings
```

**The single rule that makes this safe: arrows from the scout stream to the anchor stream carry starting points, Hessians, cull lists and warnings — never energies, never geometries that are reported, never orderings that are reported.**

**Two architectures, and which to use.** (a) Inside ORCA, via `!ExtOpt`: GOAT explores directly on the MLFF surface, better basin coverage at MLFF fidelity, but ORCA blocks on every gradient, one geometry per request, and the server is a single point of failure and of serialisation. (b) Outside ORCA, dump-and-refine: GOAT runs at GFN2-xTB on the CPU, `basename.finalensemble.xyz` is parsed, each structure is dispatched to a pool of GPU workers, the refined set is de-duplicated and re-ranked, and only survivors return to the CPU. (b) batches on the GPU, pipelines with other CPU work, gives one JSON per structure as an audit trail, and loses one structure rather than the whole run when a worker dies. **Use (b) as the production pattern and (a) as the cheap first pass — and run both simultaneously from the same start, taking the union of the two ensembles.** They cost the same wall time as either alone because they use different devices, and they explore different surfaces. That is the diversity insurance v3's one-month "completeness" row was trying to buy at 5,760 core-hours.

**Why server mode is mandatory, not optional.** A first MACE call including model load costs **≈30 s** against **≈48 ms** steady state `[M]` ([MACE profiling](https://arxiv.org/html/2510.23621v1)). A GOAT run of 900 optimisations × ~50 gradient calls = 45,000 calls. At 30 s each, standalone mode costs **375 hours**; in server mode at 48 ms it costs **36 minutes** `[D]`. This is the difference between feasible and not.

### 8A.3 The MLFF-preconditioning recipe

**Step 0 — model choice.** MACE-OFF24(M) for H/C/N/O/S organics (48 ms per inference, 9.1 M parameters, r_max 6.0 Å); AIMNet2 (`-m aimnet2-2025`) when charge or multiplicity flexibility or broader element coverage is needed. **Never mix model families within a workflow** — mixing can produce errors "of the order of tens of kcal/mol … with no failure indication" ([`oet_aimnet2` readme](https://raw.githubusercontent.com/faccts/orca-external-tools/main/readmes/aimnet2.md)).

**Step 1 — GPU pre-optimisation.** ASE + MLFF, `LBFGS` or the updating `GPMin`, `fmax = 0.02 eV/Å`. Do not chase tighter: AIMNet2's float32 energy precision is ≈4 × 10⁻⁶ Eh, so a tighter target chases noise.

**Step 2 — GPU Hessian.** By autograd (mace-torch) or finite difference of MLFF forces: 6N force evaluations, ~1–3 s for a 9-atom complex `[E]`. **Do not enable `--compile` if you need Hessians** — it is incompatible, it adds 10–60 s to the first call, and it recompiles when shapes change, which is "catastrophic in NEB" ([`oet_aimnet2` readme](https://raw.githubusercontent.com/faccts/orca-external-tools/main/readmes/aimnet2.md)).

**Step 3 — transfer.** Write the MLFF Hessian in ORCA's Cartesian Hessian format and hand it over:

```
%geom
  InHess READ
  InHessName "mlff_guess.carthess"
  Calc_Hess false
end
```

Emit the mass-unweighted Cartesian Hessian in Eh/bohr²; get the units wrong and the optimiser takes an absurd first step. **Assert that the lowest six eigenvalues are near zero before using it.**

**Step 4 — CPU DFT optimisation** with the corrected `%geom` block of §4.4, `TightSCF`, `DEFGRID3`; report the final `MaxG` and the softest force constant.

**Step 5 — validation**, §8A.5.

**What it saves.** Combining the published 2–3× force-call reduction with a starting geometry already near the DFT minimum, an optimisation that needed ~30 cycles from an xTB start should need **~12 `[E]`**.

**Caveats, plainly.** The published 2–3× is for surfaces and metal clusters, not van der Waals complexes; the transfer is an assumption and must be measured on the first three systems. An MLFF Hessian for a floppy intermolecular mode may be qualitatively wrong in exactly the soft coordinate that determines B — a wrong Hessian slows convergence but, because DFT forces still drive the step, **does not bias the minimum**. That is precisely why the MLFF may precondition and may never be stepped on near convergence.

**A stated negative result, so nobody spends a week on it.** Below 100 atoms ASE's `PreconLBFGS` reverts to plain LBFGS and the documentation recommends standard BFGS/LBFGS at that size ([ASE optimize docs](https://ase-lib.org/_sources/ase/optimize.rst)). **Geometric preconditioning does not help at 5–10 atoms; the MLFF Hessian is the only preconditioner in play.** For contrast, a Gaussian-process local optimiser buys about 25 % over ASE BFGS (42.1 ± 0.3 energy evaluations against 56.2 ± 0.5 on 1,000 random 10-atom Au clusters `[M]`, [del Río, Mortensen & Jacobsen](https://arxiv.org/pdf/1808.08588)) — real, free, and an order of magnitude smaller than the MLFF-preconditioner gain.

### 8A.4 MPS: the mandatory GPU configuration for small-molecule concurrency

This is the single most important missing GPU setting in v3.

Without MPS, "work launched to the compute engine from work queues belonging to different CUDA contexts cannot execute concurrently", which "can cause underutilisation of GPU compute resources when work launched from a single CUDA context is insufficient to use all resources" ([NVIDIA MPS architecture](https://docs.nvidia.com/deploy/mps/architecture.html)) — a precise description of a 9-atom SCF on 10,496 cores. NVIDIA's stated use case is exactly ours: "MPS is useful when each application process does not generate enough work to saturate the GPU" ([When to Use MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)).

**Hard limits.** Volta-and-later MPS supports **48 client CUDA contexts per device on CUDA 13.0 and prior, 60 on r590**; exceeding it fails `cuCtxCreate()`. Linux only, 64-bit only, one active server per user, `EXCLUSIVE_PROCESS` recommended. `CUDA_DEVICE_MAX_CONNECTIONS` defaults to 2 for Volta MPS clients and **raising it reduces the number of available clients**. Monitoring tools including `nvidia-smi` and NVML attribute **all client behaviour to the MPS server process**, so per-worker resource facts must be self-reported ([MPS overview](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf); [When to Use MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)).

**Partitioning.** `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` limits the thread fraction a client may use; the documented strategy for equal partitioning is `100 % / n` for n expected clients, and reducing it also "effectively reduce[s] the context storage allocation size", so it is a VRAM lever as well as a quality-of-service one. Hard-cap VRAM per client with `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT='0=4G'`. Set the open-file limit to **16384 or higher**, which NVIDIA recommends to avoid shared-memory-segment failures ([MPS overview](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf)).

**How many workers fit and pay.** Three limits, in order of bite:

| Limit | Value | Workers permitted |
|---|---|---|
| MPS client CUDA contexts | 48 (CUDA ≤13.0) / 60 (r590) `[M]` | 48+ — never binding here |
| VRAM 24 GB, ~1–2 GB per worker `[E]` | | 12–24 — rarely binding |
| **Host P-cores for launch-bound feeders (57 % host-side) `[M]`** | 1 core per worker | **2–4 in practice** |

**Answer: 2–4 concurrent GPU workers, and the binding constraint is host cores, not the GPU.** One MACE worker occupies the GPU 43 % of the time, so two reach ~86 % occupancy and a third saturates it; expect ≈2.2–2.4× single-worker throughput at three workers `[E]`. Beyond three you are trading ORCA ranks for nothing.

**MIG is verified unsupported.** Multi-Instance GPU requires compute capability ≥ 8.0 and the supported-product table lists only datacentre and RTX PRO parts — **no GeForce product appears** ([MIG supported GPUs](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html)). Settled; do not plan for it.

**The kernel-launch floor.** Launch overhead is **5–15 µs** ([zymtrace](https://gozymtrace.com/blog/zymtrace-03-cuda-kernel-launch-latency)); the academic micro-benchmark finds the CPU-side call "nearly equal to the latency of launching an additional kernel" for small kernels ([Zhang, Wahib & Matsuoka](https://www.hpcs.cs.tsukuba.ac.jp/icpp2019/data/posters/Poster17-abst.pdf)). Treat 10 µs as the working number and re-measure locally. This is why tiny systems are launch-bound rather than compute-bound, and it is the same floor that makes a small matrix-free Lanczos iteration lose on the GPU (§13.2).

**Setup, once, before the orchestrator starts:**

```bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
ulimit -n 16384
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
nvidia-smi -pl 280            # 80 % board power: costs a launch-bound workload
                              # little and buys back CPU clock
# ... run the campaign ...
echo quit | nvidia-cuda-mps-control
```

Before a batch campaign, **measure per-job VRAM rather than guessing** — no published figure exists for a 10-atom def2-TZVP gpu4pyscf job (**`n.a.`**):

```bash
python gpu_point.py --index 0 &
sleep 5; nvidia-smi --query-gpu=memory.used --format=csv
wait
# set N = floor(20000 MB / measured_MB), capped at the host-core limit
seq 0 4999 | parallel -j "$N" --joblog gpu.log 'CUDA_VISIBLE_DEVICES=0 python gpu_point.py --index {}'
```

Do not hand-roll CUDA streams: gpu4pyscf v1.5.0 "removed pre-allocated streams that caused inconsistent synchronization", which is a warning that stream management inside the code is not a user-facing tuning knob. Use process-level concurrency under MPS; that is what MPS is for. There is also **no batched-molecule API** in gpu4pyscf — its batching is internal (integral and grid batching) — so **`n.a.`** for that route.

### 8A.5 Integrity guards G1–G7

The danger is precise: **if a cheap surface chooses which structures survive, the cheap surface's bias becomes the reported answer while every reported number still carries a high-level label.** The pipeline creates three new leak paths — selection, ordering and Hessian seeding — and each needs its own guard.

**G1 — The cheap surface may set the starting point, never the answer.** An MLFF or reduced-precision geometry, energy or ordering may appear in no reported quantity. Every reported A, B, C, dipole or relative energy traces to a high-level calculation. Concretely: the MLFF may supply `.xyz` and `.carthess`; it may not supply the converged geometry. *(The Product-A restriction of §4.4 is the general form of this rule; MLFF geometries remain permitted for Product-C differences and for qualitative inertial-defect and planar-moment checks.)*

**G2 — Verify the final structure with a high-level Hessian.** The MLFF Hessian is a preconditioner only. The final structure carries a high-level Hessian showing the correct number of imaginary frequencies, and the softest force constant is reported. This is also the guard against converging to a saddle.

**G3 — Basin-identity check.** After the high-level re-optimisation, compute heavy-atom RMSD and, for a complex, the intermolecular centre-of-mass separation R and the two orientation angles, between the scout's predicted minimum and the anchor's. Gate: **RMSD > 0.25 Å or ΔR > 0.20 Å ⇒ flag "basin change"**, log it, and re-run the guide from the anchor geometry to check that the guide's basin still exists. **Thresholds are proposed defaults `[E]`, not published values; calibrate on the first three complexes.**

**G4 — Rank-inversion audit, mandatory before any culling.** Never cull on the cheap surface without measuring the cheap-versus-expensive rank correlation on a sample. Protocol: take the full MLFF-ranked ensemble, compute high-level energies for the **top 10 plus a random 10 from the remainder**, report Spearman ρ and the maximum rank displacement, and **cull only if ρ ≥ 0.9 and no structure outside the retained set is within the retention window at the high level.** The multi-fidelity Bayesian-optimisation literature supports exactly this precondition: benefit requires the correlation to be "very high" and the low-fidelity cost very low, and multi-fidelity max-value entropy search failed to beat single-fidelity on 2 of 4 chemistry problems ([Judge *et al.*](https://arxiv.org/html/2409.07190v1)). Foundation-MLFF interaction-energy errors of **3.5–7.3 kcal/mol on S30L `[M]`** ([MACE-POLAR-1](https://arxiv.org/html/2602.19411v1)) are larger than typical van der Waals isomer separations, so the prior expectation on this system class is that a foundation-MLFF ranking is untrustworthy and must be used as **a filter with a wide window, not a ranking**. Recommended retention window: everything within **10 kcal/mol** of the MLFF minimum, i.e. >1.4× the published maximum error.

**G5 — Uncertainty gate.** The committee criterion of §10.8 (ε = Q₃ + 1.5 × IQR on the model's own training-error distribution) applies unchanged to guide predictions. AIMNet2's ensemble members must be run separately and averaged outside ORCA, which upstream recommends for production; a committee costs 4× GPU time — cheap, and the pipeline has the headroom.

**G6 — Abort-the-guide rule.** After **n_th = 5** guide failures (guide says converged, high level says not), declare the guide unreliable and complete the job with pure high-level optimisation ([Singh & Henkelman](https://theory.cm.utexas.edu/henkelman/pubs/singh24_10022.pdf)). Log the abort.

**G7 — Determinism and precision provenance.** Record the MLFF **canonical** model key, not the alias — aliases "may be repointed in future AIMNet releases", so prefer `-m aimnet2-wb97m-d3_0` over `-m aimnet2`. Record kernel-determinism flags and checkpoint hashes for all GPU work. And note that under MPS `nvidia-smi` attributes all client activity to the server process, so per-worker resource facts must be self-reported.

**Audit trail.** One JSON line per guided decision, appended to `provenance.jsonl`:

```json
{
  "event_id": "b7f1c2…",
  "timestamp": "2026-08-09T18:41:02Z",
  "stage": "mlff_preopt",
  "decision": "seed_dft_optimisation",
  "guide": {"code": "mace-torch 0.3.x", "model_key": "MACE-OFF24-medium",
            "sha256": "…", "precision": "float32", "device": "cuda:0",
            "mps_active_thread_pct": 33},
  "input":  {"structure_id": "iso_017", "source": "goat_xtb.finalensemble.xyz#17",
             "sha256": "…"},
  "output": {"xyz_sha256": "…", "E_guide_eV": -1234.5678, "fmax_eV_A": 0.018,
             "committee_sigma_meV_atom": 4.1, "hessian_file": "iso_017.carthess"},
  "gates":  {"G4_spearman_rho": 0.93, "G4_sample_n": 20,
             "G5_uncertainty_pass": true, "G3_rmsd_A": null},
  "consumer": {"anchor_job": "iso_017_wb97xd4.inp", "hessian_transferred": true},
  "authority": "advisory_only"
}
```

After the anchor job finishes, a second line with `"stage": "anchor_verify"` records `G3_rmsd_A`, the DFT `MaxG`, the imaginary-frequency count and the softest force constant. **Rule: any reported number must be traceable to a line with `"authority": "authoritative"`, and no such line may exist for a guide stage.**

### 8A.6 The orchestrator: Parsl, with one configuration for both setups

| Tool | Heterogeneous CPU+GPU? | Verdict |
|---|---|---|
| **Parsl** | Yes, first-class. `available_accelerators` "pin[s] each worker to exactly one of the provided accelerators"; `cpu_affinity` supports `block`/`alternating`/`block-reverse`; `SlurmProvider` with `scheduler_options='#SBATCH --gres gpu:1'` ([Parsl HighThroughputExecutor](https://parsl.readthedocs.io/en/stable/stubs/parsl.executors.HighThroughputExecutor.html); [Parsl example configs](https://parsl.readthedocs.io/en/stable/userguide/configuration/examples.html)) | **Recommended for both Setup 2 and Setup 3** |
| Dask | Yes via resource annotations; less explicit accelerator pinning | second choice |
| Covalent | Yes — per-task `@ct.electron(executor=gpu_executor)` ([Covalent executors](https://docs.covalent.xyz/docs/features/executors/)) | viable; heavier |
| AiiDA | Provenance-first; ≈35,000 processes/hour with 12 workers `[M]` ([AiiDA 1.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC7479590/)) | use if provenance is the priority |
| Snakemake | File-driven DAG | wrong shape — feedback loops are not a static DAG |
| QCFractal | Chemistry-native; `queue_tags` make a worker "only claim tasks with these tags" ([QCFractal managers](https://docs.qcarchive.molssi.org/admin_guide/managers/index.html)) | right answer if the group wants a shared results database; layer it under Parsl rather than replacing it |
| GNU parallel | No, but restart is one flag on a text log ([GNU parallel](https://www.gnu.org/software/parallel/man.html)) | **the right tool for a flat PES grid**; use it for §11's campaigns |

**Setup 1 has no GPU worth planning on** — the only GPU larger runner is 4 CPU plus one Tesla T4, Team/Enterprise only and always billed. The teaching tier runs the CPU executor only, and the pipeline must degrade to a single-executor configuration by changing one line. Design for that.

```python
# hetero_config.py — one CPU executor + one GPU executor, one 13700K + one RTX 3090
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider

config = Config(
    executors=[
        HighThroughputExecutor(
            label="cpu",                    # ORCA: heavy, MPI-parallel, P-cores only
            max_workers_per_node=1,         # ONE ORCA job at a time; it owns 7 ranks
            cores_per_worker=7,             # P-cores 0-6; core 7 feeds the GPU
            cpu_affinity="block",
            mem_per_worker=28,              # GB; 7 ranks x %maxcore 3400 + headroom
            provider=LocalProvider(init_blocks=1, min_blocks=1, max_blocks=1,
                worker_init=("export OMP_NUM_THREADS=1; "
                             "export KMP_HW_SUBSET=8c:intel_core,1t")),
        ),
        HighThroughputExecutor(
            label="gpu",                    # MLFF / gpu4pyscf workers, MPS-shared
            available_accelerators=3,       # pins each worker to one slot, caps at 3
            max_workers_per_node=3,
            cores_per_worker=1,
            cpu_affinity="block-reverse",   # keep feeders away from the ORCA block
            mem_per_worker=6,
            provider=LocalProvider(init_blocks=1, min_blocks=1, max_blocks=1,
                worker_init=("export CUDA_VISIBLE_DEVICES=0; "
                             "export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=33; "
                             "export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT='0=6G'; "
                             "ulimit -n 16384")),
        ),
    ],
    retries=2,
)
```

Apps are routed by label:

```python
from parsl import python_app, bash_app

@bash_app(executors=['cpu'])
def orca(inp, stdout, stderr):
    return f"/opt/orca_6_1/orca {inp}"

@python_app(executors=['gpu'])
def mlff_relax(xyz_in, xyz_out, model="mace-off24-medium", fmax=0.02):
    ...
```

**HPC variant: replace both `LocalProvider`s with `SlurmProvider`** (`scheduler_options='#SBATCH --gres gpu:1 --gpus-per-node=1'`, `max_workers_per_node=1` on the GPU executor), keep the labels, and keep every app decorator unchanged. One code path for both setups is worth more than a marginally better tool on one of them.

**Correctness notes that matter more than style.** (1) The MLFF calculator must be cached on the Parsl worker so the 30 s model load is paid once per worker, not once per structure — the same insight as ORCA's server/client mode. (2) `write_orca_carthess` must emit the mass-unweighted Cartesian Hessian in Eh/bohr²; assert that the lowest six eigenvalues are near zero. (3) The G4 audit costs about 20 high-level single points up front. **Do not skip it to save time; skipping it is the failure mode this whole section exists to prevent.**

### 8A.7 The costed nine-atom example

**Task.** A 9-atom van der Waals complex: enumerate and rank the isomers, produce rotational constants for the top three.

**Assumptions, every one labelled and every one requiring local re-measurement per §20.1.** A1: GOAT needs ≈100 × N_atoms = **900** optimisations `[M, from the manual]`. A2: GFN2-xTB optimisation ≈ 3 s on one core `[E]`. A3: GOAT yields **40** distinct isomers after Stage-A deduplication `[E]`. A4: ωB97X-D4/def2-TZVPP on 9 atoms ≈ 250 basis functions, one gradient ≈ **25 s** on 8 P-cores, ≈29 s on 7 `[E]`. A5: DFT optimisation ≈ **30** cycles from an xTB start, ≈**12** from an MLFF start with an MLFF Hessian `[E, from the published 2–3×]`. A6: analytic DFT Hessian ≈ **1,200 s** `[E]`. A7: MPQC CCSD(T)-F12 single point ≈ **600 s** `[E]`. A8: MLFF inference **48 ms** steady state, model load **30 s** `[M]`. A9: gpu4pyscf DF-r²SCAN single point ≈ **8 s** `[E]` — below the crossover, so no large GPU win is claimed. A10: heterogeneous mode costs one P-core plus ~5 % ⇒ CPU stage times × **1.20** `[E]`.

**Route 1 — CPU only, sequential.**

| Stage | Arithmetic | Wall |
|---|---|---|
| GOAT / GFN2-xTB, 8 workers | 900 × 3 s / 8 | 0.09 h |
| DFT re-optimisation, 40 isomers | 40 × 30 × 25 s | 8.33 h |
| MPQC CCSD(T)-F12 re-rank, top 10 | 10 × 600 s | 1.67 h |
| Analytic Hessians, top 3 | 3 × 1,200 s | 1.00 h |
| **Total** | | **11.09 h**; GPU utilisation 0 % |

**Route 2 — naive GPU offload (move the DFT optimisations to gpu4pyscf; the CPU waits).**

| Stage | Arithmetic | Wall |
|---|---|---|
| GOAT / GFN2-xTB | as above | 0.09 h |
| gpu4pyscf DF-DFT re-optimisation, 40 isomers, one at a time | 40 × 30 × 10 s | 3.33 h |
| MPQC CCSD(T)-F12, CPU | | 1.67 h |
| Hessians, CPU | | 1.00 h |
| **Total** | | **6.09 h — 1.8×** |

CPU idle 3.3 h; GPU idle 2.8 h. **Both devices are half-wasted. That is what "use the GPU" means if you do not pipeline, and it is why the naive framing under-sells the hardware.**

**Route 3 — fully pipelined heterogeneous.**

| Time | CPU stream (7 P-cores) | GPU stream (3 MPS workers + 1 P-core) |
|---|---|---|
| 0 → 0.11 h | GOAT / GFN2-xTB, 900 optimisations, 7 workers | GOAT / `!ExtOpt`+AIMNet2 server — **same wall time, different surface** |
| 0.11 → 0.23 h | Stage-A dedup on the union of both ensembles → 40 isomers; write ORCA inputs | MLFF relax all 40 (200 calls × 48 ms each; 3 workers → 128 s) + MLFF Hessians (18 s total) + DF-r²SCAN screen (40 × 8 s / 3 = 107 s) → rank, run the G4 audit |
| 0.23 → 2.0 h | DFT re-optimisation of the **18 survivors** (10 kcal/mol window, G4-gated), MLFF-preconditioned: 18 × 12 × 29 s = 1.74 h | committee uncertainty, basin re-checks, PES maps around the running anchors, opportunistic MLFF maps of the culled 22 in case G4 fails |
| 2.0 → 3.2 h | MPQC CCSD(T)-F12 re-rank, top 10, as two concurrent 3-rank jobs ≈ 1.2 h | idle / next system |
| 2.9 → 3.9 h | Analytic Hessians, top 3, started as each converges ≈ 1.0 h, ~0.7 h non-overlapped | idle |
| **Total** | | **≈ 3.6 h** |

**Speedup versus CPU only: 3.1×. Versus naive offload: 1.7×.** `[E]` throughout.

**Where the 3.1× comes from, decomposed:** 40 → 18 structures reaching DFT (2.2×, from GPU pre-screening), 30 → 12 DFT cycles each (2.5×, from MLFF preconditioning), minus the 1.20× heterogeneity penalty, minus a non-parallelisable tail.

**The honest headline is Amdahl's law.** Once the optimisation stage is pipelined, **the coupled-cluster and Hessian tail is 53 % of the remaining wall time and no amount of GPU touches it** — ORCA has no GPU path and no GPU DLPNO exists. Further gain must come from Setup 3, not from this box.

**Risk-adjusted variant.** If G4 fails (ρ < 0.9) the cull is void and all 40 go to DFT: 40 × 12 × 29 s = 3.87 h, total ≈ **5.7 h, 1.9×** — still better than naive offload. **Report both numbers. The preconditioning benefit is robust; the culling benefit is conditional.**

---

## 8B. Job chaining and state reuse

### 8B.1 The executive finding

**The highest-value state transfer in this document is not the wavefunction. It is the converged geometry.** An SCF restart saves a fraction of one SCF. A converged cheap geometry saves whole optimisation cycles, and each cycle is one SCF *plus* one gradient — in the DLPNO geometry row each gradient is 6N = 60 single points, because ORCA's numerical gradient costs "6 × (number of atoms) × (time for one single point calculation)" ([ORCA numerical gradients](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/numericalgradients.html)). Removing 10 of 25 cycles removes 600 single points ≈ **150 h on one rank `[D]`** at this document's 15 min/point figure. That is the single largest lever anywhere in the document.

The second-highest is the isotopologue shortcut (§8B.4): **one force field serves every isotopologue**, converting an N-isotopologue campaign into a one-isotopologue cost.

### 8B.2 The master state inventory

| Code | File / object | Format | Contents | Written by | Read by | Notes and limits |
|---|---|---|---|---|---|---|
| ORCA 6.1 | `BaseName.gbw` | binary | geometry, basis, MO coefficients, occupations, energies | always — "the program stores the current orbitals in every SCF cycle" | `! MORead` + `%moinp "name.gbw"` | **Neither geometry nor basis need match the new job** — ORCA projects. The input `.gbw` **must have a different name** or it is overwritten "and all information is lost" ([initial guess](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/initialguess.html)) |
| ORCA 6.1 | same-named `.gbw` | binary | as above | as above | `%scf AutoStart true end` (default) | old file renamed `.ges`. **Ignored for geometry optimisations** — chained optimisations must use explicit `! MORead`; disable with `! NoAutoStart` when you need determinism |
| ORCA 6.1 | projection mode | — | how old MOs map to the new basis | — | `%scf GuessMode FMatrix` (default) or `CMatrix` (better for ROHF) | identical basis ⇒ only reorthogonalisation and renormalisation |
| ORCA 6.1 | old-release `.gbw` | binary | orbital coefficients only | — | `! rescue moread noiter` + `%moinp` | requires matching geometry **and** basis |
| ORCA 6.1 | `BaseName.hess` | text | Cartesian Hessian, masses, normal modes, frequencies, dipole derivatives | `! Freq` / `! NumFreq` | `%geom InHess Read InHessName "file.hess" end` | also accepts `.opt` and `.carthess` ([optimizations](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations.html)) |
| ORCA 6.1 | `BaseName.opt` / `.carthess` | binary/text | trajectory plus the **BFGS-updated** Hessian | every optimisation | `%geom InHess Read InHessName "basename.opt"` | the documented way to restart an interrupted optimisation; better than any model Hessian at that point |
| ORCA 6.1 | model Hessians | in memory | Almlöf / Lindh / Schlegel / Swart / **XTB0, XTB1, XTB2, GFNFF** | — | `%geom InHess Lindh` … `InHess XTB2` | costs seconds; see §8B.3 |
| ORCA 6.1 | `BaseName.res.%5d.Type` | text | per-displacement `Dipoles`, `Gradients`, `Ramans`, `Nacmes` | every `! NumFreq` | `%freq Restart true end` | the **only** frequency restart ORCA has. Level, basis and geometry must be unchanged — "Any change will produce an inconsistent, essentially meaningless Hessian" ([frequencies](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/frequencies.html)) |
| ORCA 6.1 | `.hess` present at the start of a `Freq` run | text | Hessian | — | automatic | **only the analysis is repeated** — free re-analysis with new isotopic masses. This is the isotopologue shortcut |
| ORCA 6.1 | relaxed-scan outputs | `.xyz` + `.gbw` per step | optimised structure and orbitals at every scan point | `%geom Scan … end` | `%moinp` for the next point | the state-reuse hook for the relaxed-scan rows |
| ORCA 6.1 | `BaseName.xtbw` | binary | atomic charges and multipoles for a fast xTB restart | native GFN*n*-xTB runs | auto-detected | "If such an `.xtbw` file is present, it will be used for the restart instead of the `.gbw` file" |
| ORCA 6.1 | `BaseName.mdrestart` | text | positions, velocities, thermostat state, metadynamics hills, step counters | **every MD step** | `%md Restart IfExists … end` | designed for "if the queuing system of the cluster imposes a maximum job time". Only those quantities restart; thermostats, regions, dumps, constraints and cells stay in the input |
| ORCA 6.1 | `BaseName.metarestart` | text | metadynamics hills | each new hill | metadynamics restart | metadynamics "can be easily restarted and split over" runs |
| ORCA 6.1 | `.finalensemble.xyz`, `.globalminimum.xyz`, `*.confrot.xyz` | XYZ, energy on the comment line | GOAT ensembles | `! GOAT` | GOAT can read an ensemble back — "nothing will be done, except that the filters will be reapplied" ([GOAT](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/goat.html)) | a genuine zero-cost reuse: re-filter with new thresholds without re-searching |
| ORCA 6.1 | conversion utilities | — | — | `orca_2mkl`, `orca_2json`, `orca_2aim`, **`orca_mergefrag`**, **`orca_vib`**, `orca_exportbasis` | — | `orca_mergefrag` merges MO coefficients from two converged monomer `.gbw` files into a complex guess; `orca_vib file.hess` re-diagonalises a stored Hessian with a new scale factor ([utilities](https://www.faccts.de/docs/orca/6.1/manual/contents/utilitiesvisualization/utilities.html)) |
| ORCA 6.1 | naming control | — | — | `%base "job1"` | — | **mandatory hygiene when chaining**: every stage owns its `%base` so a reader is never also the writer. Unix filenames are case-sensitive |
| ORCA 6.1 | compound scripts | — | geometry, MOs, Hessians between steps of one job | `New_Step … Step_End` | `Read_Geom(n);`, `ReadMOs(n);` | eliminates file plumbing when the whole chain fits one wall-clock window |
| **ORCA — cannot restart** | analytic frequencies | — | — | — | — | explicitly not restartable |
| **ORCA — cannot restart** | MDCI / DLPNO / canonical CC | — | — | — | — | MDCI exposes `MaxIter`, `STol`, `LShift`, `MaxDIIS` and **no restart facility** ([MDCI](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/mdci.html)). A killed CC job restarts only from the `.gbw`, losing all amplitudes |
| PySCF | `*.chk` | **HDF5** | `scf/e_tot`, `scf/mo_coeff`, `scf/mo_occ`, `scf/mo_energy`, `mol` | `mf.chkfile = '…'` | `mf.init_guess = 'chkfile'`, or `dm = scf.hf.from_chk(mol, chk)` | "there is no restart mechanism available in PySCF package, calculations can still be 'restarted' by reading in an earlier wave function as the initial guess" ([PySCF SCF](https://pyscf.org/user/scf.html)). Works **across different molecules and basis sets**. The file may be deleted on success unless set explicitly |
| PySCF | `pyscf.lib.chkfile` | HDF5 | arbitrary | `lib.chkfile.save(chk, key, value)` | `lib.chkfile.load(chk, key)` | keys are literal HDF5 paths; dicts and lists are stored recursively as groups ([PySCF lib API](https://pyscf.org/pyscf_api_docs/pyscf.lib.html)) |
| PySCF | CASSCF chkfile | HDF5 | `mcscf/mo_coeff` | `mycas.chkfile = "…"` | `chkfile.load(old, 'mcscf/mo_coeff')` | "for large calculations, it is always recommended to specify a checkpoint file" |
| PySCF | geomeTRIC hooks | — | per-step geometry, energy, gradient | `optimize(mf, callback=cb)` | user code | the callback is where each step is appended to your own HDF5 file. "except for the initial step, geomeTRIC does not support reading analytical Hessians at runtime" |
| gpu4pyscf | Python object handoff | in memory | full SCF/DFT object | `to_gpu` (PySCF ≥ 2.5.0) | — | `to_cpu` and chkfile behaviour are **not documented** in the README. Because gpu4pyscf subclasses PySCF objects, `mo_coeff` is a plain array and can be written to a chkfile by hand — **`[E]`: practical, not a documented API** |
| CFOUR | `JOBARC` + `JAINDX` | binary + index | the central archive: geometry, SCF eigenvectors, gradients, control data | every run | `GUESS=MOREAD` (default) reads the initial eigenvectors from `JOBARC`, falling back to `CORE` | redirect with `% JOBARC=/path/JOBARC` at the top of `ZMAT` ([CFOUR restarts](https://cfour.uni-mainz.de/cfour/index.php?n=Main.Non-standardFileHandlingAndRestartCalculations)) |
| CFOUR | `OPTARC` | binary | optimisation history | geometry optimisations | automatic | restart needs `JOBARC` + `JAINDX` + `OPTARC`; on a CPU-time-limit exit "resubmission auto-restarts, no keywords needed" |
| CFOUR | `MOINTS`, `MOABCD`, `GAMLAM`, `DERGAM`, `SECDER` | binary | transformed integrals, CC intermediates, derivative integrals | CC and derivative runs | CC restart needs `JOBARC`, `JAINDX`, `MOINTS`, `MOABCD` | **This is the capability ORCA lacks** and the reason the 3 d / 1 w / 1 mo coupled-cluster rows belong in CFOUR under a 24–48 h queue |
| CFOUR | `FCMINT` / `FCMFINAL` | text | initial / final force-constant matrix | user-supplied or previous job; both analytic and finite-difference frequency runs | read as the initial Hessian when updating is off; consumed downstream | the carrier of a harmonic force field into a hybrid VPT2 field |
| CFOUR | finite-difference control | — | — | `VIB=FINDIF`, `FD_CALCTYPE`, `FD_STEPSIZE`, `FD_PROJECT`, `FD_IRREP(S)` | — | **restart needs only `JOBARC` + `JAINDX`**, and `FD_IRREP` is the decomposition axis under a wall-clock cap |
| CFOUR | `ISOMASS` + `xjoda` | — | isotopic masses | — | re-run `xjoda` against a saved `JOBARC` | **the microwave isotopologue workflow**: one force field, many isotopologues, no new electronic structure |
| xtb | `--restart` | — | previous SCC state | `xtb coord --restart`; output prints a `restarted?` field | | the commonly cited `xtbrestart` filename is **not documented** — cite the flag, not the file ([xtb single point](https://xtb-docs.readthedocs.io/en/latest/sp.html)) |
| xtb | `hessian` | Turbomole | projected Hessian | `--hess`, `--ohess`, `--bhess` | `--thermo <FILE>` | frequencies come from two-sided numerical differentiation of analytic gradients ([xtb Hessian](https://xtb-docs.readthedocs.io/en/latest/hessian.html)) |
| xtb | `g98.out`, `xtbhess.coord` | Gaussian / coord | modes; **structure distorted along the imaginary mode** | `--hess` on a non-stationary structure | feed `xtbhess.coord` back to the optimiser | the automatic escape hatch when a chained geometry lands off the minimum |
| xtb | `.CHRG`, `.UHF`, `.xtbrc`, `xtb.inp` | text | charge, multiplicity, configuration | `--define --copy` | `--input xtb.inp` | **carry `.CHRG`/`.UHF` with the geometry through every stage or the chain silently changes the species** |
| CREST | `crest_conformers.xyz`, `crest_rotamers_*.xyz`, `crest.vibspectrum` | XYZ / Turbomole | conformer and rotamer ensembles | every run | `--cregen <FILE>` re-sorts standalone; `--thermo <FILE>` | **there is no general CREST restart file**; `--mrest <INT>` bounds MTD restart cycles (default 5). *(Correction: there is no `.ensemble` format — v3's filename was wrong.)* |
| ASE | `*.traj` | ASE binary | per-image positions plus energy, forces, stress, dipole, charges, magmoms | `Trajectory(fn,'w',…)`, `dyn.attach(traj.write, interval=100)` | `Trajectory(fn)`, `traj[-1]` | the right per-optimisation log ([ASE Trajectory](https://wiki.fysik.dtu.dk/ase/ase/io/trajectory.html)) |
| ASE | `ase.db` | **JSON, SQLite3, PostgreSQL, MySQL, MariaDB** | structures and properties, queryable | `db.write(atoms, …)` | `db.select(...)` | **No HDF5 back-end** ([ASE database](https://wiki.fysik.dtu.dk/ase/ase/db/db.html)). v3 implied otherwise. Use ASE db as the *catalogue* and HDF5 as the *bulk array store* |
| MACE / MLFF | model checkpoint + `.traj` | PyTorch + ASE | the fitted surface; cached geometries | training / MD | ASE calculator | the reusable state is the *geometry set* and, at the fine-tuning tier, the checkpoint itself — archive it |

### 8B.3 What each reuse actually saves

**MO guess → SCF iterations.** ORCA projects old orbitals into the new basis; PySCF's chkfile guess does the same across molecules and basis sets. **No vendor publishes an iteration-count saving, so the honest entry is `n.a.` for a cited number.** What can be stated is the shape: ORCA's optimisation default is already `OptGuess MORead`, "Use MOs of the previous point", so **inside an optimisation there is nothing to add**. The gain lives at *inter-job* boundaries — r²SCAN-3c → ωB97X-V, TZ → QZ, and each counterpoise leg — and at PES grid points, where every displaced geometry is a ~0.005 bohr perturbation of a converged one and the neighbour's `.gbw` is an almost-exact guess. **Planning value: 30–60 % of SCF cycles at a stage boundary `[E]` — measure it, do not assume it.**

**Converged cheap geometry → optimisation cycles.** This is the one with real published numbers, and they point in both directions, which is itself the finding.

*For.* Force-field-preconditioned optimisation — mathematically the same act as importing curvature information from a cheaper model — gives "at least a 2-fold, typically 5-fold decrease" in function and gradient calls ([Mones, Ortner & Csányi](https://pmc.ncbi.nlm.nih.gov/articles/PMC6143621/)). Gas-phase steps, identity versus GAFF preconditioner, all `[M]`: 5-nitrobenzisoxazole at PM6 89 → 25, at BLYP 119 → 63, at MP2 71 → 27; menthone 207 → 29; alanine tripeptide 395 → 77; THC 720 → 84; heme 500 → 175; taxol 1662 → 419; 16-mer polyalanine 3549 → 348. Saddle searches improve too: 395 → 127 and 531 → 147.

*Against.* A controlled single-system test found **no** benefit: a 45-atom high-entropy-alloy slab at GPAW/PBE from a GFN1-xTB pre-optimised geometry gave BFGS 16 → 17, LBFGS 15 → 16, QuasiNewton 8 → 9, GPMin 12 → 31, FIRE 38 → 44 `[M]`, with only MLMin improving ([Vilarrasa-García](https://doublelayer.eu/vilab/2023/04/18/dft-geometry-optimizers/)).

*Reconciliation, and the rule.* Preconditioning wins on large floppy systems with many soft coordinates; the counterexample is a metal slab whose cheap and expensive minima genuinely differ. **A 5–10 atom van der Waals complex is the floppy case** — one very soft intermolecular coordinate dominates the Hessian spectrum, and locating it cheaply is exactly what removes cycles. But the counterexample is the warning: if the cheap method places the intermolecular minimum in the wrong place, you have not saved cycles, you have donated a bad starting point. GFN2-xTB's S22 centre-of-mass maximum error is 32 pm, which at ΔB/B = 2ΔR/R is a ~14 % error in B — **fine as a starting point, catastrophic as a reported one** (rule D1 below).

**Cheap Hessian → initial Hessian: usually not worth computing on purpose.** ORCA's own position: "the use of the exact Hessian as initial one is only of little help, since in these cases the convergence is usually only slightly faster, while at the same time much more time is spent in the calculation of the initial Hessian" ([optimizations](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations.html)). The preconditioner data agree exactly: an **exact PM6 Hessian** gave 24/21/110/120 steps on four systems against 32/36/109/95 for a **free GAFF model** — better on two, worse on two `[M]` ([Mones *et al.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6143621/)). The vendor counter-claim (a DFTB Hessian fed into a DFT optimisation, "If we save just a single step there, the initial DFTB calculation will already have paid for itself") **publishes no step counts** ([SCM two-step geometry optimisation](https://www.scm.com/doc/AMS/Examples/2StepGO.html)).

> **Binding rule.** Use `%geom InHess XTB2 end` (or `Lindh`) on every DFT and post-Hartree–Fock optimisation. **`Calc_Hess true` is forbidden unless the Hessian is itself a deliverable** — a frequency calculation, a VPT2 force field, or a transition-state search. This replaces every `Calc_Hess true` in v3.
>
> **The one endorsed exception is the frequency/VPT2 chain**, where the Hessian is the product and a low-level Hessian is the standard construction: ORCA endorses "first calculate the Hessian at a lower level of theory or with a smaller basis set and use this Hessian as input", and the anharmonic-force-field literature has a named construction for it (below).
>
> **A second endorsed exception is an MLFF Hessian used as a preconditioner** (§8A.3), which is free in wall clock because it runs on the other device.

**Lower-level Hessian → higher-level frequencies / VPT2.** Here reuse is the standard method, with a taxonomy ([How to VPT2](https://par.nsf.gov/servlets/purl/10284833)):

| Construction | What is reused | Cost | Verdict |
|---|---|---|---|
| Additive | low-level quartic field + a high-level harmonic correction added | low | superseded |
| **Substituted** ✅ | low-level cubic/quartic constants retained, harmonic constants **replaced** by high-level ones | "does not require any additional cost" | **Recommended.** Assumes identical normal coordinates at both levels |
| Transformed | as substituted, plus transformation into the high-level normal coordinates | "several times more expensive" | removes the identical-coordinate assumption |
| Direct evaluation in higher-level coordinates | — | — | errors from a non-stationary geometry; "not well studied" |

Accuracy anchors from the same source, all `[M]`: small-basis C–H stretches come out too high "by 10–20 cm⁻¹ or more"; cyclopentadiene CCSD(T)/ANO lands within ~0–19 cm⁻¹; ethylene within ~10 cm⁻¹; water VPT2+K overtones within 1.4 cm⁻¹.

**The floppy-complex caveat is explicit and it is ours.** Substituted hybrids "degrade for larger, lower-symmetry systems where normal-coordinate character is method-sensitive", and are catastrophic under severe large-amplitude motion, proton-bound dimers being the named example. **A 5–10 atom complex with a 30 cm⁻¹ intermolecular bend is exactly that regime.** Hybrid force fields are therefore permitted on the *semi-rigid manifold* and forbidden as a treatment of the large-amplitude coordinate.

**Basis-set cascading.** Converge TZ, read the TZ `.gbw` into the QZ job; ORCA projects with `GuessMode FMatrix`. Hard constraint: do not use `moread noiter` if redundant basis components were removed. No cited iteration saving — **worthwhile, small, and free to try `[E]`**.

### 8B.4 The canonical chained pipeline

| # | From → To | File passed | Keyword that consumes it | Saving |
|---|---|---|---|---|
| 1 | MLFF/xTB GOAT search → CREST cross-check | `BaseName.finalensemble.xyz` | `crest --cregen ens.xyz --ethr 0.05 --bthr 0.01 --rthr 0.125`; or fed back to GOAT, which re-applies filters only | re-filtering costs zero gradients — **100 % of the search cost avoided on every threshold change** |
| 2 | Ensemble → GFN2-xTB refinement | `.xyz` per conformer, with `.CHRG`/`.UHF` travelling alongside | `xtb conf.xyz --opt vtight --strict`, or ORCA `! XTB2 TightOpt` reusing `.xtbw` | seconds; the point is to reach r²SCAN-3c with a sane intermolecular distance |
| 3 | GFN2-xTB → r²SCAN-3c optimisation | `xtbopt.xyz` **plus a GFN2 model Hessian, not a computed one** | geometry inline; `%geom InHess XTB2 end` | **≥2×, typically ~5× fewer optimisation steps** on floppy systems by analogy with the preconditioner data — **the transfer of that factor to this class is `[E]`** |
| 4 | r²SCAN-3c → ωB97X-V/def2-TZVPP tight optimisation | `s2.xyz` + `s2.gbw` + `s2.opt` | geometry inline; `! MORead` + `%moinp "s2.gbw"`; `%geom InHess Read InHessName "s2.opt" end` | cycles removed (dominant); `.opt` carries the *updated* BFGS Hessian, better than any model Hessian at that point |
| 5 | ωB97X-V/TZ → ωB97M-V/def2-QZVPP | `s3.gbw` (different basis — ORCA projects) | `! MORead` + `%moinp "s3.gbw"`, `%scf GuessMode FMatrix end` | the expensive QZ SCF starts from a converged TZ density `[E]` |
| 6 | Tight optimisation → analytic DFT Hessian | `s4.xyz` (**must be the same geometry**) + `s4.gbw` | `! Freq` at the identical level; `! MORead` | skips re-converging the SCF. **A Hessian on a geometry that is not stationary at that level is meaningless** (rule D2) |
| 7 | **Hessian → all isotopologues** | `s5.hess` | re-run `Freq` with the `.hess` present → analysis only; or `orca_vib s5.hess`; or CFOUR `ISOMASS` + `xjoda` on a saved `JOBARC` | **N isotopologues for the price of one force field — a 6–15× saving `[D]` on a parent + ¹³C + D campaign** |
| 8 | Tight optimisation → high-level single point | `s4.xyz` + `s4.gbw` | `! DLPNO-CCSD(T1) … MORead` + `%moinp "s4.gbw"` | saves the SCF only. **No CC restart exists in ORCA** — if the CC dies you repeat it |
| 9 | Counterpoise legs | one dimer geometry + one exported basis (`orca_exportbasis`, read back with `%basis GTOName …`) | ghost atoms with `:` after the element symbol | guarantees the three legs share an identical basis. **`.gbw` reuse is valid leg-to-leg, never dimer → monomer** (rule D4) |
| 10 | DFT force field → CFOUR anharmonic VPT2 | `FCMINT` in, `FCMFINAL` out | `FCMINT` read when Hessian updating is off | **substituted hybrid force field: high-level harmonic + low-level anharmonic, at no additional cost** — permitted only on the semi-rigid manifold |
| 11 | Any → the next step of the same ORCA process | geometry and MOs implicitly | `Read_Geom(n);` and `ReadMOs(n);` inside `New_Step … Step_End` | eliminates all file plumbing when the chain fits one wall-clock window |

**Working driver — shell.**

```bash
#!/usr/bin/env bash
# chain.sh -- canonical vdW-complex state-chaining pipeline (ORCA 6.1)
# Each stage owns its %base so a reader is never also the writer:
# ORCA overwrites a same-named input .gbw "and all information is lost".
set -euo pipefail

SEED=${1:?usage: chain.sh seed.xyz}
CHG=${CHG:-0}; MUL=${MUL:-1}; NPROC=${NPROC:-7}; MEM=${MEM:-3400}
WORK=$(pwd)/chain; mkdir -p "$WORK"; cd "$WORK"

run () {  # run <stagename>   (reads <stagename>.inp)
  echo "== $1 =="; orca "$1.inp" > "$1.out" 2> "$1.err"
  grep -q "ORCA TERMINATED NORMALLY" "$1.out" || { echo "FAILED: $1"; exit 1; }
}

# ---- Stage 1: GFN2-xTB pre-optimisation ------------------------------------
xtb "$SEED" --opt vtight --strict --chrg "$CHG" --uhf $((MUL-1)) > s1_xtb.out
cp xtbopt.xyz s1.xyz

# ---- Stage 2: r2SCAN-3c, xTB MODEL Hessian as preconditioner ---------------
cat > s2.inp <<EOF
! r2SCAN-3c TightOpt TightSCF DefGrid3
%base "s2"
%pal nprocs $NPROC end
%maxcore $MEM
%geom
  InHess XTB2          # model Hessian; an exact one is "only of little help"
  TolE 1e-7  TolRMSG 3e-6  TolMaxG 1e-5  TolRMSD 5e-5  TolMaxD 1e-4
  MaxIter 200
end
* xyzfile $CHG $MUL s1.xyz
EOF
run s2

# ---- Stage 3: wB97X-V/def2-TZVPP: geometry + MOs + updated Hessian ---------
cat > s3.inp <<EOF
! wB97X-V def2-TZVPP def2/J RIJCOSX TightOpt TightSCF DefGrid3 MORead
%base "s3"
%moinp "s2.gbw"                 # projected onto the new basis (GuessMode FMatrix)
%pal nprocs $NPROC end
%maxcore $MEM
%geom
  InHess Read
  InHessName "s2.opt"           # the BFGS-updated Hessian from stage 2
  TolE 1e-7  TolRMSG 3e-6  TolMaxG 1e-5  TolRMSD 5e-5  TolMaxD 1e-4
  MaxIter 200
end
* xyzfile $CHG $MUL s2.xyz
EOF
run s3

# ---- Stage 4: wB97M-V/def2-QZVPP, MO cascade across basis ------------------
cat > s4.inp <<EOF
! wB97M-V def2-QZVPP def2/J RIJCOSX TightOpt TightSCF DefGrid3 MORead
%base "s4"
%moinp "s3.gbw"
%pal nprocs $NPROC end
%maxcore $MEM
%geom
  InHess Read
  InHessName "s3.opt"
  TolE 1e-7  TolRMSG 3e-6  TolMaxG 1e-5  TolRMSD 5e-5  TolMaxD 1e-4
end
* xyzfile $CHG $MUL s3.xyz
EOF
run s4

# ---- Stage 5: ANALYTIC Hessian at the SAME level and SAME geometry ---------
# Not restartable. Must fit one wall-clock window (see 8B.6).
cat > s5.inp <<EOF
! wB97M-V def2-QZVPP def2/J RIJCOSX Freq TightSCF DefGrid3 MORead
%base "s5"
%moinp "s4.gbw"
%pal nprocs $NPROC end
%maxcore $MEM
* xyzfile $CHG $MUL s4.xyz
EOF
run s5

# ---- Stage 5b: every isotopologue from the SAME Hessian, for free ----------
for iso in 13C 18O D; do
  cp s5.hess "iso_$iso.hess"
  orca_vib "iso_$iso.hess" > "iso_$iso.vib.out" || true
done

# ---- Stage 6: high-level single point on the stage-4 geometry --------------
cat > s6.inp <<EOF
! DLPNO-CCSD(T1) TightPNO cc-pVDZ-F12 (paired with CABS) cc-pVDZ-F12 (paired with CABS)/C def2/JK TightSCF MORead
%base "s6"
%moinp "s4.gbw"
%pal nprocs $NPROC end
%maxcore $MEM
%mdci TCutPNO 1e-7 DoLED true StorageType Shared end
* xyzfile $CHG $MUL s4.xyz
EOF
run s6          # if this dies, the CC restarts from nothing: no MDCI restart

echo "chain complete: geometry s4.xyz, Hessian s5.hess, energy in s6.out"
```

**Working driver — Python, recording every arrow into one HDF5 file.**

```python
#!/usr/bin/env python3
"""chain.py -- state-chaining driver that records every arrow into one HDF5 file.

Design rules encoded here:
  * every stage gets its own %base, so no job reads and writes the same .gbw
  * geometry is the primary state; MOs and Hessians are secondary
  * each stage records provenance (level, files consumed, files produced)
"""
from __future__ import annotations
import json, shutil, subprocess, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np, h5py

TIGHT = ("  TolE 1e-7\n  TolRMSG 3e-6\n  TolMaxG 1e-5\n"
         "  TolRMSD 5e-5\n  TolMaxD 1e-4\n")


@dataclass
class Stage:
    name: str
    level: str                     # the ! line
    blocks: str = ""               # extra %-blocks
    geom_from: str | None = None   # stage whose optimised geometry we start from
    mo_from: str | None = None     # stage whose .gbw we project
    hess_from: str | None = None   # stage whose .opt/.hess seeds the optimiser


def read_xyz(p: Path):
    lines = p.read_text().splitlines()
    n = int(lines[0].split()[0])
    sym, xyz = [], []
    for ln in lines[2:2 + n]:
        f = ln.split()
        sym.append(f[0]); xyz.append([float(x) for x in f[1:4]])
    return sym, np.asarray(xyz, float)


def parse_hess(p: Path):
    """Minimal ORCA .hess reader: returns the Cartesian Hessian (3N x 3N)."""
    if not p.exists():
        return None
    tok = p.read_text().split("\n")
    for i, ln in enumerate(tok):
        if ln.strip() == "$hessian":
            n = int(tok[i + 1].split()[0])
            H = np.zeros((n, n)); j = i + 2; col = 0
            while col < n:
                cols = [int(c) for c in tok[j].split()]; j += 1
                for r in range(n):
                    vals = tok[j + r].split()
                    for k, c in enumerate(cols):
                        H[int(vals[0]), c] = float(vals[k + 1])
                j += n; col += len(cols)
            return H
    return None


def final_energy(out: Path):
    e = None
    for ln in out.read_text().splitlines():
        if "FINAL SINGLE POINT ENERGY" in ln:
            e = float(ln.split()[-1])
    return e


class Chain:
    def __init__(self, workdir="chain", h5="campaign.h5", charge=0, mult=1,
                 nproc=7, maxcore=3400):
        self.w = Path(workdir); self.w.mkdir(exist_ok=True)
        self.h5 = self.w / h5
        self.charge, self.mult, self.nproc, self.maxcore = charge, mult, nproc, maxcore

    def _input(self, st: Stage, geom_file: str) -> str:
        head = f"! {st.level}" + (" MORead" if st.mo_from else "")
        body = [head, f'%base "{st.name}"',
                f"%pal nprocs {self.nproc} end", f"%maxcore {self.maxcore}"]
        if st.mo_from:
            body.append(f'%moinp "{st.mo_from}.gbw"')
        geomblk = [TIGHT.rstrip("\n")]
        if st.hess_from:
            src = (f"{st.hess_from}.opt" if (self.w / f"{st.hess_from}.opt").exists()
                   else f"{st.hess_from}.hess")
            geomblk += ["  InHess Read", f'  InHessName "{src}"']
        else:
            geomblk += ["  InHess XTB2"]          # cheap model Hessian by default
        body.append("%geom\n" + "\n".join(geomblk) + "\nend")
        if st.blocks:
            body.append(st.blocks)
        body.append(f"* xyzfile {self.charge} {self.mult} {geom_file}")
        return "\n".join(body) + "\n"

    def run(self, st: Stage, seed_xyz: str | None = None) -> dict:
        geom_file = f"{st.geom_from}.xyz" if st.geom_from else seed_xyz
        assert geom_file, "stage needs geom_from or a seed"
        inp = self.w / f"{st.name}.inp"; inp.write_text(self._input(st, geom_file))
        t0 = time.time()
        with (self.w / f"{st.name}.out").open("w") as fh:
            subprocess.run(["orca", inp.name], cwd=self.w, stdout=fh,
                           stderr=subprocess.STDOUT, check=True)
        wall = time.time() - t0
        out = self.w / f"{st.name}.out"
        assert "ORCA TERMINATED NORMALLY" in out.read_text(), f"{st.name} failed"
        gx = self.w / f"{st.name}.xyz"
        if not gx.exists():                       # single points emit no .xyz
            shutil.copy(self.w / geom_file, gx)
        sym, xyz = read_xyz(gx)
        rec = dict(stage=st.name, level=st.level, wall_s=wall,
                   energy=final_energy(out), symbols=sym, geometry=xyz,
                   hessian=parse_hess(self.w / f"{st.name}.hess"),
                   consumed=[f for f in (st.geom_from and f"{st.geom_from}.xyz",
                                         st.mo_from and f"{st.mo_from}.gbw",
                                         st.hess_from and f"{st.hess_from}.opt") if f],
                   produced=[p.name for p in self.w.glob(f"{st.name}.*")])
        self._record(rec)
        return rec

    def _record(self, rec: dict):
        with h5py.File(self.h5, "a") as f:
            g = f.require_group(f"chain/{rec['stage']}")
            g.attrs["level"] = rec["level"]
            g.attrs["wall_s"] = rec["wall_s"]
            g.attrs["consumed"] = json.dumps(rec["consumed"])
            g.attrs["produced"] = json.dumps(rec["produced"])
            g.attrs["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if rec["energy"] is not None:
                g.attrs["energy_hartree"] = rec["energy"]
            for k, v in (("geometry", rec["geometry"]), ("hessian", rec["hessian"])):
                if v is None:
                    continue
                if k in g:
                    del g[k]
                g.create_dataset(k, data=np.asarray(v, "f8"),
                                 compression="gzip", compression_opts=4, shuffle=True)
            g.attrs["symbols"] = json.dumps(rec["symbols"])


if __name__ == "__main__":
    import sys
    seed = sys.argv[1]
    c = Chain()
    s2 = Stage("s2", "r2SCAN-3c TightOpt TightSCF DefGrid3")
    s3 = Stage("s3", "wB97X-V def2-TZVPP def2/J RIJCOSX TightOpt TightSCF DefGrid3",
               geom_from="s2", mo_from="s2", hess_from="s2")
    s4 = Stage("s4", "wB97M-V def2-QZVPP def2/J RIJCOSX TightOpt TightSCF DefGrid3",
               geom_from="s3", mo_from="s3", hess_from="s3")
    s5 = Stage("s5", "wB97M-V def2-QZVPP def2/J RIJCOSX Freq TightSCF DefGrid3",
               geom_from="s4", mo_from="s4")
    s6 = Stage("s6", "DLPNO-CCSD(T1) TightPNO cc-pVDZ-F12 (paired with CABS) cc-pVDZ-F12 (paired with CABS)/C TightSCF",
               geom_from="s4", mo_from="s4",
               blocks="%mdci TCutPNO 1e-7 DoLED true StorageType Shared end")
    c.run(s2, seed_xyz=seed)
    for st in (s3, s4, s5, s6):
        c.run(st)
    print("done ->", c.h5)
```

### 8B.5 Dangerous reuses: rules D1–D5

**D1 — A geometry whose intermolecular error exceeds the target, reported as the higher level.**
*Rule.* A geometry may be **passed forward as a starting point at any level**. It may be **reported**, or used for a rigid property evaluation attributed to level *L*, **only if it is stationary at level *L*** to the §4.4 convergence block. Because ΔB/B = 2ΔR/R, a ±0.1 % window in B is a ±0.05 % window in R — at R ≈ 3.5 Å that is **±1.8 pm**. GFN2-xTB's S22 centre-of-mass maximum error is 32 pm and r²SCAN-3c's S66 centre-of-mass mean absolute deviation is 5.6 pm: **neither is inside the window; both are excellent starting points.**
*Detection.* (a) The gradient norm at the inherited geometry — xtb checks it automatically, printing "Hessian on incompletely optimized geometry!" and writing `xtbhess.coord`; **`--strict` turns the warning into a non-zero exit and is mandatory in an automated chain** ([xtb Hessian](https://xtb-docs.readthedocs.io/en/latest/hessian.html)). (b) Report ΔR between consecutive stages **in MHz of ΔB**, not in pm. (c) Any imaginary frequency at the high level means the inherited geometry was in the wrong basin; restart from `xtbhess.coord`.

**D2 — A Hessian reused at a different geometry or method, reported as the higher level.**
*Rule.* A Hessian is a second derivative *at a point*. Reusing it as an optimiser preconditioner is always safe — the optimiser corrects it. Reusing it as the *reported* force field is safe only under the **substituted hybrid** construction, and only where the normal coordinates are genuinely level-insensitive. For a restarted numerical frequency ORCA is categorical: level, basis and geometry must be unchanged, because "Any change will produce an inconsistent, essentially meaningless Hessian".
*Detection.* (a) Count imaginary modes; a Hessian evaluated off-stationarity typically produces spurious small imaginary frequencies. (b) Compare the two lowest harmonic frequencies at the low and high level **on the same geometry** — if the intermolecular bend shifts by more than ~20 %, the normal coordinates are method-sensitive and the substituted hybrid is invalid for this complex (**a working rule `[E]`, not a published criterion**). (c) Project translations and rotations and confirm six near-zero eigenvalues — necessary, not sufficient. (d) **Flag any mode below ~100 cm⁻¹ and exclude it from the hybrid treatment.**

**D3 — SCF converging to a different, or symmetry-broken, solution from a reused density.**
*Rule.* A reused density is a *bias*, not just a speed-up. If the projected density sits in a different SCF basin, the SCF converges there and reports success. **Never trust a reused-guess SCF energy without a stability check.**
*Detection.* (a) Run `! Stability`; ORCA can restart automatically "using new unrestricted start orbitals" via `STABRestartUHFifUnstable true` ([stability analysis](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/stabilityanalysis.html)); CFOUR has `HFSTABILITY=ON|FOLLOW`. (b) Re-run one point per stage **with `! NoAutoStart`** and confirm agreement to the convergence threshold — **the cheapest audit in the pipeline; make it a mandatory spot-check on ~5 % of points.** (c) A reused guess that takes *more* iterations than a fresh one is a basin-change signature. (d) Check ⟨S²⟩ and the orbital-energy gap against the fresh-guess run.

**D4 — Counterpoise and ghost-atom inconsistency across a chain.**
*Rule.* Boys–Bernardi requires each fragment energy at the dimer geometry in the dimer basis, which ORCA implements with a `:` after the element symbol. Three consequences: (1) **never reuse a dimer `.gbw` as the guess for a ghosted monomer leg** — the orbital count differs and the occupancies are wrong, and ORCA will project, converge to something, and report it; (2) **pin the basis** by exporting it once with `orca_exportbasis` and reading the identical `.bas` in all three legs; (3) **a counterpoise-corrected energy and an uncorrected geometry are a mismatched pair** — record counterpoise status as a *method attribute* in the HDF5 store (`counterpoise: none | half | full`) so a later fit cannot silently mix them.
*Detection.* Assert `n_electrons(monomer leg) == n_electrons(isolated monomer)` and `n_basis_functions(monomer leg) == n_basis_functions(dimer)` from every leg's output header. Report raw, CP and half-CP for every isomer; a CP term that jumps between chained stages is the basis-inconsistency alarm.

**D5 — Silent state contamination from a same-named file.**
*Rule.* An input `.gbw` with the same name as the job "is overwritten … and all information is lost", and `AutoStart` will silently pick up a stale same-named `.gbw`. **Every stage gets a unique `%base`, and re-run directories are never reused.** Unix filenames are case-sensitive, which is a documented ORCA gotcha.

### 8B.6 Restart under wall-clock caps

**The caps.** GitHub-hosted runner jobs are limited to **6 hours** and are *terminated*, not suspended, at the limit; self-hosted jobs get 5 days; a run may span 35 days; a matrix generates at most 256 jobs; concurrency is 20/40/60/500 by plan; artifact storage is 500 MB (Free) with a 10 GB cache ([GitHub Actions limits](https://docs.github.com/en/actions/reference/limits)). HPC queues here are 24–48 h.

| Job type | Code | Restartable? | Mechanism | Under a 6 h cap |
|---|---|---|---|---|
| SCF single point | ORCA | yes, effectively | `.gbw` written every cycle; re-run or `MORead` | fine — relaunch |
| SCF | PySCF | yes, effectively | chkfile guess | fine |
| Geometry optimisation | ORCA | **yes** | `basename.opt` / `.carthess` via `InHess Read` | **chain N × 6 h jobs from the last `.xyz` + `.gbw` + `.opt`** |
| Geometry optimisation | CFOUR | **yes** | `JOBARC` + `JAINDX` + `OPTARC`; on a time-limit exit "resubmission auto-restarts, no keywords needed" | fine — cache the three files |
| Numerical frequency | ORCA | **yes** | `%freq Restart true end` reading `BaseName.res.*` | fine; also cut work with `NumHessTransInvar` (saves 6 gradients two-sided, 3 one-sided) |
| **Analytic frequency** | ORCA | **NO** | explicitly not restartable | **must fit one job.** Otherwise switch to `NumFreq` and accept the cost, or move to CFOUR |
| Numerical gradient | ORCA | not as a unit | but each displacement is an independent single point, and ORCA can run "each displacement as a parallel calculation" | **decompose** — 6N = 60 displacements at N = 10 fits a 256-job matrix trivially |
| Finite-difference frequency | CFOUR | **yes** | restart needs only `JOBARC` + `JAINDX`; **`FD_IRREP` partitions the run by irreducible representation** | **best in class** — both a checkpoint *and* a decomposition axis |
| **DLPNO / canonical CC** | ORCA | **NO** | MDCI documents no restart | **must fit one job**; a 1 w-tier canonical CC job is out of reach on Setup 1 |
| CC | CFOUR | **yes** | `JOBARC`, `JAINDX`, `MOINTS`, `MOABCD` | **the reason the 3 d / 1 w / 1 mo CC rows belong in CFOUR under a 24–48 h queue** |
| MD / PIMD | ORCA | **yes** | `basename.mdrestart` written every step, loaded by `%md Restart IfExists end`; designed for exactly this | ideal for chained short jobs |
| Metadynamics | ORCA | yes | `basename.metarestart` | fine |
| Conformer search | CREST | **no general restart** | `--mrest`; `--cregen` re-filters; `--keepdir`/`--scratch` preserve intermediates | **decompose by seed**, one job per seed, union the ensembles |
| GOAT | ORCA | no documented mid-run restart | an ensemble file can be read back for re-filtering only | decompose by seed; union and re-filter |
| PES grid | any | N/A — embarrassingly parallel | the HDF5 `todo()` pattern of §8C | **the canonical decomposition**: 960 points → 256-job matrix × 4 waves, one HDF5 shard per job, one merge job |
| MLFF fine-tuning | MACE | yes | framework checkpoints per epoch | chain 6 h jobs; cache the checkpoint |

**Job types that cannot be checkpointed and must be decomposed:** ORCA analytic frequencies, ORCA MDCI/CC, CREST and GOAT searches, and any single DLPNO point exceeding the cap. **The decomposition unit is always an independent displacement or grid point** — which is exactly why the numerical-gradient and PES rows survive a 6 h cap and the analytic-Hessian rows do not.

**Worked decomposition example — CFOUR `FD_IRREP`.** A finite-difference frequency job in CFOUR restarts from `JOBARC` + `JAINDX` alone, and `FD_IRREP` restricts a run to selected irreducible representations. Under a hard cap, submit one job per irrep with `FREQ_ALGORITHM=PARALLEL`, `ANH_ALGORITHM=PARALLEL` and `FD_PROJECT=OFF`, cache `JOBARC`/`JAINDX` between them, and assemble with `xjoda`, `xsymcor`, `xja2fja` and `xcubic`. **Note the C₁ caveat**: a floppy complex usually has no symmetry, so `FD_IRREP` gives one irrep and no decomposition. In that case the decomposition axis is the *displacement*, driven externally.

**A correction to v3's artifact rule.** "Never `.gbw` as an artifact" is right for *artifacts* and wrong for *chaining*. Carry `.gbw`, `.opt` and `.hess` between chained 6 h jobs through the **Actions cache** (10 GB, keyed on a hash of the input), and upload only `.xyz`, `.hess` and the compressed HDF5 shard as artifacts with an explicit `retention-days`.

---

## 8C. The HDF5 PES store

### 8C.1 Why HDF5, and where it actually lives in this stack

A potential-energy-surface campaign asks for 500–2,000 coupled-cluster points and 960-point two-dimensional grids, and the geometry rows generate ~1,500 single points for 25 optimisation cycles. As a directory of `.out` files that is ~10⁴ inodes per complex, with no index, no typed access, and no atomic way to ask "which points converged at `TCutPNO 1e-7`?"

**Correction to v3 first.** ASE's `db` has five documented back-ends — JSON, SQLite3, PostgreSQL, MySQL and MariaDB — and **HDF5 is not among them** ([ASE database documentation](https://wiki.fysik.dtu.dk/ase/ase/db/db.html)). The genuine HDF5 layer in this stack is **PySCF's chkfile**, written and read through `pyscf.lib.chkfile` with `/`-separated HDF5 paths, with dicts and lists stored recursively as groups and `lib.misc.H5FileWrap` wrapping `h5py.File` ([PySCF lib API](https://pyscf.org/pyscf_api_docs/pyscf.lib.html)). PySCF's `ao2mo` can also stream transformed integrals straight into an HDF5 dataset. **Use ASE `db` as the queryable catalogue and HDF5 as the bulk array store.**

**The two mechanics that matter, both documented** ([h5py Datasets](https://docs.h5py.org/en/stable/high/dataset.html)):

- **Chunked storage** divides a dataset into regularly sized pieces indexed by a B-tree. Chunking is what makes datasets *resizable* and what makes *compression filters* possible at all. A campaign grows point by point, so resizability is not optional: create with `maxshape=(None, …)` and `resize()` as points arrive.
- **Compression.** `gzip` is "available with every installation of HDF5", gives "good compression, moderate speed", and takes `compression_opts` 0–9 with a default of 4; `lzf` is "low to moderate compression, very fast"; the `shuffle` filter "may improve the compression ratio" and "has no significant speed penalty"; `fletcher32` adds a per-chunk checksum so corrupted reads fail loudly instead of poisoning a fit.
- **Chunk sizing.** The documented range is **10 KiB – 1 MiB**. For an N = 10 complex a geometry row is 10 × 3 × 8 B = 240 B, so **chunk 512 points at a time** → ~120 KiB, squarely in range.
- **Do not use the lossy `scaleoffset` filter on energies.** It "does not preserve special floating-point values" including `NaN` and `inf`, and a microhartree-level surface cannot afford truncated mantissas.

**Metadata vocabulary: borrow QCSchema rather than inventing one.** Its input components are `molecule` (`geometry`, `symbols`), `driver` (energy | gradient | hessian | properties), `model` (`method`, `basis`) and `keywords`; its output components are `success` (with `error`/`error_type`), `return_result`, `provenance` (`creator`, `version`, `routine`) and `properties` (e.g. `calcinfo_nbasis`, `scf_total_energy`, `nuclear_repulsion_energy`) — and "the input components are duplicated in the output so that the result is a complete trace of the requested computation from input specification to results" ([QCSchema specification components](https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html)). Store those names as HDF5 attributes and any QCArchive-aware tool can read the campaign.

**This one artefact enables four things v3 handled separately or not at all:** restart after an interrupted campaign, incremental refinement, active learning, and direct feeding of the discrete-variable-representation solver.

### 8C.2 The working code

```python
#!/usr/bin/env python3
"""pes_h5.py -- HDF5 interchange layer for a van der Waals PES campaign.

Layout
------
/meta                              attrs: schema_name, schema_version, created_utc,
                                          complex, n_atoms, symbols(JSON)
/methods/<method_id>               attrs (QCSchema names): method, basis, aux_basis,
                                          program, program_version, keywords(JSON),
                                          driver, frozen_core, counterpoise
/points/<method_id>/coordinates    (Npts, N, 3) float64  Angstrom      [resizable, gzip+shuffle]
/points/<method_id>/energy         (Npts,)      float64  Hartree       [+ fletcher32]
/points/<method_id>/gradient       (Npts, N, 3) float64  Hartree/Bohr  [optional]
/points/<method_id>/converged      (Npts,)      bool
/points/<method_id>/wall_s         (Npts,)      float64
/points/<method_id>/provenance     (Npts,)      vlen str  JSON: creator/version/routine/host
/points/<method_id>/point_id       (Npts,)      vlen str  stable key -> grid coords / conformer
/grids/<grid_id>                   axis datasets + attrs describing the mesh
/hessians/<label>                  (3N, 3N) float64 + attrs (level, geometry_ref)

Why these choices
-----------------
* chunked + resizable: chunking is what makes datasets resizable and compressible
  at all, and the campaign grows point by point.
* chunk ~512 points: 512*10*3*8 B = 120 KiB, inside the documented 10 KiB - 1 MiB range.
* gzip level 4 (the documented default) + shuffle (better ratio, no significant penalty).
* fletcher32 on energies: corrupted chunks fail loudly instead of poisoning a fit.
* NO scaleoffset anywhere: it is lossy and does not preserve NaN/inf.
"""
from __future__ import annotations
import json, os, platform, socket, time
from typing import Iterable, Sequence
import numpy as np, h5py

VLEN = h5py.string_dtype(encoding="utf-8")
CHUNK_PTS = 512


class PESStore:
    def __init__(self, path: str, complex_name: str = "", symbols: Sequence[str] = ()):
        self.path = path
        new = not os.path.exists(path)
        with h5py.File(path, "a") as f:
            m = f.require_group("meta")
            if new:
                m.attrs["schema_name"] = "vdw_pes_campaign"
                m.attrs["schema_version"] = 1
                m.attrs["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if complex_name:
                m.attrs["complex"] = complex_name
            if symbols:
                m.attrs["symbols"] = json.dumps(list(symbols))
                m.attrs["n_atoms"] = len(symbols)
            self.n = int(m.attrs.get("n_atoms", 0))

    # ------------------------------------------------------------------ methods
    def register_method(self, method_id: str, **attrs) -> None:
        """attrs follow QCSchema naming: method, basis, driver, keywords, program..."""
        with h5py.File(self.path, "a") as f:
            g = f.require_group(f"methods/{method_id}")
            for k, v in attrs.items():
                g.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
            g.attrs.setdefault("registered_utc",
                               time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # ------------------------------------------------------------------ datasets
    def _ds(self, f, mid: str, name: str, shape_tail: tuple, dtype, checksum=False):
        grp = f.require_group(f"points/{mid}")
        if name in grp:
            return grp[name]
        kw = dict(shape=(0,) + shape_tail, maxshape=(None,) + shape_tail,
                  dtype=dtype, chunks=(CHUNK_PTS,) + shape_tail)
        if dtype != VLEN:                       # filters on numeric data only
            kw.update(compression="gzip", compression_opts=4, shuffle=True)
            if checksum:
                kw["fletcher32"] = True
        return grp.create_dataset(name, **kw)

    @staticmethod
    def _append(ds, block: np.ndarray):
        i = ds.shape[0]
        ds.resize(i + len(block), axis=0)
        ds[i:] = block
        return i

    # ------------------------------------------------------------------ writing
    def add_points(self, method_id: str, coords, energies, *, point_ids=None,
                   gradients=None, converged=None, wall_s=None,
                   creator="ORCA", version="6.1", routine="sp") -> int:
        coords = np.asarray(coords, "f8")
        if coords.ndim == 2:
            coords = coords[None]
        npts, natm = coords.shape[0], coords.shape[1]
        energies = np.atleast_1d(np.asarray(energies, "f8"))
        prov = json.dumps(dict(creator=creator, version=version, routine=routine,
                               host=socket.gethostname(), platform=platform.platform(),
                               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        with h5py.File(self.path, "a") as f:
            i0 = self._append(self._ds(f, method_id, "coordinates", (natm, 3), "f8"), coords)
            self._append(self._ds(f, method_id, "energy", (), "f8", checksum=True), energies)
            self._append(self._ds(f, method_id, "converged", (), np.bool_),
                         np.ones(npts, bool) if converged is None
                         else np.asarray(converged, bool))
            self._append(self._ds(f, method_id, "wall_s", (), "f8"),
                         np.zeros(npts) if wall_s is None else np.asarray(wall_s, "f8"))
            self._append(self._ds(f, method_id, "provenance", (), VLEN),
                         np.array([prov] * npts, dtype=object))
            self._append(self._ds(f, method_id, "point_id", (), VLEN),
                         np.array(point_ids if point_ids is not None
                                  else [f"{method_id}:{i0+k}" for k in range(npts)],
                                  dtype=object))
            if gradients is not None:
                g = np.asarray(gradients, "f8")
                self._append(self._ds(f, method_id, "gradient", (natm, 3), "f8"),
                             g[None] if g.ndim == 2 else g)
        return i0

    def add_hessian(self, label: str, H, *, level: str, geometry_ref: str):
        with h5py.File(self.path, "a") as f:
            g = f.require_group("hessians")
            if label in g:
                del g[label]
            d = g.create_dataset(label, data=np.asarray(H, "f8"),
                                 compression="gzip", compression_opts=4, shuffle=True)
            d.attrs["level"] = level
            d.attrs["geometry_ref"] = geometry_ref

    def register_grid(self, grid_id: str, axes: dict):
        with h5py.File(self.path, "a") as f:
            g = f.require_group(f"grids/{grid_id}")
            for name, vals in axes.items():
                if name in g:
                    del g[name]
                g.create_dataset(name, data=np.asarray(vals, "f8"))
            g.attrs["axis_order"] = json.dumps(list(axes))
            g.attrs["shape"] = [len(v) for v in axes.values()]

    # ------------------------------------------------------------------ reading
    def todo(self, method_id: str, wanted_ids: Iterable[str]) -> list:
        """Incremental refinement / restart: which requested points are not yet done?"""
        with h5py.File(self.path, "a") as f:
            p = f.get(f"points/{method_id}")
            if p is None or "point_id" not in p:
                return list(wanted_ids)
            have = {s.decode() if isinstance(s, bytes) else s
                    for s, ok in zip(p["point_id"][:], p["converged"][:]) if ok}
        return [i for i in wanted_ids if i not in have]

    def dataset(self, method_id: str, converged_only=True):
        with h5py.File(self.path, "r") as f:
            p = f[f"points/{method_id}"]
            m = p["converged"][:] if converged_only else slice(None)
            return p["coordinates"][:][m], p["energy"][:][m]

    def delta_pairs(self, low: str, high: str):
        """Aligned (geometry, E_high - E_low) pairs for Delta-learning."""
        with h5py.File(self.path, "r") as f:
            def idx(mid):
                p = f[f"points/{mid}"]
                ids = [s.decode() if isinstance(s, bytes) else s for s in p["point_id"][:]]
                return {k: j for j, (k, ok) in enumerate(zip(ids, p["converged"][:])) if ok}
            il, ih = idx(low), idx(high)
            keys = sorted(set(il) & set(ih))
            X = f[f"points/{high}/coordinates"][:][[ih[k] for k in keys]]
            dE = (f[f"points/{high}/energy"][:][[ih[k] for k in keys]]
                  - f[f"points/{low}/energy"][:][[il[k] for k in keys]])
        return keys, X, dE

    def dvr_grid(self, method_id: str, grid_id: str):
        """Energies reshaped onto a registered product grid, for the DVR solver."""
        with h5py.File(self.path, "r") as f:
            shape = tuple(f[f"grids/{grid_id}"].attrs["shape"])
            p = f[f"points/{method_id}"]
            ids = [s.decode() if isinstance(s, bytes) else s for s in p["point_id"][:]]
            V = np.full(int(np.prod(shape)), np.nan)
            for j, k in enumerate(ids):
                if k.startswith(f"{grid_id}:") and p["converged"][j]:
                    V[int(k.split(":")[1])] = p["energy"][j]
        return V.reshape(shape)      # NaNs mark holes -> feed straight back to todo()


# --------------------------------------------------------------------------
if __name__ == "__main__":
    S = PESStore("campaign.h5", complex_name="Ar-HCl", symbols=["Ar", "H", "Cl"])
    S.register_method("dlpno_avtz",
                      method="DLPNO-CCSD(T1)", basis="cc-pVDZ-F12 (paired with CABS)",
                      aux_basis="cc-pVDZ-F12 (paired with CABS)/C", program="ORCA", program_version="6.1",
                      driver="energy", frozen_core=True, counterpoise="none",
                      keywords={"TCutPNO": 1e-7, "PNO": "TightPNO", "SCF": "TightSCF"})
    R = np.linspace(2.8, 8.0, 40); TH = np.linspace(0, np.pi, 24)
    S.register_grid("g2d", {"R": R, "theta": TH})
    ids = [f"g2d:{i*len(TH)+j}" for i in range(len(R)) for j in range(len(TH))]
    print("points still to compute:", len(S.todo("dlpno_avtz", ids)))
```

### 8C.3 What falls out of the schema

**Restart and idempotence.** `todo()` makes the campaign idempotent — re-running the driver after a killed job computes only the missing points. That is precisely how a 6 h cap is survived on an embarrassingly parallel grid, and it is why the PES rows are `D` (decomposable) rather than `S` (serial bottleneck) in the Concurrency column.

**Δ-learning.** `delta_pairs(low, high)` produces the aligned (geometry, ΔE) training set the Δ-learning row needs, with the low- and high-level points matched by stable `point_id` rather than by array position.

**Direct DVR feeding.** `dvr_grid()` returns the potential on the registered mesh with `NaN` holes that route straight back into `todo()`. No intermediate text format, no re-parsing, no silent ordering bug.

**Counterpoise bookkeeping.** `counterpoise` is a *method attribute*, not a comment, so a later fit cannot silently mix corrected and uncorrected points (rule D4).

**Provenance that satisfies §20.2.** The `provenance` dataset carries creator, version, routine, host, platform and UTC per point, in QCSchema field names. **The §20.2 record lives here**, not in a separate log.

**One caution.** HDF5 has no multi-writer story without the parallel build. With `parallel -j 16` writing concurrently, have each worker write its own shard (`campaign_<rank>.h5`) and add a merge step, or serialise writes through a single collector process. Do not let sixteen processes open one file for append.

---

## 8D. Analytical Hessian CC Mandate: 3-Tier Routing Protocol

Chapter 9 of the Method Matrix (Codes and Acquisition) is updated to reflect the new Graceful Fallback Protocol for coupled-cluster VPT2 force fields.  
**Routing Logic Update:**

> 1. **Tier 1 (Optimal):** CFOUR detected $\rightarrow$ Route to Analytic CCSD(T) Hessian.  
> 2. **Tier 2 (ORCA):** ORCA detected $\rightarrow$ Route to Analytic CCSD(T) Hessian.  
> 3. **Tier 3 (Open-Source Exact Physics):** MPQC or Psi4 detected $\rightarrow$ Halt and prompt the user with the following UI dialogue:*"Analytical CCSD(T) Hessians are unavailable in this engine. Running Numerical Hessians will require $\approx 176,000$ single points. **Options:** \[A\] Execute Numerical CCSD(T) (Warning: High Compute Time), or \[B\] Calculate harmonic at CCSD(T) but extract anharmonic VPT2 corrections at MP2/DFT (Substituted Hybrid Force Field \- Recommended)."*



## 9. Codes and acquisition: the MPQC track and Legacy/Proprietary Alternates (ORCA & CFOUR)


### 9.0 The Tier 1 Standard: Valeev Stack (MPQC)
The **Valeev Stack (MPQC [CCSD(T)-F12 / TiledArray])** is the primary open-source standard for vdW interaction energies. It completely replaces canonical MPQC CCSD(T)-F12 basis set extrapolation. MPQC's explicitly correlated CCSD(T)-F12 natively hits Complete Basis Set (CBS) limits with `cc-pVTZ-F12` and eliminates Time Heuristic Drift.

**Escalator Restriction:** MPQC must *only* receive geometries that have been pre-optimized by the PySCF DFT escalator. It is strictly a Single-Point (SP) energy engine and cannot perform analytic VPT2 optimizations.

### 9.1 Legacy/Proprietary Alternates: ORCA and CFOUR
Due to severe licensing friction (EULAs and wet-signature requirements) which disrupt automated CI/CD and student environments, ORCA and CFOUR have been demoted to legacy/proprietary alternates. Use MPQC for energetics.

### 9.1 The multi-code decision, restated

v3 presented a single implied driver. That presentation hides capabilities only CFOUR has and implies capabilities ORCA does not have. **v4 runs two tracks.**

The decisive fact: **CFOUR has analytic second derivatives for closed-shell CCSD(T); ORCA does not have analytic Hessians for any correlated method** — "At present, analytical Hessians can be calculated for SCF only" ([ORCA frequencies](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/frequencies.html); [CFOUR analytic energy derivatives](https://cfour.uni-mainz.de/cfour/index.php?n=Main.AnalyticEnergyDerivatives)). Every claim in this document about a CCSD(T)-quality anharmonic force field is a **CFOUR** claim.

**The arithmetic settles the division of labour.** For a non-linear N = 10 complex, 3N − 6 = 24 normal modes, VPT2 needs 1 + 2 × 24 = **49 Hessians** = 6N − 11. Then:

| Track | Route | Jobs at N = 10 | Warrant |
|---|---|---|---|
| A — CFOUR | 49 **analytic CCSD(T)** Hessians (`VIB=EXACT`, `ANHARM=VPT2`) | **49** | closed-shell RHF and UHF CCSD(T) analytic second derivatives exist |
| B — ORCA, DFT | 49 analytic DFT Hessians | **49** | the only route `!VPT2` accepts unmodified — cheap, and DFT-quality |
| C — ORCA, canonical CCSD(T) | central differences of AUTOCI analytic gradients: 49 × 60 (54 with `NumHessTransInvar`) | **2,940** | but `!VPT2` refuses the job: "only methods for which analytical Hessians are available are supported" — this route needs an external driver |
| D — ORCA, DLPNO | numerical gradient 6N = 60 single points; numerical Hessian 60 × 60 = 3,600; total 49 × 3,600 | **176,400** | "Analytic gradients are not available" for DLPNO ([MDCI](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/mdci.html)) |

**Track D / Track A = (6N)² = 36N² = 3,600× at N = 10**, independent of per-job cost `[D]`. For DLPNO to win it would have to be faster than canonical CCSD(T) by a factor exceeding 3,600/k, where k is the cost ratio of an analytic CCSD(T) Hessian to a CCSD(T) energy. **k was not verified from any source — `n.a.`** — but even generously at k ≈ 20, DLPNO would need to be ~180× faster per point, and at 5–10 atoms its linear-scaling advantage has barely switched on. **There is no crossover in this size range. CFOUR owns every CCSD(T)-quality anharmonic tier; ORCA owns search, energetics, F12/DLPNO refinement and DFT-quality VPT2.**

A second, non-arithmetic reason to prefer analytic Hessians: the CFOUR developers "encourage you to use analytic second derivatives whenever possible, especially when one is planning to calculate an anharmonic force field", because cubic and quartic constants are obtained by numerical differentiation and inherit the accuracy of what is differentiated. ORCA says the same in the negative: VPT2 "is very sensitive to numerical noise due to convergence, approximations and other settings". A numerical-gradient-of-numerical-gradient DLPNO force field is three differentiations away from the energy, on a surface that is only "virtually but not perfectly smooth (like any method that involves cut-offs)". **Cost aside, it is the wrong tool.**

### 9.2 Acquisition and practical friction, side by side

Both are free for academic use, and that phrase hides two very different friction profiles.

| Dimension | ORCA | CFOUR |
|---|---|---|
| Distributor | Neese group; academic distribution via the ORCA Forum of the MPI für Kohlenforschung, commercial licences via FACCTs ([FACCTs](https://www.faccts.de/orca/)) | University of Mainz; licence signed with Prof. Jürgen Gauss, Duesbergweg 10-14, D-55128 Mainz ([CFOUR download](https://cfour.uni-mainz.de/cfour/index.php?n=Main.Download)) |
| **How you get it** | create a Forum account, log in, accept the EULA, download ([Sigma2 install guide](https://documentation.sigma2.no/software/userinstallsw/ORCA.html); [ETH Zürich HPC docs](https://docs.hpc.ethz.ch/software/chemistry/orca/)) | **print, sign, and post or fax** the licence form; a password is then supplied for a source download. Distribution is via a GitLab server hosted at the University of Florida |
| Turnaround | minutes | **weeks**, ending in a *compile*, not an install |
| Cost, academic | free — "ORCA is and will remain free for academic and personal use" | free — "For non-commercial purposes there is no charge to obtain CFOUR; one must simply sign a license agreement" |
| Cost, commercial | paid via FACCTs; price not published → **`n.a.`** | separate agreement; price not published → **`n.a.`** |
| Binaries vs source | prebuilt binaries, e.g. `orca_6_0_0_linux_x86-64_avx2_shared_openmpi416.tar.xz` ([CSC ORCA docs](https://docs.csc.fi/apps/orca/)) | **source only**; needs Fortran and C/C++ compilers plus BLAS; `GENBAS` must be present ([CFOUR installation](https://cfour.uni-mainz.de/cfour/index.php?n=Main.Installation)) |
| Platforms | Linux, macOS, Windows; OpenMPI or MS-MPI | not enumerated beyond compiler requirements → **`n.a.`**; site installs exist on standard Linux HPC |
| Known build hazards | the MPI version must match the binary build | documented compiler bug: "Intel 2017 compilers have a serious bug affecting CFOUR, please use Intel 2016 or 2018" |
| Version cadence | 4.0 (2017) → 6.1.1 (Dec 2025) ([Wikipedia release history](https://en.wikipedia.org/wiki/ORCA_(quantum_chemistry_program))) | first production release 06/2010; **v2.1, July 2019 is the most recent announced** ([CFOUR home](https://cfour.uni-mainz.de/cfour/index.php?n=Main.HomePage)) |
| **Public vs developers' version** | not applicable — releases are the product | **the public release lags.** GUINEA (deperturbed VPT2, anharmonic intensities), `ANHARM=FULLQUARTIC`, Raman intensities, CASSCF, MRCC-driven calculations and IRC are "not part of the public release" ([features](https://cfour.uni-mainz.de/cfour/index.php?n=Main.Features)) |
| Redistribution / shared images | the EULA is non-transferable and non-sublicensable; the licensee may not make the software publicly accessible ([ORCA EULA](https://hpc.hku.hk/wp-content/uploads/document/orca-eula_2021.pdf)) | worse: source and object files "must be protected from users"; no redistribution, no sublicensing, no publication of source or object code, plus a confidentiality clause ([CFOUR licence](https://cfour.uni-mainz.de/cfour/index.php?n=Main.MainLicense)) |
| Support | forum-based community support | **"The authors will not provide a help line for usage problems"** |
| Licence term | not stated in the fetched EULA → **`n.a.`** | **two-year term, renewing yearly**; German law; mandatory citation string |
| **Teaching tier (50 students)** | deployable — each student registers, or the institution installs centrally | **not deployable.** Wet-signature licensing, a source build, per-user access restriction and a no-help-line clause do not scale to a class |

**Sequencing consequence: start the CFOUR paperwork on day one of the project.** Mail-and-sign turnaround plus a source build is the long pole, not the science.

**Container consequence, for both.** A shared Docker or Singularity image handed to non-licensed collaborators is inconsistent with ORCA's no-public-accessibility clause and worse for CFOUR, whose object code the licensee must protect. Site installs use restricted per-user access lists. **Shared cloud images fail both licences.**

### 9.3 Capability matrix, verified from official documentation

| Capability | CFOUR (public release) | ORCA 6.1 |
|---|---|---|
| Analytic gradients, coupled cluster | CCSD, CCSD(T) for RHF/UHF; ROHF CCSD(T) gradient-level only | **yes — via AUTOCI for RHF/UHF CCSD, CCSD[T], CCSD(T), CCD, CISD, CEPA(0), MP2–MP4** ([AUTOCI](https://orca-manual.mpi-muelheim.mpg.de/_sources/contents/modelchemistries/autoci.md)). *v3 implied ORCA cannot optimise at CCSD(T). It can; it is DLPNO that cannot* |
| **Analytic Hessians** | HF, TCSCF, MP2–MP4, CCD, CCSD, **CCSD(T) closed-shell RHF and UHF**, CCSDT-n, CC3, CCSDT, general CC | **SCF only** (HF, DFT), and **not** for double hybrids or RI-JK |
| Anharmonic force field / VPT2 | `ANHARM=VPT2` (full cubic + semidiagonal quartic), `ANHARM=VIBROT`; `ANHARM=FULLQUARTIC` **not public**; cubic/quartic constants by numerical differentiation of analytic second derivatives, default step 0.05 | `!VPT2` / `%vpt2`; "Currently, only methods for which analytical Hessians are available are supported"; **"Linear molecules are not supported yet"** ([ORCA VPT2](https://orca-manual.mpi-muelheim.mpg.de/contents/spectroscopyproperties/vpt2.html)) |
| α constants, B₀ from B_e | yes — `ANHARM=VIBROT` computes exactly the cubic constants needed; output prints "Be, B0 AND B-B0 SHIFTS FOR SINGLY EXCITED VIBRATIONAL STATES (CM-1)" | yes at HF/DFT: VPT2 gives "rotational and vibrational-rotational constants" |
| Quartic centrifugal distortion | yes, via VPT2 | **yes — "centrifugal distortion constants" and "Watson's symmetrically and asymmetrically reduced Hamiltonian parameters"**. *v3 under-credited this* |
| **Sextic centrifugal distortion** | **yes** — implementation described by Puzzarini and co-workers; oxirane agreement ~0.1 % rotational, 2–3 % quartic, 3–4 % sextic ([oxirane study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4630858/)) | **not documented** among the VPT2 outputs → **`n.a.`** — treat as unavailable |
| Nuclear quadrupole coupling | electric field gradients from `PROPS=FIRST_ORDER`, with the documented conversion χ(kHz) = EFG(a.u.) × Q(mbarn) × 234.96474 | EFGs available; for closed-shell DLPNO-CCSD an unrelaxed density permits analytic first-order properties, but "an unrelaxed density for CCSD(T) is **NOT** available" |
| Dipole and higher multipoles | dipole, quadrupole, octopole from `PROPS=FIRST_ORDER` | from unrelaxed densities for closed-shell DLPNO-CCSD; not for CCSD(T) |
| Nuclear spin–rotation | `SPINROT=ON` | not documented → **`n.a.`** |
| Diagonal Born–Oppenheimer correction | `DBOC=ON` at HF, MP1, MP2, CCSD for RHF/UHF | not documented → **`n.a.`** |
| Relativistic corrections | `RELATIVISTIC=` MVD1, MVD2, DPT2, SF-DPT4, DPT4, SF-DPT6, DPT2-1E, SFREE, X2C1E, X2CMF | a `REL` module exists among the parallelised modules; detailed list not fetched → **`n.a.`** |
| F12 | not on the public features page → **`n.a.`** | extensive: `CCSD(T)-F12`, `-F12/RI`, `-F12D/RI`, `MPQC CCSD(T)-F12-F12`, `DLPNO-CCSD(T1)-F12(/D)`; F12 costs "usually 10-30 %" of the DLPNO correlation step. **No F12 *gradient*** |
| DLPNO / local correlation | not available → **`n.a.`** | yes, but **"Analytic gradients are not available"**; numerical gradients "have been attempted and reported to have been successful" |
| Multireference | CASSCF **not public**; MRCC-driven likewise, though an interface exists | CASSCF / NEVPT2 / CASPT2, MRCI, FIC-MRCI/-NEVPT2/-NEVPT3/-MRCC via AUTOCI |
| DFT | **none listed** on the public features page | broad, including double hybrids |
| Excited states | EOM-CC for excited, ionised and attached states; EOM-CCSD partially parallelised | CIS/TDDFT, ROCIS, MCRPA, EOM through MDCI |
| Isotopologues | `%masses` / `%isotopes` sections and an `ISOTOPES` file | `!Mass2016` selects most-abundant-isotope masses; `orca_vib` re-analyses a stored `.hess` |
| Fit of r_e structures to rotational constants | listed as a feature | not documented → **`n.a.`** |
| **Pickett/SPCAT export** | not documented → **`n.a.`** | **yes**: `%output Pickettname "pickett.txt" end` produces templates for SPCAT `.var` input |
| GPU | no GPU support documented on any fetched page → **`n.a.`** ("not documented", not "no") | **none** — ORCA "does NOT support GPU acceleration" ([PERUN HPC](https://wiki.perun.tuke.sk/env/orca/)) |

**Two corrections to v3 that unfairly disadvantaged ORCA, stated explicitly.** (1) ORCA **does** have canonical CCSD(T) analytic gradients via AUTOCI, so any claim that ORCA cannot optimise at CCSD(T) is wrong. (2) ORCA's VPT2 **does** output centrifugal distortion, Watson parameters, α constants and a Pickett/SPCAT export — better than v3 credited. Sextic distortion remains CFOUR-only.

### 9.4 Parallelism and memory: the two models are structurally different

**ORCA.** `! PAL4` … `PAL64` or `%pal nprocs N end`; never launched with `mpirun`; always called with its full path. Guidance: "for RI-DFT perhaps up to 16 processors are a good idea while for hybrid DFT and Hartree-Fock a few more processors are appropriate"; "Coupled-cluster calculations usually scale well up to at least 8 processors but probably it is also worthwhile to try 16" ([ORCA parallel manual](https://orca-manual.mpi-muelheim.mpg.de/contents/essentialelements/parallel.html)). **The decisive ORCA feature for this document is multi-process mode over displacements** — numerical gradients and frequencies, overtones, VPT2, NEB and GOAT — with two-level grouping (`!PAL32(8x4)` = 8 groups of 4) and the recommendation "For Numerical Frequencies or Gradient runs it makes sense to choose nprocs = 4 or 8 times 6*Number of Atoms." Beware the OpenMPI binding trap: since v1.8 "mpirun automatically binds processes… For NumCalc this can result in all displacements being run on the same set of cores, leading to severe performance degradation" — pass `"--bind-to none"`.

**CFOUR.** Launch with `xcfour`, never `mpirun xcfour`. **"CFOUR parallelises poorly" is too crude and v3 should not have said it.** The precise version: parallelism covers HF-SCF, CCD and CCSD energies and first and second derivatives for RHF/UHF/ROHF, and CCSD(T) energies and derivatives for RHF/UHF, **only with `ABCDTYPE=AOBASIS` together with `CC_PROG=ECC`**; EOM-CCSD is partially parallel; and "Because of the structure of CFOUR it does not make sense to run MP2 in parallel up to now" ([running CFOUR in parallel](https://cfour.uni-mainz.de/cfour/index.php?n=Main.RunningCfourInParallel)). The overview paper reports OpenMP parallel efficiency of "∼50 % for eight or more cores" for `xncc`, and states the hope to add "scalable distributed-parallel implementations in the next release" ([CFOUR overview paper](https://par.nsf.gov/servlets/purl/10177577)). One third-party benchmark reports 920.98 → 465.49 → 241.43 → 138.08 s on 1/2/4/8 CPUs, ~83 % efficiency at 8 cores ([trends-in-science](http://trends-in-science.blogspot.com/2010/04/cfour-in-parallel.html)) — a single job, but it directly contradicts a blanket dismissal.

**Verdict:** *CFOUR scales adequately within one node for the coupled-cluster steps that matter, and does not offer ORCA's displacement-level parallelism.* With MPI, CFOUR's memory and disk requirements scale roughly with the number of ranks ([sobereva Note 150](http://sobereva.com/150)).

**The memory models are inverted, and getting this backwards is the commonest configuration error when moving between the two codes:**

| | ORCA | CFOUR |
|---|---|---|
| Keyword | `%maxcore` in MB | `MEMORY_SIZE` × `MEM_UNIT` |
| Scope | **per process** | **one global allocation** |
| Default | none — set it | ≈**762 MB** (100,000,000 integer words) |
| On a 64 GB / 8-core box | `%maxcore 3000` (or 3400 at 7 ranks) | **must be raised by hand**: `MEMORY_SIZE=32, MEM_UNIT=GB`, or the job thrashes |

### 9.5 The CFOUR ZMAT worked example

An Ar···HCN-type 5-atom complex, chosen so the example is honest about linearity handling. **Every angle must avoid 0° and 180°, so a quasi-linear chain requires dummy atoms** ([molecular geometry input](https://cfour.uni-mainz.de/cfour/index.php?n=Main.MolecularGeometryInput)).

```
Ar-HCN anharmonic force field, CCSD(T)/ANO1
Ar
X  1 RX
H  1 RH  2 AH
C  3 RC  1 AC  2 DC
N  4 RN  3 AN  1 DN

RX = 1.000000
RH = 4.400000
AH = 90.000000
RC = 1.065000
AC = 88.000000
DC = 180.000000
RN = 1.156000
AN = 179.000000
DN = 0.000000

*CFOUR(CALC=CCSD(T)
BASIS=ANO1
REFERENCE=RHF
FROZEN_CORE=ON
ABCDTYPE=AOBASIS
CC_PROG=ECC
SPHERICAL=ON
UNITS=ANGSTROM
VIB=EXACT
ANHARM=VPT2
ANH_STEPSIZ=50000
FD_PROJECT=ON
PROPS=FIRST_ORDER
MEMORY_SIZE=32
MEM_UNIT=GB
SCF_CONV=10
CC_CONV=10
LINEQ_CONV=10
GEO_CONV=5)

%isotopes
1
2
1
12
14
```

| Keyword | Why it is there |
|---|---|
| `CALC=CCSD(T)` | selects CCSD(T); CFOUR's `CALCLEVEL` table maps CCSD = 10, CCSD(T) = 22, CC3 = 32 |
| `BASIS=ANO1` | ANO0/1/2, cc-pVXZ, aug-pVXZ, cc-pCVXZ and `SPECIAL` are available; read from `GENBAS` |
| `REFERENCE=RHF` | closed-shell RHF is the reference for which CCSD(T) **analytic second derivatives** exist |
| `ABCDTYPE=AOBASIS`, `CC_PROG=ECC` | strongly recommended for large CCSD/CCSD(T) frequency work **and required for parallel execution** |
| `VIB=EXACT` | analytic second derivatives. `VIB=ANALYTIC` is "in the current release not available"; the manual also warns "Please do not use VIB=2!" |
| `ANHARM=VPT2` | full cubic + semidiagonal quartic field; drives resonance analysis and is **required if isotopologue constants are wanted** — `ANHARM=VIBROT` restricts cubic constants to φ_nij with n totally symmetric and is **not sufficient for isotopologues of lower symmetry** |
| `ANH_STEPSIZ=50000` | the default, 0.05 in reduced normal coordinates |
| `FD_PROJECT=ON` | correct **at a stationary point**; must be `OFF` at non-stationary geometries and in the parallel VPT2 recipe |
| `PROPS=FIRST_ORDER` | dipole, quadrupole and octopole moments and electric field gradients for χ, via EFG × Q(mbarn) × 234.96474 → kHz |
| `MEMORY_SIZE=32`, `MEM_UNIT=GB` | overrides the ≈762 MB default — omit this and the job thrashes |
| `FROZEN_CORE=ON` | stated for the record; note §4.8 — frozen core is a −0.81 % bias in B_e at quadruple zeta and needs a core correction if ≤0.5 % is claimed |
| `%isotopes` | per-atom masses for isotopologue force fields |

**Notes.** The geometry above must already be a converged CCSD(T)/ANO1 stationary point — run a prior job with `*` appended to the variable list to optimise. Files `FCM`, `FCMINT` and `DIPDER` are produced by the harmonic step. For a queue-split run add `FREQ_ALGORITHM=PARALLEL`, `ANH_ALGORITHM=PARALLEL`, `FD_PROJECT=OFF` and process with `xjoda`, `xsymcor`, `xja2fja`, then `xcubic`.

**ZMAT hazards that break pipelines** ([geometry input](https://cfour.uni-mainz.de/cfour/index.php?n=Main.MolecularGeometryInput); [Cartesian coordinates](https://cfour.uni-mainz.de/cfour/index.php?n=Main.UseOfCartesianCoordinates)):

- The input file **must be named `ZMAT`**, and `GENBAS` must be in the working directory.
- Structure: title line, geometry, **blank line**, variable definitions, then `*CFOUR(...)`, then optional `%`-sections.
- **Variable names are limited to three characters**; fields are separated by exactly one space.
- **Angles of 0° or 180° are forbidden** — linear fragments, a common vdW motif (OC···HF, HCN dimers), need dummy atoms `X`. Ghost atoms `GH` provide counterpoise.
- **Cartesian input is a trap**: with `COORDINATES=CARTESIAN` "only single-point calculations can be performed". A converter that dumps XYZ from ORCA into a CFOUR job **silently produces a code that cannot optimise or run a force field**. Psi4 ships a CFOUR interface that writes `ZMAT` for you (methods prefixed `c4-`), and it writes Cartesian input — convenient for single points, useless for force fields. **For force fields, hand-write the Z-matrix.**

**The matching ORCA input, for comparison:**

```
! B3LYP D4 def2-TZVPP TightSCF DEFGRID3 VPT2
%pal nprocs 16 nprocs_group 2 end
%maxcore 3000
%method Z_Tol 1e-14 end
%vpt2
  VPT2 On
  AnharmDisp 0.05
  HessianCutoff 1e-12
end
%output Pickettname "arhcn_pickett.txt" end
* xyz 0 1
  <tightly optimised coordinates>
*
```

**What this ORCA job cannot produce, and why.** (1) A CCSD(T) anharmonic force field — `!VPT2` supports only analytic-Hessian methods. (2) Anything for a **linear** complex — "Linear molecules are not supported yet", a serious limitation for OC···HX and HCN-containing dimers. (3) Sextic centrifugal distortion. (4) A double-hybrid or DLPNO force field. (5) CCSD(T)-quality electric field gradients — "an unrelaxed density for CCSD(T) is NOT available". (6) Nuclear spin–rotation constants and the diagonal Born–Oppenheimer correction.

### 9.6 The recommended hybrid pipeline

1. **ORCA / GFN2-xTB GOAT** — conformer ensemble, filtered on RMSD, ΔE and rotational constants (§9B).
2. **CREST `--nci --nocross --noreftopo`** — an independent second search; carry the union.
3. **ORCA / r²SCAN-3c then ωB97X-D4 or ωB97M-V** — optimise the survivors; an analytic Hessian confirms minima.
4. **ORCA / DLPNO-CCSD(T1)-F12D/RI** — relative and binding energies, Boltzmann weights; F12 adds 10–30 % to the DLPNO correlation time and introduces no new truncation parameters.
5. **ORCA / VPT2 at DFT** — a fast first-pass α set, quartic distortion and a Pickett file. *Skip for linear complexes — unsupported.*
6. **CFOUR / CCSD(T) optimisation** of the single best conformer (internal or XYZ2INT coordinates only; `GEO_CONV=5`, RMS gradient 1×10⁻⁵ E_h/bohr, `GEO_MAXCYC=50`).
7. **CFOUR / `VIB=EXACT` + `ANHARM=VPT2`** — the production anharmonic force field, α constants, quartic **and sextic** distortion, and isotopologue force fields from the same field via `%isotopes`.
8. **CFOUR / `PROPS=FIRST_ORDER` + `SPINROT=ON`** (+ `DBOC`, `RELATIVISTIC`) — χ tensors, dipole components, spin–rotation, small corrections.
9. **Hand-off to SPFIT/SPCAT** — ORCA writes the Pickett template; **CFOUR constants must be transcribed by hand or by a script (no documented export → `n.a.`).**

### 9.7 Other codes, briefly

| Code | Licence and cost | Spectroscopic-constant capability |
|---|---|---|
| **Molpro** | commercial, 1–4 year terms. Academic first-year fees: group €1,700, site €5,100, service €6,800, node16 €550, node32 €850; renewals €1,500 / €4,500 / €6,000 / €500 / €750; VAT extra ([Molpro product catalogue](https://www.molpro.net/info/products.php)) | **the only route to an F12 gradient**, hence to a junChS-F12 *geometry*. VPT2 with centrifugal distortion is discussed by the developers |
| **Psi4** | free, LGPL-3, no registration ([psicode.org](https://psicode.org/)) | **the only route to a focal-point gradient driver**; its route to CFOUR-style constants is to drive CFOUR via the `c4-` interface |
| **PySCF / gpu4pyscf** | free, Apache-2.0 ([pyscf.org](https://pyscf.org/about.html)) | no VPT2 or anharmonic force fields → **no**. But the production double-precision GPU DFT engine of §8.2 |
| **Gaussian** | commercial; the pricing page publishes **no amounts** → **`n.a.`** ([Gaussian pricing](https://gaussian.com/pricing/)) | yes in practice — `Freq=VibRot`, `Freq=Anharm`, `Output=Pickett` |
| **TURBOMOLE** | commercial; an **Educational licence is free of charge for students and teaching** ([TURBOMOLE Educational](https://store.turbomole.org/product/turbomole-8-0-educational/)). Academic research price not rendered → **`n.a.`** | not established from fetched sources → **`n.a.`** |
| **MRCC** | free for academic use; institutional e-mail required, then a printed, signed agreement returned by e-mail ([MRCC registration](https://www.mrcc.hu/index.php/getting-started/registration)) | high-order CC through CFOUR's `CC_PROG=MRCC` — but MRCC-driven runs are **not in CFOUR's public release** |
| **autoPES / flex-autoPES** | licence **not stated in the manual → `n.a.`** | SAPT(DFT)-based automatic intermolecular surfaces; **requires ORCA 3.0.1, Dalton 2.0 and SAPT2016**, which is a real deployment obstacle ([autoPES manual](https://www.physics.udel.edu/~szalewic/SAPT/autoPES_manual.pdf)) |

**Teaching-tier verdict.** Only ORCA, Psi4, PySCF and the free TURBOMOLE Educational licence are administratively viable at 50 seats, and **only ORCA produces VPT2 spectroscopic constants natively** (at HF/DFT level, with a Pickett export). Psi4 and PySCF cannot substitute for CFOUR in a rotational-spectroscopy course without CFOUR underneath. **Do not list CFOUR in a teaching stack.**

**Two stack consequences of adopting the composite recipes of §9A:** a focal-point *gradient* requires Psi4, and an F12 *gradient* requires Molpro. Neither is available in ORCA 6.1, and Molpro is commercial. Recipes R3 and R7 are therefore licence-gated rows.

---

## 9A. Composite and combined methods

### 9A.1 The verdict on the frozen-monomer scheme

The scheme — high-level (CCSD(T)) monomers, frozen, with a cheaper dispersion-corrected and counterpoise-corrected method relaxing only the intermolecular coordinates — **is real, established, and should be adopted as the default.** Its literature name is the **rigid- or frozen-monomer composite geometry**, and in the microwave-structure literature it is the construction underlying the semi-experimental approach to complexes: "the application of the SE approach in such cases implies **fixing the intra-molecular parameters at those of the isolated fragments** and then fitting the most significant inter-molecular parameters" ([junChS-F12](https://cris.unibo.it/retrieve/handle/11585/868585/ae4939e6-d216-426d-9d79-edb47b92c82c/junChS-F12.pdf)). Its energetic ancestor is the focal-point / additive composite, and its monomer-frozen bookkeeping is the same convention S66 uses ([Řezáč, Riley & Hobza](https://pmc.ncbi.nlm.nih.gov/articles/PMC3152974/)).

**But the propagation arithmetic changes the emphasis, and the headline must be stated correctly.** From §4.5: on CO₂···H₂O at R = 2.836 Å, ΔR = 0.002 Å costs the same in B as a **16.8 mÅ uniform error in every monomer bond**, and no method errs by that much covalently. Therefore **monomer quality does not dominate B and C — the intermolecular separation does, by about two orders of magnitude.** Where monomer quality dominates is **A**: a 10 mÅ monomer error is −1.71 % in A while ΔR contributes 0.000 %.

> **The headline: freeze good monomers to fix A, and spend the remaining budget on R to fix B and C.**
> Upgrading monomers from B3LYP-class to CCSD(T)-class buys **1.2 percentage points in A** and only **0.056 pp (2.6 MHz) in B**. Tightening R from a 0.020 Å error to a 0.005 Å error buys **1.0 pp (46 MHz) in B**. `[D]`

The scheme is therefore *structurally correct, essentially free, and not sufficient on its own*. With B3LYP-D4/CP for the intermolecular part it gives 1–2 % in B; it reaches 0.1–0.3 % only when the intermolecular optimiser is ωB97M-V/quadruple-zeta class or better (recipe R2).

**Independent support for the transferability premise.** junChS reports that **intramolecular distances agree within 4 mÅ across all levels tested, while the intermolecular distance shows an average deviation of −0.005 Å and a maximum of 0.03 Å** `[M]` ([junChS-F12](https://cris.unibo.it/retrieve/handle/11585/868585/ae4939e6-d216-426d-9d79-edb47b92c82c/junChS-F12.pdf)). The literature's own numbers say the intermolecular coordinate is where the error lives.

**Where the freeze is unsafe.** Freezing at the *isolated-monomer* geometry rather than the *in-complex* one neglects the deformation channel. That is **<1 kcal/mol for dispersion-bound pairs but up to 9.5 kcal/mol for multiply hydrogen-bonded ones** `[M]` ([deformation-energy assessment](https://pubs.aip.org/aip/jcp/article/158/24/244106/2899786/A-quantitative-assessment-of-deformation-energy-in)); geometrically it appears as donor X–H lengthening of order 5 mÅ on complexation. **Binding consequence: the frozen-monomer recipe is safe for dispersion-bound and weak-electrostatic complexes and must be flagged for strongly hydrogen-bonded ones**, where the frozen X–H is systematically 3–7 mÅ too short and A is correspondingly ~1 % too high.

**Substituting experimental monomer geometries is recommended, with three caveats.** (1) Use r_e^SE, not r₀ — r₀ carries the monomer's own vibrational averaging, which is then double-counted when ΔB_vib is added for the complex. (2) The freeze neglects complexation relaxation, above. (3) A frozen experimental monomer inside a DFT complex is a small internal inconsistency: the DFT gradient on the frozen coordinates is not zero, so the "optimised" structure is a constrained stationary point. **Report the residual gradient on the frozen coordinates; if it exceeds `TolMaxG` the constraint is doing real work and must be disclosed.**

### 9A.2 The frozen-monomer flag

Every geometry row now carries a flag with three values:

| Value | Meaning | Consequence |
|---|---|---|
| `relaxed` | all coordinates optimised at the row's level | A inherits the row's monomer error, typically ~1 % at B3LYP class |
| `frozen-iso` | monomers frozen at isolated-monomer geometries (experimental r_e^SE or high-level) | A to <0.2 %; deformation channel neglected — **flag for H-bonded systems** |
| `frozen-inc` | monomers frozen at in-complex geometries taken from a higher level | A to <0.2 %; deformation retained; the S66 convention |

The distinction changes the reported constants by about 1 % in A and was entirely invisible in v3.

### 9A.3 Focal-point analysis and additivity

**The construction** (Allen and co-workers): E(CCSD(T)/large) ≈ E(MP2/large) + [E(CCSD(T)/small) − E(MP2/small)]. The assumption is that basis-set convergence and correlation-treatment convergence are independent.

**Applied to geometries, it is implemented and it pays.** "In geometry optimizations … **linear combinations of gradients are used to form a focal-point gradient at each iteration**", and "when CCSD(T) in an (aug-)cc-pVXZ basis set is combined with CBS-extrapolated MP2, the accuracy in optimized geometries is similar to that using standard CCSD(T) with a basis set one ζ level higher … while **reducing the computational cost by an order of magnitude or more**" `[M]` ([focal-point study](https://par.nsf.gov/servlets/purl/10566327)). Harmonic-frequency MAEs against CCSD(T)/CBS(a[Q5]Z): CBS([TQ]Z;δ:DZ) 10.4 cm⁻¹; CBS(a[TQ]Z;δ:aDZ) 7.7; CBS([Q5]Z;δ:TZ) 4.8; CBS(a[Q5]Z;δ:aTZ) 3.7. Cost, H₂O VPT2 on one older CPU: DZ 8.0 min; aTZ 19.8; CBS(a[Q5]Z;δ:aTZ) **44.9 min**; full a[Q5]Z **1334.2 min** — **the focal point costs about 3 % of the brute-force calculation** `[M]`.

**Psi4 automates the focal-point driver; ORCA does not.** A focal-point gradient in this stack therefore requires Psi4 or an `ExtOpt` driver script.

**Where additivity breaks down.** (i) Small-basis δ terms: the C₂H₂ CH bend at aug-cc-pVDZ errs by **359.5 cm⁻¹**, and F₂ is a systematic outlier. (ii) Core-valence inclusion "gives no systematic improvement" in that frequency study. (iii) **Additive diffuse-function corrections fail outright** — see §9A.5. (iv) For N₂O⋯CO the diffuse and counterpoise contributions "were not additive; did not compensate each other; had almost the same value but opposite signs" ([Demaison *et al.* 2021](https://pubs.aip.org/aip/jcp/article/154/19/194302/565922/How-accurate-is-the-determination-of-equilibrium)).

### 9A.4 ChS, junChS and junChS-F12 — the rows to buy

**ChS ("Cheap" scheme, CBS+CV), parameter by parameter:**

R(ChS) = R[fc-CCSD(T)/cc-pVTZ] + ΔR[MP2/CBS] + ΔR[MP2/CV]

where ΔR[MP2/CBS] is the MP2 triple→quadruple extrapolation (n⁻³ for correlation, Hartree–Fock extrapolated separately) and ΔR[MP2/CV] = R[MP2/cc-pwCVTZ, all-electron] − R[MP2/cc-pVTZ, frozen core]. The energy version is E(ChS) = E(fc-CCSD(T)/TZ) + ΔE(MP2/CBS) + ΔE(MP2/CV) ([Puzzarini group composite review](https://pmc.ncbi.nlm.nih.gov/articles/PMC9863398/)).

**What each term buys.** Basis extrapolation is the largest: fc-CCSD(T)/VTZ has a 0.90 % MAE in B_e and CBS brings it to 0.28 %. Core-valence takes CBS from 0.28 % to 0.06 %; omitting it degrades the A15 energy MAE from 0.73 % to 1.26 %. **Diffuse functions are *not* in plain ChS**, which is why plain ChS is poor for weak complexes — junChS puts them in the basis itself.

**Accuracy in B_e**, MAE against semi-experimental references, all `[M]` ([Puzzarini & Stanton](https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04706c)):

| Level | MAE in B_e |
|---|---|
| fc-CCSD(T)/cc-pVTZ | 0.90 % |
| fc-CCSD(T)/cc-pVQZ | 0.43 % |
| fc-CCSD(T)/cc-pV5Z | 0.32 % |
| fc-CCSD(T)/cc-pV6Z | 0.30 % |
| fc-CCSD(T)/CBS | 0.28 % |
| CCSD(T)/cc-pV6Z + CV | 0.06 % |
| CCSD(T)/CBS+CV+fT+fQ | 0.04 % |
| **ChS (molecules ≤16 atoms)** | **0.13 %** |

The authors' own verdict: CCSD(T)/CBS+CV is the "most cost effective scheme … to meet the so-called '0.1 % accuracy' … for molecules up to 10 atoms", and is not affordable beyond ~30 atoms.

**Cost, measured.** Pyridine–H₂O on 64 CPUs / 120 GB: cc-pVTZ (308 basis functions) **7 h 47 min**; jun-cc-pVTZ (371) **16 h 12 min**; cc-pVDZ-F12 (paired with CABS) (483) **59 h 52 min** `[M]` ([jun-ChS](https://cris.unibo.it/retrieve/fdcbe2fd-290c-49a7-88cc-01c515d136bd/Extension%20of%20the%20%E2%80%9Ccheap%E2%80%9D%20composite%20approach.pdf)). Scaled to 8–16 cores and a 6–8 atom complex that is roughly **6–20 h wall `[E]`** for the whole composite geometry — **which places ChS at the 12 h tier, not at 1 w.**

**junChS** replaces cc-pVnZ with the calendar set jun-cc-pVnZ throughout, putting diffuse functions in the basis rather than in an increment. A15 interaction-energy MAEs `[M]`: ChS **2.92 %** (max ~9 %), aug-ChS 1.52 %, jul-ChS 0.84 %, **jun-ChS 1.20 %** (0.73 % against the best reference), maug-ChS 7.1 %. jun- is the sweet spot: near-aug accuracy at ~27 % fewer functions.

**junChS-F12** substitutes CCSD(T)-F12b and MP2-F12 for the conventional legs. A14 interaction energies `[M]` ([Puzzarini group composite review](https://pmc.ncbi.nlm.nih.gov/articles/PMC9863398/)):

| Scheme | MAX (kJ/mol) | MUE | RMSD |
|---|---|---|---|
| junChS | 0.34 | 0.11 | 0.14 |
| **junChS-F12** | **0.11** | **0.06** | **0.07** |

Geometry performance on the SE100 subset: **MUE(r) 0.0011 Å for second-row, 0.0005 Å for third-row species; MUE(angles) 0.15–0.19°** `[M]`. Relative errors in interaction energy: junChS CP 1.38 % / NCP 2.06 %; junChS-F12 CP 0.68 % / NCP 1.10 % / half-CP 0.88 %.

**The single most important cost fact in this section:** junChS-F12 costs "no more than twice the time of the underlying coupled-cluster step and … **one order of magnitude faster than the CBS+CV counterparts**", with F12 itself adding ≤20 % `[M]`. **junChS-F12 is ~10× cheaper than ChS for better accuracy — it is the buy.** The catch is that ORCA 6.1 has no F12 gradient, so a junChS-F12 *geometry* requires Molpro; ORCA can supply junChS-F12 single points only.

**Residual defects, stated.** ChS is not BSSE-free — "all ChS models … affected by small, but not entirely negligible BSSE". And every composite here produces **B_e**; ΔB_vib (0.1–0.7 % of B_e) must still be added before any B₀ claim (§3.0).

**Template scaling and linear-regression augmentation (the Bologna "Lego-brick" family).** Template model: r_e = r_e^level + Δr_e transferred from a molecule with a known semi-experimental structure. Linear-regression augmentation: r = a_XY · r^DFT + b_XY, coefficients fitted per bond type over ~100 semi-experimental values (rDSD examples: C–C 0.99816, C–H 0.99761, C–Br 0.97099 with b = 0.05037) ([Nano-LEGO](https://pmc.ncbi.nlm.nih.gov/articles/PMC10291548/)). Performance `[M]`: bond-length MAE from up to 7.5 mÅ down to **<1.5 mÅ**; CHBrF₂ rotational-constant MAPE 1.05 % → **0.13 %**; bromobenzene 0.41/0.45 % → 0.12/0.19 %; mean deviation in B_e 0.5 % → TM-SE 0.4 % → **TM-SE_LR 0.3 %** → 0.2 % with a linkage-angle correction ([Lego-brick](https://ricerca.sns.it/retrieve/fe3d5821-f41e-48ed-927f-34be686e050b/acs.jpca.1c07828.pdf)). **Critical caveat: "TM-SE on B3LYP geometries nearly doubles the relative deviations"** — the template works only on a revDSD/rDSD-class starting geometry. **And the gap: there is no template for the *intermolecular* coordinate, because there is no bond type to regress.** The family transfers to the monomer blocks of a complex and not to the thing that dominates B and C.

**Thermochemical composites — HEAT, Wn, Gn, ccCA — are the wrong tools here, and belong in a footnote.** HEAT gives ΔfH MAE 0.37 (HEAT-I) / 0.24 (HEAT-II) kJ/mol with geometries accurate to ±0.003 Å, at up to 3 days per molecule for the CCSDTQ step; W4 gives MAD 0.066 kcal/mol against Active Thermochemical Tables; ccCA gives G2-1 MAD 1.33 kcal/mol. **They target enthalpies to sub-kJ/mol and take a fixed, moderate-quality geometry as an input, not as an output. None produces a better rotational constant.** Gn additionally contains a fitted higher-level correction with no physical meaning, which disqualifies it for a geometry-driven problem.

### 9A.5 Prohibitions

**Prohibition 1 — no additive diffuse-function correction, ever.** Adding a diffuse-function correction as an increment rather than computing in the diffuse basis moved the A15 interaction-energy MAE from **1.52 % to 12.74 %** (aug), 10.6 % (jul), 3.94 % (jun-QZ) `[M]`; the authors' conclusion is unambiguous: "the 'Δα approach' is thus not recommended for any application to energies" ([jun-ChS](https://cris.unibo.it/retrieve/fdcbe2fd-290c-49a7-88cc-01c515d136bd/Extension%20of%20the%20%E2%80%9Ccheap%E2%80%9D%20composite%20approach.pdf)). For *geometries* it fails equally: an N···O distance off by 20 mÅ and **CH₄···NH₃ off by 0.2 Å** ([junChS-F12](https://cris.unibo.it/retrieve/handle/11585/868585/ae4939e6-d216-426d-9d79-edb47b92c82c/junChS-F12.pdf)). **Rule: diffuse functions must be present in the underlying basis of every leg of a composite for a weak complex; they may never be added as an increment.** This is a *silent* failure — it produces a plausible number eight times worse than the uncorrected one — so it is also listed in §16 as a failure mode.

**Prohibition 2 — ONIOM and QM/QM2 are rejected at 5–10 atoms.** ORCA's QM/QM2 works and is cheap — barriers of 24.34 (full ωB97X-D3), 24.49 (ωB97X-D3/PBE), 25.48 (MPQC CCSD(T)-F12/PBE) against 25.65 kcal/mol reference, in 30 seconds on 16 cores ([ORCA multiscale tutorial](https://www.faccts.de/docs/orca/6.1/tutorials/multi/basics-otheroniom.html)). It does not earn its place here for three reasons. (1) **Scale mismatch** — ORCA's multiscale machinery is documented for "tens and hundreds of thousands of atoms" with active regions of "hundreds of atoms" ([ORCA QM/MM general](https://orca-manual.mpi-muelheim.mpg.de/contents/multiscalesimulations/qmmm-general.html)); a 5–10 atom dimer *is* the high-level region and there is nothing left to demote. (2) **No bond to cut** — ONIOM needs a covalent bond and a link atom; a van der Waals dimer has none between the fragments, so the natural partition is fragment-wise, which is not ONIOM but the many-body expansion or simply the frozen-monomer composite of §9A.1. (3) **The saving is zero** — the whole complex at the high level is affordable at every tier from 30 minutes upward. *It becomes relevant only for a complex microsolvated by ≥3 additional waters, i.e. outside this scope.*

**Prohibition 3 — do not treat the many-body expansion as free at trimers.** For a dimer the expansion is trivially exact at second order and contributes nothing. At trimers and above, three-body contributions reach **~15–20 % of the total interaction energy for cyclic water structures, four-body ~1 %** `[M]` ([water many-body review](https://pmc.ncbi.nlm.nih.gov/articles/PMC5450669/)), and Puzzarini's parallel finding is that "high-level (post-MP2) contributions are essential for three-body terms, desirable for four-body terms and negligible for higher terms". **D3 and D4 as used here are pairwise-additive with only an approximate three-body term, and strictly pairwise dispersion deviates "by as much as 2 kcal/mol" on water clusters.** Any trimer row must state that its dispersion model does not carry the true three-body induction.

### 9A.6 The recipe menu R1–R9

Wall times assume 8–16 cores on the reference workstation, a 6–8 atom complex, and the tight `%geom` block of §4.4. Times marked ×/÷ 3 are extrapolated from published anchors, not measured here.

| Rank | Recipe | Steps | Codes / input | Wall (8–16 cores) | Expected error in B | Verdict |
|---|---|---|---|---|---|---|
| **R1** | **Experimental monomers + r²SCAN-3c intermolecular optimisation** | 1. take r_e^SE monomer geometries from the literature or CCCBDB · 2. build the dimer and constrain all intramolecular internals · 3. optimise the 6 intermolecular degrees of freedom · 4. report B_e and state that ΔB_vib is unapplied | ORCA: `! r2SCAN-3c TightSCF DefGrid3` + §4.4 `%geom` + `%geom Constraints {…} end`. **No `D4`, no `gCP` tokens — both are inside the composite** | **2–5 min `[E]`** | **1–3 % in B `[E]`** (r²SCAN-3c has no diffuse functions on the intermolecular coordinate); **A improves to <0.2 %** | the cheapest defensible structure in the document; screening and search-window seeding |
| **R2** | **CCSD(T)/CBS monomers frozen + ωB97M-V/def2-QZVPP intermolecular + MPQC CCSD(T)-F12 single point + VPT2** — *the user's scheme, executed properly* | 1. monomers from literature CCSD(T)/CBS or fc-CCSD(T)/cc-pVTZ (MAD 0.003 Å) · 2. freeze intramolecular coordinates · 3. optimise the 6 intermolecular degrees of freedom at ωB97M-V/def2-QZVPP with §4.4 thresholds · 4. three-leg Boys–Bernardi counterpoise at DLPNO-CCSD(T1)/TightPNO/cc-pVDZ-F12 (paired with CABS) · 5. ΔB_vib from ωB97X-V/def2-TZVPP VPT2 on the semi-rigid manifold · 6. report B_e, ΔB_vib, B₀ and the residual gradient on the frozen coordinates | ORCA throughout | **≈5 h `[E]`** (3 h optimisation + 45 min for 3 CP legs + 1 h VPT2) | **0.4–1.5 % in B_e**, ~0.3–0.5 % if semi-rigid; **A to <0.2 %** | **the best de novo accuracy-per-core-hour row in the document.** Replaces v3's Table 3 12 h and 1 d rows |
| **R3** | **junChS-F12 composite geometry** | 1. CCSD(T)-F12b/jun-cc-pVTZ optimisation · 2. MP2-F12 jun-cc-pVTZ→QZ CBS ΔR · 3. MP2 core-valence ΔR (cc-pwCVTZ, ae − fc) · 4. add parameter-wise · 5. ΔB_vib as R2 step 5 | **Molpro** for the F12 gradient (RMS 4e-6, gradient 1e-6); ORCA for F12 single points only | **8–24 h `[E]`** | **~0.1–0.3 % in B_e**; interaction-energy MUE 0.06 kJ/mol | **best de novo accuracy available. Molpro is commercial — a licence-gated row** |
| **R4** | **ChS / CBS+CV** | 1. fc-CCSD(T)/cc-pVTZ optimisation · 2. + ΔR[MP2/CBS(T→Q), n⁻³] · 3. + ΔR[MP2/CV, cc-pwCVTZ] · 4. parameter-wise addition · 5. ΔB_vib | ORCA compound scripts or CFOUR; **use jun-cc-pVnZ (junChS), never plain cc-pVnZ, for a weak complex** | **6–20 h `[M]`-anchored** | **0.13 % MAE in B_e** for ≤16 atoms; A15 interaction energies 1.20 % | the published "0.1 % accuracy" workhorse for ≤10 atoms, and **licence-free in ORCA** |
| **R5** | **Template-scaled / linear-regression-augmented constants** | 1. revDSD-PBEP86-D4 or rDSD monomer geometry · 2. apply the per-bond correction r = a_XY r^DFT + b_XY · 3. freeze; optimise the intermolecular degrees of freedom at R1 or R2 level | any DFT code plus a spreadsheet; coefficients from [Nano-LEGO](https://pmc.ncbi.nlm.nih.gov/articles/PMC10291548/) | **+seconds** on top of the underlying geometry | monomer frameworks to **<1.5 mÅ**, MAPE(B) 0.08–0.20 % **for the covalent part**; intermolecular error unchanged | free accuracy on A. **Never apply to a B3LYP geometry — it nearly doubles the deviation** |
| **R6** | **Semi-experimental anchoring to a measured parent** | scale the trial geometry to reproduce the measured A, B, C of the parent, then substitute masses | Kisiel suite plus any geometry source | **1 min** | **0.03–0.1 %** | the document's best cell — but it needs a measurement, so it is Product B, not Product A |
| **R7** | **Focal-point composite gradient** | MP2/CBS gradient + δ[CCSD(T)]/small-basis gradient combined at each optimisation step | **Psi4** (has the driver); ORCA needs an `ExtOpt` wrapper | **~3 %** of brute force | "similar to CCSD(T) with a basis set one ζ level higher" | excellent, but the diffuse-increment prohibition means the δ term must itself carry diffuse functions. Psi4 dependency |
| **R8** | **Δ-CCSD(T) single point on a DFT geometry** | DFT geometry, then a counterpoise-corrected DLPNO- or F12-CCSD(T) energy | ORCA | 15 min (×/÷ 3) | **zero improvement in B** | **energy only. Explicitly not a geometry recipe.** Listed to be ruled out — it improves a quantity that does not appear in the rotational Hamiltonian |
| **R9** | **ONIOM / QM-QM2** | high-level fragment inside low-level surroundings | ORCA `!QM/QM2 …` | 30 s on 16 cores | **`n.a.` for 5–10 atoms** | **rejected for this scope** (§9A.5) |

**Which families actually reach 0.1 % in a *reported* constant.** Only two: (a) anything anchored to a measurement — template scaling, semi-experimental anchoring, regression fitted to semi-experimental data; and (b) CBS+CV-class composites, and only for **B_e**, and only for semi-rigid species. **Δ-CCSD(T) on a DFT geometry is worthless for rotational constants.**

### 9A.7 Protocol rules (binding)

1. **One dispersion model per composite.** D4's s_n is "a functional-dependent scaling factor" and ORCA tabulates per-functional s6/s8/a1/a2/s9 — B3LYP 1.0 / 2.02929367 / 0.40868035 / 4.53807137; r²SCAN 1.0 / 0.60187490 / 0.51559235 / 5.77342911; PW6B95 carries a **negative** s8 = −0.31926054 ([ORCA dispersion manual](https://orca-manual.mpi-muelheim.mpg.de/contents/modelchemistries/dispersioncorrections.html)). The D4 parameters must belong to the functional producing the **intermolecular** gradient, not to the monomer method.
2. **Never add D4 to ωB97X-V or ωB97M-V**, whose VV10 nonlocal kernel is already the dispersion treatment. **Never add D4 or gCP to r²SCAN-3c** (has D4 + gCP) **or to ωB97X-3c** (has D4 and deliberately no gCP — "small residual BSSE effects are efficiently absorbed by the D4 damping scheme", [ωB97X-3c README](https://github.com/grimme-lab/wB97X-3c/blob/main/README.md)).
3. **Counterpoise legs are computed at the frozen in-complex geometry**, all three legs identical in geometry and basis. **Do not re-relax the monomer for the monomer legs** — that mixes the deformation energy into the counterpoise term. If you want the deformation energy, compute it as a separate fourth leg and report it separately.
4. **Full counterpoise at double zeta for energies; do not counterpoise the optimisation below triple zeta; half-counterpoise at triple zeta and above; counterpoise-free with F12.** The counterpoise/non-counterpoise spread has already shrunk to <0.5 percentage points with F12 (junChS-F12: CP 0.68 %, half-CP 0.88 %, NCP 1.10 %), which is why it becomes optional there.
5. **In a focal-point energy, counterpoise each term at its own basis level.** The recommended construction averages the counterpoise-corrected and uncorrected MP2 terms and counterpoise-corrects the δ term ([Burns, Marshall & Sherrill](https://pubs.acs.org/doi/10.1021/ct400149j)); "whether CP correction, no correction, or the average is favored depends upon the theoretical method, basis set, and binding motif".
6. **No additive diffuse-function correction, ever** (§9A.5).
7. **Freezing monomers does not remove BSSE.** ChS remains "affected by small, but not entirely negligible BSSE". Both the freeze and counterpoise are required.
8. **Report the residual gradient on the frozen coordinates.** A constrained stationary point is not a stationary point; if it exceeds `TolMaxG` = 1e-5, the deformation channel is non-negligible — escalate to a relaxed-monomer optimisation.
9. **Report the extrapolation formula used.** n⁻³ versus n⁻⁵ is worth 3–5 mÅ, i.e. **0.20–0.34 % in B** `[D]`.
10. **Frozen monomers are safe for dispersion-bound complexes and flagged for strongly hydrogen-bonded ones** (deformation up to 9.5 kcal/mol).
11. **Use r_e^SE, not r₀, for experimental monomers**, or ΔB_vib is double-counted.
12. **Every composite reports B_e**; B₀ requires ΔB_vib. **Trimers and above additionally state that D3/D4 is pairwise-additive** and that three-body terms carry 15–20 % of the interaction energy.

**One residual `n.a.` narrowed.** v3 left the magnitude by which counterpoise-relaxed optimisation lengthens R as `n.a.` It can now be narrowed to **"a few mÅ, sign positive"**: counterpoise-corrected versus non-corrected reference geometries differ "by a few mÅ", with the energetic consequence for jun-ChS being 0.73 % → 0.87–0.95 %. A pm-precision figure for a single dimer still requires the local measurement specified in §17.

---

## 9B. Conformer and isomer search: GOAT, CREST, and the union

### 9B.1 The verdict

**CREST is not superior for these systems, and the default should not change.** Keep GOAT as the primary engine and add CREST as a second, *independent* search whose union with GOAT is carried forward — not as a replacement. Three reasons, in order of weight.

1. **Deduplication is a tie, not a CREST advantage.** ORCA states that GOAT's filtering is "**precisely the same as that of CREST**" — RMSD 0.125 Å, ΔE 0.100 kcal/mol, rotational-constant difference 1.00–2.50 % ([ORCA GOAT manual](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/goat.html)) — matching CREGEN's Ewin 6.0, Ethr 0.05, Bthr 1.0–2.5 %, Rthr 0.125 Å ([CREST, *J. Chem. Phys.* 160, 114110](https://pubs.aip.org/aip/jcp/article/160/11/114110/3278084/CREST-A-program-for-the-exploration-of-low-energy)). **Any v3 text implying CREGEN is the better filter is corrected.** CREGEN's value is as an engine-neutral *referee over the union*, not as a better algorithm.
2. **The one independent head-to-head with a coverage metric favours GOAT.** The `racer` transition-state conformer benchmark, 20 reactions, one CPU core each, gives **GOAT average F1 = 0.93** — the best of all methods tested — against **CREST 0.74–0.80** `[M]` ([racer benchmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977065/)).
3. **CREST's mechanism is specifically hostile to weakly bound complexes.** Its bias is a function of Cartesian RMSD from previously visited minima, and for a bound complex the cheapest way to raise RMSD is to pull the fragments apart. **CREST's own NCI mode exists to patch this**, auto-adding an ellipsoidal wall because otherwise maximising RMSD "would … dissociate the complex within a few ps" ([CREST NCI example](https://crest-lab.github.io/crest-docs/page/examples/example_3.html)).

**The MLFF-driven GOAT experience is documented practice, not an off-label hack** (§9B.4).

### 9B.2 The comparison table

| Dimension | **ORCA GOAT** | **CREST** |
|---|---|---|
| Algorithm | stochastic uphill push → re-optimise (basin/minima-hopping lineage); **no molecular dynamics** ([GOAT manual](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/goat.html)) | RMSD-biased metadynamics (iMTD-GC) + 400/500 K MD + genetic Z-matrix crossing; an iMTD-sMTD variant ([CREST workflows](https://crest-lab.github.io/crest-docs/page/overview/workflows.html)) |
| Work scaling | ~**100 × N_atoms** optimisations, reducible to **<3 × N_atoms** with aggressive settings; per-worker floor max(N,15) | MTD length ≈ **0.3–0.4 × N ps per MTD** × many runs, plus MD blocks and crossing; thousands of optimisations |
| **Scaling at N = 10** | ~1,000 optimisations — trivial at GFN2-xTB | **CREST does not get proportionally cheaper as the system shrinks: the MD block has a floor of ~3–4 ps per run** |
| Deduplication | RMSD 0.125 Å, ΔE 0.100 kcal/mol, ΔB 1.00–2.50 %, "precisely the same as that of CREST" | CREGEN: D4 topology check → Ewin 6.0 → ΔE 0.05 → \|ΔB\| 1.0–2.5 % → RMSD 0.125 Å |
| Topology handling | `BONDFACTOR 1.2`, `MAXTOPODIFF`; **`GOAT-EXPLORE` drops the topology constraint entirely** | CREGEN **discards** structures whose bond topology differs from the input **by default** — a feature for conformers, **a defect for isomer searches where a changed contact topology is the answer**. Disable with `--noreftopo` |
| Cost (independent benchmark) | ≈**125× CREST** on constrained transition-state conformers — but GOAT ran at GFN2-xTB while CREST ran GFN2//GFN-FF, so this is partly a level-of-theory artefact | 4× faster than that GOAT setup; **38 min measured** for a 33-atom constrained NCI-MTD on a laptop at GFN-FF, returning 122 conformers `[M]` |
| Coverage quality | **F1 0.93** `[M]` | F1 0.74–0.80, but returns the most structures in 16 of 20 cases — **"the number of conformers does not necessarily correlate with the amount of conformer space explored"** |
| Method flexibility | **any ORCA gradient**: DFT, QM/MM, ONIOM, broken symmetry, excited states, `!ExtOpt` MLFF — "all you need is the gradient". **The search engine and the energy method are orthogonal** | tblite / GFN0 / GFN-FF native; ORCA and a `generic` script calculator via TOML (`gradtype = "engrad"`); ONIOM. DFT-through-CREST is **explicitly advised against** by the authors |
| Constraints | exact fixing | **"does not allow exact fixing; it only allows constraining"** — frozen atoms drift |
| Strengths | best measured coverage; exact constraints; composes with any method; cheap at small N; ensemble plus S_conf/G_conf out of the box | free and open (LGPL-3.0); CREGEN as an engine-neutral referee; NCI mode; entropy mode (≈1 cal mol⁻¹ K⁻¹ accuracy); protonate/deprotonate/tautomerize; quantum cluster growth for microsolvation |
| Failure modes | fragment blow-off during the uphill push; **ambiguous `MAXEN`**; degeneracy default g_i = 1; numerical gradients through `ExtOpt`; expensive at DFT (ORCA itself recommends `%PAL NPROCS 32` for r²SCAN-3c and warns "be prepared to use many cores or wait for a few days") | **RMSD bias dissociates weak complexes (4.2 ps)**; start-structure sensitivity of 50 kJ/mol; run-to-run irreproducibility; soft constraints; genetic crossing fails on clusters |
| Licence | ORCA (free for academics) | free, open-source, LGPL-3.0 |
| **Best use here** | **primary engine**; `GOAT-EXPLORE` for binding-site isomerism; MLFF-driven enumeration | **independent cross-check** (`--nci`), **CREGEN as union referee**, entropy convergence, microsolvation |

**The quantified CREST failure mode on this system class.** An independent study of flexible non-covalent clusters (asphaltene dimers and trimers, 70–120 atoms) measured: hydrogen bonds and π-stacking **disrupted after 4.2 ps**; most generated conformers >12 kcal/mol above the start and therefore screened out; the best conformer usually nearly identical to the input; **strong starting-structure sensitivity — a different start gave a conformer 50 kJ/mol lower**; two runs from the same start giving very different results; the problem **not** fixed by longer runs, NCI mode, manual constraining potentials or shorter sampling intervals; and genetic crossing that "works as intended for single molecules but not for clusters" `[M]` ([LEDE-CREST](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/646b83b4b3dd6a65308e7595/original/a-variant-on-the-crest-algorithm-for-non-covalent-clusters-of-flexible-molecules.pdf)). Their fix is to cut the bias: `kpush` 0.015–0.1 Eh instead of the standard 0.1688–1.125 Eh or the NCI-mode 0.1125–0.225 Eh.

**What the microwave literature actually uses, which is sobering.** CREST is the community default and nobody in the fetched sample used GOAT: furan⋯n-hexane (CREST, 34 minima within 5 kJ/mol, global minimum assigned); the trifluorobutanol dimer (CREST + manual dihedral scans, ~1,500 candidates → 102 → 16 → 5 observed); a flexible monomer where **CREST alone gave 176 conformers but CREST + a second engine gave 18 distinct within 1000 cm⁻¹, with two found only by CREST and three only by the other engine** ([*Molecules* 26, 5162](https://www.mdpi.com/1420-3049/26/17/5162)); and HFIP⋯Ne / HFIP⋯Ar, where **no automated search was used at all** — nine binding topologies were enumerated by hand and the predicted most stable was the one observed ([HFIP⋯Rg](https://pubs.acs.org/doi/10.1021/acs.jpca.1c03757)).

Three conclusions. (a) **CREST's dominance in this literature is sociological, not evidential** — it predates GOAT by five years and no paper in the sample benchmarked the two. (b) The *Molecules* case is the strongest empirical argument for **union merging**: two independent engines each found conformers the other missed. (c) For genuinely small rare-gas complexes, published practice is **hand enumeration of binding topologies, and it works** — the number of distinct minima is small enough to enumerate exhaustively, which neither GOAT nor CREST can claim to prove.

### 9B.3 The six-step union protocol

**Step 0 — seeds and the completeness argument.** Build 3–6 chemically distinct starting geometries, one per plausible binding site or topology. For 5–10 atom complexes, **enumerate the binding topologies by hand**, as done for HFIP⋯Ne/Ar. *This hand enumeration, not the stochastic search, is your completeness argument, and it is the only route in this document that can support one.*

**Step 1 — MLFF-driven GOAT (enumeration, minutes).** Start the server first, then per seed:

```bash
~/bin/oet-aimnet2/oet_server aimnet2 --nthreads 4 -d cuda &
```
```text
! GOAT-EXPLORE ExtOpt TightOpt PAL8
%method
  ProgExt "<USER_HOME>/bin/oet-aimnet2/oet_client"
  Ext_Params "-b localhost:8888"
end
%scf TolE 1e-5 end
%goat
  maxen 12.0
  conftemp 298.15
  confdegen auto
  maxglobaliter 100
  minglobaliter 3
end
* xyzfile 0 1 seed01.xyz
```

Output: `seedNN.finalensemble.xyz`. **Never quote these energies or geometries.**

**Step 2 — GOAT at GFN2-xTB (independent, no MLFF bias).**

```text
! GOAT XTB2 PAL8
%goat maxen 12.0 confdegen auto gfnuphill gfnff end
* xyzfile 0 1 seed01.xyz
```

**Step 3 — CREST NCI cross-check.**

```bash
crest seed01.xyz --nci --gfn2 --ewin 12 --nocross --noreftopo --T 8 --niceprint
# if the complex still dissociates, cut the bias or tighten the wall:
crest seed01.xyz --nci --wscal 0.9 --gfn2 --ewin 12 --T 8
```

`--nocross` because genetic crossing is invalid for clusters; `--noreftopo` because the default topology check silently deletes isomers with a different contact topology, which for an isomer search is the answer rather than the noise; `--ewin 12` because the 6 kcal/mol default is too tight.

**Step 4 — union, one common level, one referee.**

```bash
cat seed*.finalensemble.xyz crest_conformers.xyz > union.xyz
crest --screen union.xyz --gfn2 --ewin 12 --T 8      # re-optimise at one level, then CREGEN
```

Then re-optimise the survivors in ORCA (`! r2SCAN-3c TightOpt Freq TightSCF DefGrid3 PAL8`) and referee the QM ensemble:

```bash
crest --cregen r2scan_ensemble.xyz --ewin 12 --ethr 0.05 --bthr 0.01 --rthr 0.125
```

**Do not compare pre-deduplicated lists from two engines.** GOAT's degeneracy convention (g_i = 1 by default) differs from CREST's, which folds rotamer degeneracy into S_conf, and identical structures optimised at different levels differ by more than `--ethr`. **Re-optimise the union at one common level before CREGEN**, and set `CONFDEGEN auto` before comparing any entropy.

**Step 5 — spectroscopic deduplication.** Two structures whose rotational constants agree to better than the tier's accuracy window are not distinguishable by the experiment; merge them and say so. At 12 GHz, 0.1 % = 12 MHz. **CREGEN's and GOAT's default `bthr` of 1 % ≈ 120 MHz is ten times looser than a microwave experiment can resolve** — for spectroscopic purposes tighten to `--bthr 0.001` and treat that as Stage B of the two-stage deduplication protocol.

**Step 6 — report the diagnostics, and state the limit.** Report the number of isomers found by GOAT alone, by CREST alone, by both, and in the union; the number of independent seeds; the S_conf trajectory from each engine; and an explicit statement that no completeness proof exists. **Both engines are stochastic global optimisers — "keep going until nothing new turns up" heuristics — and neither delivers a completeness proof.** The only situation in which exhaustiveness may be claimed is a rigid or near-rigid complex whose binding topologies are hand-enumerable. Otherwise report "no new isomer found in N independent runs from M distinct seeds", which is a convergence statement, not a proof.

**The union-coverage diagnostic is a mandatory reported quantity**, warranted by the case where two engines each found conformers the other missed.

### 9B.4 The MLFF-GOAT recipe, with its hard limit

**It is officially supported.** `!ExtOpt` is explicitly combinable with `!GOAT`, and the official tutorial's **Example 3 is literally GOAT + AIMNet2 on ibuprofen** with input `! EXTOPT GOAT PAL8`, producing `basename.finalensemble.xyz` ([ORCA external-methods tutorial](https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extopt.html)). The wrapper repository ships **AIMNet2, MACE, UMA (via fairchem), g-xTB, MOPAC and MLatom**, with `oet_server`/`oet_client` for persistent serving ([orca-external-tools](https://github.com/faccts/orca-external-tools)).

**Model choice.** AIMNet2 as the default for organic van der Waals complexes (fast, small, well-tested in the tutorial); UMA when broader element coverage is needed; MACE when you intend to fine-tune later — noting that `oet_mace` supports only the `mp` and `omol` suites, **not MACE-OFF**.

**Three practical details v3 lacked.**

1. **AIMNet2 and UMA dependencies are mutually incompatible and need two separate virtual environments.** Python ≥ 3.11 is required, and the scripts link to the venv's absolute path, so **do not move it afterwards**:
   ```bash
   git clone https://github.com/faccts/orca-external-tools.git && cd orca-external-tools
   python install.py --venv-dir ~/oet-aimnet2-venv --script-dir ~/bin/oet-aimnet2 -e aimnet2
   python install.py --venv-dir ~/oet-uma-venv     --script-dir ~/bin/oet-uma     -e uma
   ```
2. **Server mode is effectively mandatory.** Standalone mode re-imports the whole ML stack on every gradient; a first MACE call including model load is **≈30 s** against **≈48 ms** steady state. A GOAT run issues ~100 × N_atoms optimisations, so above ~100 calls the difference is between minutes and hundreds of hours (§8A.2).
3. **AIMNet2's float32 precision (~4 × 10⁻⁶ Eh) sits at or below ORCA's default `TolE` = 5 × 10⁻⁶ Eh**, so without `! TightOpt` and `%scf TolE 1e-5 end` the optimiser chases numerical noise ([`oet_aimnet2` readme](https://raw.githubusercontent.com/faccts/orca-external-tools/main/readmes/aimnet2.md)).

**A fourth detail worth budgeting for:** g-xTB gradients through `ExtOpt` are computed **numerically**, which the tutorial warns slows "applications that compute gradients often, such as GOAT". That is a 3N–6N penalty. **Measure one gradient locally before budgeting a g-xTB GOAT run.**

**The hard limit, stated plainly.** Foundation-model **interaction** energies — precisely the quantity that ranks van der Waals isomers — carry errors of **3.5–7.3 kcal/mol on S30L and 29.9 kcal/mol on PLA15** `[M]`, improving only to 3.52/3.35 kcal/mol for the best current model ([MACE-POLAR-1](https://arxiv.org/html/2602.19411v1)). **MLFF-driven GOAT is an excellent enumerator and a poor judge.** Therefore: keep a wide window (`MAXEN 12.0`, not 6), escalate every survivor to r²SCAN-3c or better, and **never report an MLFF geometry or an MLFF energy ordering.** Guard G4 of §8A.5 is the enforcement mechanism.

**A documentation conflict to work around.** GOAT's `%GOAT` keyword table lists `MAXEN 6.0` kcal/mol while the body text of the same manual page says "By default conformers up to 12.0 kcal/mol from the global minimum are included." **Set `MAXEN` explicitly in every production input rather than trusting the default.**

**Variants worth knowing.** `GOAT-ENTROPY` stops on ΔS_conf < 0.1 cal mol⁻¹ K⁻¹; **`GOAT-EXPLORE` removes the topology constraint and is the right variant when the "isomers" are different binding sites of the same two fragments** rather than torsional conformers; `GOAT-REACT` sets `MAXTOPODIFF 8` and `AUTOWALL true`; `GOAT-DIVERSITY` is energy-blind; `GOAT-COARSE` treats fragments as rigid bodies with a claimed >10× speed-up. Defaults: 8 workers, `MINGLOBALITER 3`, `MAXGLOBALITER 100`, `MAXITER 128`, `MAXOPTITER 256`, `MAXCORESOPT 32`, `CONFTEMP` 298.15 K.

**CREST-side external methods, so the document stops implying CREST cannot drive them.** CREST 3.0 replaced subprocess calls to `xtb` with a modular calculator API. `method = "orca"` in `[[calculation.level]]` drives ORCA (the authors advise against it — DFT inside iMTD-GC "would lead to unfavorably high computation times"), and `method = "generic"` with `bin`, `gradfile` and `gradtype = "engrad"` drives **any** program or script, writing coordinates to `genericinp.xyz` ([CREST input files](https://crest-lab.github.io/crest-docs/page/documentation/inputfiles.html)). **This is the same `.engrad` contract as `!ExtOpt`, so one wrapper script serves both engines.** There is no dedicated MLIP or ASE calculator in CREST — **`n.a.`** — so the generic route is the only path, and it re-introduces the subprocess overhead the CREST paper warns "should generally be avoided for procedures with a high number of calculator evaluations".

**Flag-syntax correction.** It is `--gfn1` / `--gfn2` / `--gff` / `--gfn2//gfnff`, **not `--gfn <n>`**; and `-xnam <BIN>` takes one dash. Use CREST ≥ 3.0.2, which is 1.6–4.7× faster than 2.x.

### 9B.5 Where the search sits in the budget

At 5–10 atoms **every semi-empirical and MLFF search lands in the ≤1 h tiers, so the search is never the bottleneck.** The ladder above 3 h should be spent on energies and anharmonic corrections, not on more searching. The exception is the 12 h row, a direct `! GOAT r2SCAN-3c` on the two or three leading isomers, which catches basins the semi-empirical or MLFF surface misplaced — and which is at risk on 16 threads, since ORCA itself advises 32 cores and "a few days" for an r²SCAN-3c GOAT.

**All conformer-search wall times in this document are estimates.** No measured GOAT or CREST timing on this hardware exists in any source consulted. The only measured anchors are 38 min for a 33-atom constrained NCI-MTD on a laptop and the CREST 3.0 speed-up factor. **`n.a.` until locally calibrated**; the mandatory calibration is one GOAT run on the actual complex with the wall time recorded.

**Quantum cluster growth (`--qcg`) operational details are `n.a.`** — the documentation page indexed for it contains only a citation — and it is off the critical path for a binary complex.

---

## 10. The ORCA external-tool contract, implemented

This section is normative. It is the user's granted exception: `ExtOpt` and OPI's `ExternalTools.EXTOPT` bring g-xTB and machine-learned potentials inside ORCA, and this is exactly how they are wired.

### 10.1 Invocation

```
! ExtOpt
%method
  ProgExt   "/full/path/to/wrapperscript"
  Ext_Params "optional command line arguments"
end
```

Alternatives to `ProgExt`: a file or link named `otool_external` next to the ORCA executables, or the `EXTOPTEXE` environment variable set to the full path. **"All information that you give on the electronic structure is discarded."** ([ORCA 6.1 manual, optimizing with external methods](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations.html))

In OPI 2.0 the same mechanism is:

```python
from opi.input.simple_keywords import ExternalTools, Opt
calc.input.add_simple_keywords(ExternalTools.EXTOPT, Opt.OPT)
calc.input.add_blocks(BlockMethod(ProgExt=path_to_wrapper, Ext_Params='"--method PM7"'))
```

Both quote types are required in `Ext_Params`; OPI requires ORCA ≥ 6.1.1 and Python ≥ 3.11 ([External Methods — OPI 2.0 docs](https://www.faccts.de/docs/opi/2.0/docs/contents/notebooks/extopt.html); [faccts/opi](https://github.com/faccts/opi); [OPI paper](https://pubmed.ncbi.nlm.nih.gov/41885262/)).

### 10.2 The file contract, verbatim

ORCA writes `basename_EXT.extinp.tmp` at every step, containing one item per line:

```
basename_EXT.xyz    # xyz filename: string, ending in '.xyz'
0                   # charge: integer
1                   # multiplicity: positive integer
1                   # NCores: positive integer
0                   # do gradient: 0 or 1
pointcharges.pc     # point charge filename: string (optional)
```

"Comments from `#` until the end of the line should be ignored." ORCA also writes `basename_EXT.xyz` (standard XYZ, ångström) to the working directory, then calls `scriptname basename_EXT.extinp.tmp [args]`. The wrapper must produce `basename_EXT.engrad`:

```
#
# Number of atoms: must match the XYZ
#
3
#
# The current total energy in Eh
#
-5.504066223730
#
# The current gradient in Eh/bohr: Atom1X, Atom1Y, Atom1Z, Atom2X, etc.
#
-0.000123241583
 0.000000000160
 ...
```

"Comments from `#` until the end of the line are ignored, as are any comment-only lines." ORCA 5 required exactly three comment lines between sections; ORCA 6 does not. Output control: `%output Print[P_EXT_OUT] 1` and `Print[P_EXT_GRAD] 1`, both default 1. Exit code 0 means success; a non-zero exit aborts the ORCA job. ([ORCA 6.1 manual, optimizing with external methods](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations.html))

**The LAM document's filename `.extcomp.inp` is wrong for ORCA 6.1 and must be corrected wherever it appears.**

### 10.3 Units and the sign flip

```
input coordinates  Angstrom
output energy      Eh
output gradient    Eh/bohr        <-- NOT eV/Angstrom, NOT Eh/Angstrom
conversions from ASE (eV, eV/A):
    E_Eh    = E_eV / 27.211386245988
    g_Eh_a0 = (-F_eV_per_A) * 0.529177210903 / 27.211386245988
```

**The sign flip is mandatory: ASE returns forces, ORCA wants the gradient. A wrapper that omits it will optimise uphill.** Getting either the sign or the unit wrong produces an "optimisation" that runs uphill or converges to a scaled geometry, and neither failure announces itself.

### 10.4 What `ExtOpt` unlocks, and what it does not

**Unlocked** — anything driven purely by energy and its first derivative: `Opt`, `OptTS`, `NEB-TS`/`NEB-CI`, `IRC`, `GOAT`, `%geom Scan`, `MD`, `NumFreq`. The ORCA 6.1 tutorial demonstrates `Opt`, `GOAT`, `NEB-TS`, `FREQ` and `NUMFREQ` with external methods ([ORCA with external methods](https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extopt.html)).

**Not unlocked** — anything requiring the wavefunction or an analytic second derivative: analytic `Freq`, **VPT2** (which requires analytic Hessians), dipoles and polarizabilities, TD-DFT, LED, NBO and population analysis, magnetic properties, and CPCM/SMD solvation of the external method. For AIMNet2 specifically, `--compile` is "incompatible with Hessians; do not use with NEB, OptTS, IRC" ([`oet_aimnet2` readme](https://raw.githubusercontent.com/faccts/orca-external-tools/main/readmes/aimnet2.md)).

### 10.5 Two verified constraints that must be stated plainly

**(i) `oet_mace` supports only the `mp` and `omol` suites. MACE-OFF is not supported.** The wrapper's readme lists exactly two suites for `-s/--suite`: `mp` (mace-mp, Materials Project) and `omol` (mace-omol, the OMOL foundation model), with `omol` the default; other options are `-m/--model` (MP: `medium-mpa-0`, `medium`, `small`; OMOL: `extra_large` or a local path), `--default-dtype` (`float32` for MD speed, `float64` for optimisation accuracy), `--device`, and MP-only `--dispersion`, `--damping`, `--dispersion-xc`, `--dispersion-cutoff`, `--head` ([`oet_mace` readme](https://raw.githubusercontent.com/faccts/orca-external-tools/main/readmes/mace.md)). **Every row claiming "GOAT with MACE-OFF24(M) inside ORCA" presupposes a wrapper that does not ship. Either write it (Section 10.6) or switch the row to MACE-OMOL-0 or AIMNet2.**

FACCTs ships wrappers for MOPAC (PM7), AIMNet2, UMA (via fairchem), g-xTB, MLatom, MACE-MP and MACE-OMOL, plus an `oet_server`/`oet_client` pair, installed by `python install.py -e aimnet2` ([faccts/orca-external-tools](https://github.com/faccts/orca-external-tools)). **A user-written wrapper is required for:** MACE-OFF, Orb, SevenNet, ANI, and any fine-tuned or Δ-learned local model.

**(ii) g-xTB gradients through the `ExtOpt` path are numerical.** ORCA's own tutorial states: "The g-xTB binary is a preliminary version that currently works only on Linux-based systems. **Gradients are computed numerically.** Therefore, the gradient calculations take a significant time, slowing down applications that compute gradients often like GOAT" ([ORCA 6.1 tutorial, optimisation with external methods](https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extopt.html)). For a nonlinear N-atom system this multiplies every gradient by 6N, i.e. **60× at N = 10**, and **demotes g-xTB GOAT by roughly two tiers unless the run is energies-only.**

There is a countervailing claim: the g-xTB repository documents `xtb struc.xyz --gxtb --grad` and `--opt` and describes numerical Hessians built *from analytic gradients* ([grimme-lab/g-xtb](https://github.com/grimme-lab/g-xtb)), and a FACCTs social-media note claims that renaming the binary to `otool_xtb` and using `%xtb XTBInputString "--gxtb" end` gives "efficient fully analytic gradients" ([FACCTs LinkedIn, 2026-05-11](https://www.linkedin.com/posts/faccts_orca-xtb-gxtb-activity-7452978014499495937-tiY3)). That last source is social-media grade and is flagged as unverified. **The document's position: budget for numerical gradients through `ExtOpt`, and measure locally before assuming otherwise. The answer moves the g-xTB GOAT tier by a factor of about 6N.**

**Two operational rules.** Use `oet_server`/`oet_client -b host:port` for any tier making more than about 100 model calls, because MACE's first call costs roughly 30 s including compilation and spawning that per gradient makes GOAT — about 100 × N_atoms optimisations — unusable ([faccts/orca-external-tools](https://github.com/faccts/orca-external-tools); [Speeding up MACE](https://arxiv.org/html/2510.23621v1)). And **AIMNet2 runs float32 internally with absolute-energy precision of about 4 × 10⁻⁶ Eh, at or below ORCA's default `TolE = 5 × 10⁻⁶ Eh`**, so pair it with `! TightOpt` and `%scf TolE 1e-5 end` or the optimiser will chase numerical noise ([`oet_aimnet2` readme](https://raw.githubusercontent.com/faccts/orca-external-tools/main/readmes/aimnet2.md)).

### 10.6 Worked wrapper design (a): g-xTB

Already shipped as `oet_gxtb`; use it rather than writing one.

```
! ExtOpt Opt Freq NumFreq
%method
  ProgExt "/opt/orca-external-tools/oet_gxtb"
end
* xyzfile 0 1 complex.xyz
```

Parameter files must be placed in `~/`. The binary is Linux-only and preliminary. A macOS parallel-diagonalisation bug in numerical Hessians is documented, so run serially there ([grimme-lab/g-xtb](https://github.com/grimme-lab/g-xtb)). Budget every gradient at 6N energy evaluations.

### 10.7 Worked wrapper design (b): a MACE model via ASE

Needed because `oet_mace` does not support the MACE-OFF suite.

```python
#!/usr/bin/env python3
# oet_maceoff -- ORCA ExtOpt wrapper for MACE-OFF via mace-torch / ASE.
# Invoked by ORCA as:  oet_maceoff <base>_EXT.extinp.tmp [args]
import sys, argparse
from ase.io import read
from mace.calculators import mace_off

EH_PER_EV  = 1.0 / 27.211386245988
BOHR_PER_A = 1.0 / 0.529177210903          # A^-1 -> bohr^-1

def read_extinp(path):
    vals = [l.split('#')[0].strip() for l in open(path)
            if l.split('#')[0].strip()]
    xyz, chrg, mult, ncores, dograd = (vals[0], int(vals[1]), int(vals[2]),
                                       int(vals[3]), int(vals[4]))
    pcfile = vals[5] if len(vals) > 5 else None
    return xyz, chrg, mult, ncores, dograd, pcfile

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extinp")
    ap.add_argument("-m", "--model",  default="medium")    # small|medium|large
    ap.add_argument("-d", "--device", default="cuda")
    ap.add_argument("--dtype",        default="float64")   # float64 for Opt
    a = ap.parse_args()

    xyz, chrg, mult, ncores, dograd, pcfile = read_extinp(a.extinp)
    if pcfile:                     # MACE-OFF has no point-charge embedding
        sys.exit("point charges not supported by this wrapper")
    if chrg != 0 or mult != 1:     # MACE-OFF is neutral closed-shell only
        sys.exit("MACE-OFF is trained on neutral, closed-shell systems only")

    atoms = read(xyz)                                      # Angstrom
    atoms.calc = mace_off(model=a.model, device=a.device,
                          default_dtype=a.dtype)

    e_eV = atoms.get_potential_energy()                    # eV
    e_Eh = e_eV * EH_PER_EV

    base = a.extinp.rsplit(".extinp.tmp", 1)[0]
    with open(base + ".engrad", "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{e_Eh:.12f}\n")
        if dograd:
            forces = atoms.get_forces()                    # eV/Angstrom
            for fx, fy, fz in forces:
                for comp in (fx, fy, fz):
                    # gradient = -force ; eV/A -> Eh/bohr
                    f.write(f"{-comp * EH_PER_EV / BOHR_PER_A:.12f}\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

Three implementation notes that will otherwise cost a day each. **Sign and units** — see Section 10.3. **Warm-up** — use the server/client pattern above about 100 calls. **Precision** — `float64` for optimisations, `float32` only for molecular dynamics ([`oet_mace` readme](https://raw.githubusercontent.com/faccts/orca-external-tools/main/readmes/mace.md)); and recall from Section 4.4 that no float32 model can converge an intermolecular geometry to the corrected `TolMaxG`.

### 10.8 Committee uncertainty inside the wrapper

ORCA has no notion of an ensemble, so the committee lives in the wrapper:

```
for each member m in {0..M-1}:
    E_m, g_m = model_m(geometry)
E_bar   = mean(E_m);   g_bar = mean(g_m)
sigma_E = std(E_m) / sqrt(natoms)                 # normalised estimator
U_F     = max over atoms of max over m |g_m,i - g_bar,i|
if sigma_E > eps_E or U_F > eps_F:
    write a marker file and (optionally) exit non-zero to halt the run
write E_bar and g_bar to .engrad
```

Thresholds ε from the training-error distribution as Q₃ + 1.5 × IQR ([arXiv:2508.03405](https://arxiv.org/html/2508.03405v1)). For AIMNet2 this is what the readme itself recommends — "for production run all four and average energies/gradients outside ORCA", the wrapper's `--ensemble-member` flag selecting one member at a time ([`oet_aimnet2` readme](https://raw.githubusercontent.com/faccts/orca-external-tools/main/readmes/aimnet2.md)) — and the wrapper is the right place to do it.

---

## 11. Software licensing

### 11.1 ORCA

ORCA is free of charge for academic use, with commercial licences through FACCTs. The academic end-user licence agreement grants "a simple, non-exclusive, non-transferable and non-sublicensable right to use" the software, and states that the licensee "is not entitled … to transfer SOFTWARE or make it available to third parties in any form … In particular, LICENSEE is not permitted to sell, loan, lease or rent the SOFTWARE or sub-license it in any other way, or **to reproduce SOFTWARE or make it accessible publicly**", nor "to transfer SOFTWARE from one computer via a network or other data transfer channel to another computer or another receiving device if said transfer is not within the scope of this Agreement". Use is "exclusively in ACADEMIA for ACADEMIC PURPOSES and for PRIVATE USE", where academic purpose expressly excludes "research and development in cooperation or other collaboration with or sponsored by a for-profit organization" ([ORCA End User License Agreement](https://www.hpc.unipr.it/dokuwiki/lib/exe/fetch.php?media=calcoloscientifico:cluster:softwareapplicativo:orca4.0.x-eula.pdf); [ORCA EULA, June 2021 copy](https://hpc.hku.hk/wp-content/uploads/document/orca-eula_2021.pdf)).

Three consequences bear on deployment.

1. **A shared prebuilt Codespaces or container image containing ORCA may not be distributed to a class.** Publishing such an image is making the software available to third parties. Each student must register, accept the licence individually and obtain their own binaries, which is how ORCA distribution works in practice; Texas A&M, for example, requires every user to register on the ORCA Forum and accept the EULA before access is granted ([Texas A&M HPRC](https://hprc.tamu.edu/kb/Software/ORCA/)).
2. **A public repository or public container-registry image carrying an ORCA tarball is not permitted.** A personal codespace into which a student loads binaries they themselves downloaded is defensible; a shared prebuilt image is not.
3. **Codespaces prebuilds are created by running a workflow on third-party runners and are cached and distributed from third-party infrastructure**, which engages the network-transfer clause directly.

**The conclusion neither version 1 nor version 2 drew, and which this version draws: the teaching stack contains no ORCA at all** (Section 19).

Two caveats stated rather than glossed. The publicly retrievable EULA text is the ORCA 4.0.x MPI version; ORCA 6 is distributed under FACCTs terms that could not be retrieved, so the ORCA 6.1.x-specific wording is **`n.a.`** here. The prohibition on public reproduction is the operative clause and is very unlikely to have loosened. And whether routing an academic teaching workflow through commercially operated infrastructure engages the for-profit exclusion is an institutional legal determination rather than a technical one, and is **`n.a.`**

### 11.2 Machine-learned models

The MACE-OFF models are distributed under an **Academic Software Licence**: "By downloading the models you agree to the Academic Software License … free to use them for academic purpose … not for commercial purposes" ([ACEsuit/mace-off](https://github.com/ACEsuit/mace-off)); MACE-OMOL-0 and MACE-POLAR-1 are likewise academic-licensed ([ACEsuit/mace](https://github.com/acesuit/mace)). The eSEN and OMol25 checkpoints carry the **FAIR Chemistry License v1**, with an acceptable-use policy and geographic restrictions, over a CC-BY-4.0 dataset ([facebook/OMol25](https://huggingface.co/facebook/OMol25)). **Where a deliverable will be distributed beyond academic use, or where an industrial partner is involved, prefer Orb (Apache-2.0) or SevenNet (MIT).** This is a deployment blocker, not a footnote.

### 11.3 Other components

g-xTB is GPL-3.0, distributed as preliminary modified `xtb` 6.7.1 binaries pending a `tblite` implementation, and any deployment should track the terms attached to that release ([grimme-lab/g-xtb](https://github.com/grimme-lab/g-xtb)). xtb, CREST, Psi4 (all LGPL-3.0), PySCF and JAX (Apache-2.0), ASE (LGPL), NumPy and SciPy (BSD) are freely redistributable. Molpro is commercial. CFOUR, MCTDH, MULTIMODE and ABCluster are academic-registration. SPFIT/SPCAT, PGOPHER, pyckett, PySpecTools and the Kisiel program suite are free. **The licence terms of QCxMS, QCxMS2, AIMNet2 and SHARC were not audited for this revision and are `n.a.`;** they must be checked before any non-academic deployment.

---

## 12. How to read the tier tables

### 12.1 The reference configuration

Every wall-clock tier is a **design target on one named reference configuration**: eight performance cores of an Intel Core i7-13700K running an AVX2 build of ORCA 6.1.x with `%maxcore 3000` (or **seven ranks at `%maxcore 3400` whenever a GPU companion stage is co-scheduled**, §8A.1), 64 GB DDR5, NVMe scratch, and an RTX 3090 available for machine-learned inference **and, above roughly 100 basis functions, for double-precision DFT via gpu4pyscf** (§8.2).

**The tiers are estimates anchored on published benchmarks, not measurements on this hardware, and they must be re-measured locally before being used as a scheduling contract** (§20.1). The mandatory calibration is one single point on the actual complex at the tier's level, wall time recorded, all point counts rescaled — **run twice, once with the GPU idle and once with it loaded**, or every pipelined estimate is unanchored.

### 12.2 Two tracks, and where each cannot go

Tables 3, 4, 6 and 8 are presented as **two parallel tracks**, because the codes are not substitutes (§9.1):

- **ORCA track** — search, energetics, F12 and DLPNO refinement, DFT-quality VPT2, Pickett export.
- **CFOUR track** — CCSD(T)-quality anharmonic force fields, sextic centrifugal distortion, spin–rotation, the diagonal Born–Oppenheimer correction, isotopologue force fields from one field.

Where a track cannot fill a tier, the row says so explicitly rather than being omitted. **The CFOUR track has no 10 s and no 1 min row** — it has no semi-empirical engine, no global optimiser and no DFT on its public feature list — **and no DLPNO or F12 row at any tier**. **The ORCA track has no 1 month row for a CCSD(T) anharmonic force field**, and no sextic-distortion capability at any tier. Tables 1, 2, 5, 7, 9 and 10 are single-track (ORCA and its satellites); the reason CFOUR cannot fill them is stated in each table's preamble.

### 12.3 Seven visible columns, and the expansion block

A fifteen-column table cannot be read at a glance in any renderer. Each tier table therefore shows seven columns inline and carries a **per-row expansion block** immediately beneath it, keyed by Row ID.

| Visible column | Content |
|---|---|
| **Row ID** | a stable identifier, `T3-12h` or `T4C-3d`, used by every cross-reference in this document. The letter after the table number, where present, names the track: `O` = ORCA, `C` = CFOUR |
| **Tier** | wall-clock budget on the reference configuration: 10 s, 1 min, 30 min, 1 h, 3 h, 12 h, 1 d, 3 d, 1 w, 1 mo |
| **Method / Code** | the physical method and the program that does the work |
| **Delivers / accuracy** | **the quantity — `B_e`, `B₀`, `ΔB_vib`, `D₀`, an ensemble — and the half-width of the emitted window**, never a point estimate, with a provenance tag |
| **Conc.** | the concurrency class (below) |
| **Product** | A (absolute de novo), B (semi-experimental / template-anchored), C (differences) |
| **Limitations** | a real limitation of this row. **No row is exempt and no cell is empty** |

| Expansion column | Content |
|---|---|
| **Core-h** | C = cores × wall-clock hours **at the row's own execution setup**. Setup-3-only rows quote the 128-core figure, not an 8-core-derived one |
| **Input / workflow** | valid ORCA 6.1.x or CFOUR syntax. Blocks go in `%block … end`, never on the `!` line |
| **State-in / State-out** | the exact files consumed and emitted, with the keyword that reads them, plus a restart flag: **R** = restartable from its own output, **D** = not restartable, must be decomposed under a cap (§8B) |
| **Frozen-mono** | `relaxed`, `frozen-iso`, `frozen-inc`, or `—` where the row has no geometry (§9A.2) |
| **Mem / scratch** | peak per rank and peak scratch — the binding constraint on Setups 1 and 3 more often than time |
| **Licence** | free, academic registration, or commercial |
| **Setup 1?** | teaching-tier feasibility under the 6 h job cap, 20-job concurrency and the ORCA licence |
| **Max benchmark error** | the largest observed error on a named in-domain validation set, or "not benchmarked". Never a mean alone |
| **Mitigation / notes** | the concrete action that bounds the limitation; Pareto status; "dominated by `<Row ID>`" where applicable |

### 12.4 The Concurrency classes

| Tag | Meaning |
|---|---|
| **C** | **CPU-bound.** Runs on ORCA / CFOUR / Molpro ranks. Nothing to offload |
| **G** | **GPU-bound.** MLFF inference and MD, DF-DFT screening, GPU DFT above the crossover |
| **P** | **Pipelineable.** The row has a cheap GPU companion that can run concurrently and feed it — a geometry seed, a Hessian seed, a candidate cull |
| **S** | **Serial bottleneck.** Non-restartable and non-offloadable; sets the Amdahl floor. Canonical and local coupled cluster, analytic Hessians |

**Standing annotation for every GPU-tagged row on Setup 2:** GPU rows assume MPS is running and that one P-core per GPU worker is reserved. **A GPU tag never implies the row is faster than its CPU equivalent for a single 5–10 atom job; it implies throughput across many such jobs.**

### 12.5 Rules governing every cell

**Rule 1 — the class reflects the weakest link.** A composite row computing coupled-cluster energies on machine-learned geometries is graded on the sampling error, not the electronic-structure error.

**Rule 2 — report a distribution, not a mean.** On S66, MP2C/CBS has an average signed error of **−0.01 kcal/mol** and a largest error of **174 %**; MP2/CBS is 0.69 kcal/mol RMSE with a 40 % maximum; CCSD/CBS 0.70 with 73 %; SCS-CCSD/CBS and SCS-MI-CCSD/CBS both 6 % maximum despite less impressive means ([Řezáč, Riley & Hobza](https://pmc.ncbi.nlm.nih.gov/articles/PMC3152974/)). A cell reporting only a mean would rate MP2C as unbiased; it is unbiased on average over cancelling large errors.

| Method | RMSE (kcal/mol) | MUE | AVG (signed) | **MAX (%)** |
|---|---|---|---|---|
| MP2/TZ | 0.70 | 0.56 | +0.43 | **29** |
| MP2/CBS | 0.69 | 0.45 | −0.44 | **40** |
| **MP2C/CBS** | 0.71 | 0.47 | **−0.01** | **174** |
| SCS-MP2/CBS | 0.87 | 0.74 | +0.73 | **79** |
| MP3/CBS | 0.62 | 0.45 | +0.44 | **64** |
| MP2.5/CBS | 0.16 | 0.12 | 0.00 | **16** |
| CCSD/CBS | 0.70 | 0.62 | +0.62 | **73** |
| SCS-CCSD/CBS | 0.25 | 0.15 | +0.12 | **6** |
| SCS-MI-CCSD/CBS | 0.08 | 0.06 | −0.04 | **6** |

**Rule 3 — S66 and GMTKN55 are out of domain for this system class, and are labelled so.** S66's weakest member is about five times stronger than a typical rare-gas complex. ωB97X-D shows an RMSD of **36.34 pm on rare gases** against 0.58 pm on A21, a sixty-fold degradation on exactly this class ([ωB97X-V paper](https://escholarship.org/content/qt7297t9vf/qt7297t9vf_noSplash_ae27d0ce06218f8fa9b5e5ef1289d1d8.pdf)). In-domain statistics come from the six-system working set of §17. **One benchmark may not certify several different accuracy bands**: where v3 cited the same rare-gas RMSD across five rows spanning ±0.4 % to ±3 %, v4 either sources each tier's maximum from the working set or merges the tiers.

**Rule 4 — no row may claim an accuracy its thresholds cannot support.** Every row producing A, B, C at or below 0.5 % uses the corrected `%geom` block of §4.4, reports the final maximum gradient and the softest force constant, uses at least an augmented triple-zeta basis or `BSSEOptimization.cmp` (§4.7), and carries a core correction if frozen core is used (§4.8). Rows that do not are graded no better than 2 %.

**Rule 5 — monotonicity is not assumed.** Where a later tier is not better than an earlier one for a given observable, the notes read **"dominated by `<Row ID>`"**. §15 collects these into a frontier.

**Rule 6 — circularity is excised.** A computed B_e benchmarked against a semi-experimental r_e that was itself built with a computed ΔB_vib is circular unless the two vibrational corrections come from different levels and the spread is reported; Fortenberry and co-workers attribute part of their agreement to "fortuitous cancellations between basis set incompleteness errors and the B3LYP/jun-cc-pVTZ vibrational corrections". Every semi-experimental comparison names the ΔB_vib level and reports the spread from at least two levels.

**Rule 7 (new) — every number carries a provenance tag.** `[M]` measured or published benchmark, `[D]` derived by arithmetic from a specification or another measurement, `[E]` expert estimate. **No `[D]` or `[E]` value may be the sole support for a hardware exclusion, a routing gate or an accuracy claim.** Where one currently is — the GPU crossover at 50–90 basis functions, the semi-rigid and floppy B₀ bands, the teaching-runner throughput ratio, the DLPNO 15 min/point figure, the g-xTB gradient penalty — **the document says so and marks the value as requiring local measurement.** The five such values are collected in §21.2.

**Rule 8 (new) — every row states whether it delivers B_e or B₀.** They are different specifications (§3.0).

**Rule 9 (new) — Setup-3-only rows quote core-hours at their own execution setup.** v3 derived core-hours as 8 × wall-clock even for rows that cannot run on an eight-core workstation; on a 128-core node the same wall clock is sixteen times the core-hours. Nineteen rows were affected. **Both figures are given where the row can plausibly run on either.**

### 12.6 Machine-checkable invariants

The build script `mm/verify_v4.py` enforces the following on every tier table, and the document does not ship if any fails: every row has a non-empty Row ID, Concurrency tag and Limitations cell; every accuracy cell names a quantity and carries a provenance tag; every table has exactly ten wall-clock tiers; every track is labelled; no accuracy claim in Table 7 exceeds the ±14 % benchmark cap; no Product cell contains prose; no reference is a bare URL, a file-sharing link or a document-mirror stub.

---

## 13. Tables 1–5: search, surface, geometry, averaging, energetics

### 13.1 Table 1 — Conformer and isomer search

**Single track (ORCA and its satellites).** *The CFOUR track cannot fill any row of this table: it has no semi-empirical engine, no global optimiser and no DFT on its public feature list.* The two-stage deduplication protocol is retained from v3 and tightened: **Stage A** at engine defaults (RMSD 0.125 Å, ΔE 0.100 kcal/mol, ΔB 1.0–2.5 %), **Stage B** at the spectroscopic threshold `--bthr 0.001`, because 1 % ≈ 120 MHz at 12 GHz is ten times looser than the experiment resolves (§9B.3).

| Row ID | Tier | Method / Code | Delivers / accuracy | Conc. | Product | Limitations |
|---|---|---|---|---|---|---|
| **T1-10s** | 10 s | Hand-enumerated binding topologies, then `! XTB2 TightOpt` per seed — ORCA/xtb | an ensemble of 3–9 seeds; **no accuracy claim** | C | A | no search at all; a topology missed by hand is invisible to every later row |
| **T1-1min** | 1 min | `! GOAT XTB2 PAL8`, GFN-FF uphill — ORCA | full GOAT ensemble at N ≤ 10 (~100 × N_at optimisations `[M]`) | C | A | semi-empirical ranking only; GFN2-xTB's S22 centre-of-mass maximum error is 32 pm `[M]`, ≈14 % in B |
| **T1-30min** | 30 min | `! GOAT-EXPLORE ExtOpt` + `oet_server aimnet2 -d cuda` — ORCA + AIMNet2 | free-topology enumeration, many seeds in parallel; **Stage A dedup runs here** | **G** | A | float32 noise (~4 × 10⁻⁶ Eh) and interaction-energy errors of **3.5–7.3 kcal/mol on S30L `[M]`** — an enumerator, not a judge |
| **T1-1h** | 1 h | `crest --nci --gfn2 --ewin 12 --nocross --noreftopo`, one run per seed — CREST | an independent second ensemble | C | A | the RMSD bias can still dissociate the complex; hydrogen bonds disrupted after **4.2 ps `[M]`**; a different seed found a conformer **50 kJ/mol lower** |
| **T1-3h** | 3 h | union merge → `crest --screen` → r²SCAN-3c re-optimisation → `crest --cregen` — CREST + ORCA | the production ensemble; **Stage B dedup**; ensemble energies ±1–3 kcal/mol `[E]` | **P** | A | a union is not a proof; report per-engine and overlap counts |
| **T1-12h** | 12 h | `! GOAT r2SCAN-3c`, `%PAL NPROCS 16`, on the 2–3 leading isomers — ORCA | QM-level search around the assigned minima | C | A | **at risk on 16 threads**: ORCA advises 32 cores and warns of "a few days" for an r²SCAN-3c GOAT |
| **T1-1d** | 1 d | `! GOAT-ENTROPY XTB2` + `crest --entropy` on the same seeds — ORCA + CREST | **convergence evidence**: ΔS_conf < 0.1 cal mol⁻¹ K⁻¹ against a converged CREST ensemble entropy | C | A | the two S_conf values are not comparable unless `CONFDEGEN auto` is set — GOAT defaults to g_i = 1 |
| **T1-3d** | 3 d | ωB97X-V/def2-TZVPP re-optimisation of Stage-B survivors; **no new searching** — ORCA | geometries good enough to feed a 0.1 % constant; ΔE ±0.3–1.0 kcal/mol `[E]` | **P** | A | the search is no longer the bottleneck; this row buys ranking, not coverage |
| **T1-1w** | 1 w | fine-tune MACE or AIMNet2 on 100–500 system-specific DFT points, re-run GOAT + CREST — MACE + ORCA | insurance against a missed basin; a reusable checkpoint | **G**, then S | A | fine-tuning is **documented as experimental** and states no required dataset size — **`n.a.`**; buys no improvement in any reported constant |
| **T1-1mo** | 1 mo | exhaustive union: all seeds × both engines × `--v4` / `GOAT-DIVERSITY`, then coupled-cluster re-ranking | the maximum coverage claim available | **G**, then **S** | A | **partly obsolete**: running two GOATs concurrently on two devices from the same start delivers much of this diversity for zero extra wall time (§8A.2). Retain only as completeness insurance |

**Expansion block — Table 1**

| Row ID | Core-h | Input / workflow | State-in | State-out | Frozen-mono | Mem / scratch | Licence | Setup 1? | Max benchmark error | Mitigation / notes |
|---|---|---|---|---|---|---|---|---|---|---|
| T1-10s | 0.022 | `! XTB2 TightOpt`; or `xtb --opt vtight --strict` | seed `.xyz`; `.CHRG`/`.UHF` | `xtbopt.xyz`, `.xtbw` — **R** | relaxed | <0.5 GB / negligible | free | Yes | GFN2-xTB S22 COM max 32 pm | `--strict` mandatory in a chain; hand enumeration is the only completeness argument |
| T1-1min | 0.13 | `%goat maxen 12.0 confdegen auto gfnuphill gfnff end` | seed `.xyz` | ensemble `.xyz`, energy on the comment line — **D** (decompose by seed) | relaxed | 1 GB / 1 GB | ORCA academic | Yes | as above | set `MAXEN` explicitly — the manual gives 6.0 in the table and 12.0 in the text |
| T1-30min | 4 | `! GOAT-EXPLORE ExtOpt TightOpt PAL8` + `%scf TolE 1e-5 end` + `%method ProgExt … end` | seeds `.xyz`; MLFF checkpoint | `.finalensemble.xyz`, `.globalminimum.xyz`, `*.confrot.xyz` — **D** | relaxed | 2 GB host / 1–2 GB VRAM | free model, ORCA academic | Yes (CPU model only) | S30L 7.31 kcal/mol; PLA15 29.9 `[M]` | `! TightOpt` + `TolE 1e-5`; server mode; two venvs (AIMNet2 and UMA conflict); **never report these energies** |
| T1-1h | 8 | `crest seed.xyz --nci --gfn2 --ewin 12 --nocross --noreftopo --T 8` | seed `.xyz` | `crest_conformers.xyz`, `crest_rotamers_*.xyz` — **D** (no general restart) | relaxed | 2 GB / 2 GB | free, LGPL-3.0 | Yes | 50 kJ/mol seed sensitivity `[M]` | ≥3 chemically distinct seeds; `--wscal 0.9` or reduced `kpush` if it still dissociates |
| T1-3h | 24 | `cat *.finalensemble.xyz crest_conformers.xyz > union.xyz`; `crest --screen`; `! r2SCAN-3c TightOpt Freq` ; `crest --cregen … --bthr 0.001` | both ensembles | one deduplicated QM ensemble — **R** for the optimisations, **D** for the searches | relaxed | 2 GB per rank / 3 GB | mixed | Yes (at risk) | r²SCAN-3c ROT34 AMAX 1.5 % `[M]` | re-optimise the union at one common level *before* CREGEN; report GOAT-only, CREST-only, both, union |
| T1-12h | 96 | `! GOAT r2SCAN-3c` + `%PAL NPROCS 16` | leading isomers `.xyz` | refined ensemble + `.gbw` per conformer — **D** | relaxed | 3 GB per rank / 6 GB | ORCA academic | No (>6 h) | — | if the budget will not stretch, run `GOAT-COARSE` with rigid fragments instead |
| T1-1d | 192 | `! GOAT-ENTROPY XTB2`; `crest --entropy` | same seeds | S_conf trajectories — **D** | relaxed | 2 GB / 2 GB | mixed | No | — | set `CONFDEGEN auto`; this row produces *evidence of convergence*, not new structures |
| T1-3d | 576 | `! wB97X-V def2-TZVPP def2/J RIJCOSX TightOpt` + §4.4 `%geom` + `InHess XTB2` | Stage-B ensemble `.xyz`, `.gbw` | per-isomer `.xyz`, `.gbw`, `.opt` — **R** | frozen-iso available (recipe R1/R2) | 2 GB per rank / 4 GB | ORCA academic | No | ωB97X-D rare-gas RMSD 36.34 pm `[M]` | seed each optimisation with an MLFF `.carthess` (§8A.3) for ~2–2.5× fewer cycles `[E]` |
| T1-1w | 1,344 | `mace_run_train --foundation_model=small --multiheads_finetuning=True` | 100–500 DFT points; ensemble | fine-tuned checkpoint — **R**; **archive it, it is reusable state for every later campaign** | — | 8 GB host / 8 GB VRAM | free | No | required dataset size **`n.a.`** | multihead replay "prevents catastrophic forgetting" and is the recommended mode; mark the row experimental |
| T1-1mo | 5,760 @ 8 cores; 21,504 @ 128 | all seeds × both engines × `--v4` / `GOAT-DIVERSITY`; DLPNO re-ranking | everything above | maximal ensemble — **D** | relaxed | 8 GB / 20 GB | mixed | No | — | **dominated for every reported observable**; the concurrent two-device union (§8A.2) delivers most of it free |

### 13.2 Table 2 — Intermolecular potential surfaces, scanning, and the DVR

**Single track (ORCA / PySCF / gpu4pyscf / fitting codes).** *CFOUR can supply reference points but has no surface-fitting or scanning machinery — no row here is a CFOUR row.*

**This table is the most heavily re-tiered in v4.** Active learning is worth 20–100× in points, not 2×: a six-dimensional H₂O–He surface reached **0.3253 cm⁻¹ weighted RMSE from 472 actively selected points**, validated against a **47,945-point** test set, and 0.0710 cm⁻¹ from 613 points `[M]` ([active-learning PES](https://chemrxiv.org/engage/chemrxiv/article-details/675b9e3bf9980725cfe8476a)). Δ-learning multiplies with it — "with as few as **200 CCSD(T) energies**" a permutationally-invariant-polynomial Δ-correction reproduced benchmark CCSD(T) `[M]` ([Δ-learning PES](https://arxiv.org/abs/2011.11601v1)). **Consequently the 1 w row becomes a 12 h row and the 1 mo row becomes 3 d.** One non-obvious caveat is printed on the row it affects: **pure variance-maximisation acquisition plateaus an order of magnitude worse** than error-based or two-set search ([Uteva *et al.*](https://nottingham-repository.worktribe.com/OutputFile/1190028)).

**Two corrections to v3 carried here.** (i) autoPES: flex-autoPES reports **0.03 kcal/mol ≈ 9.1 cm⁻¹ RMSE** on negative-energy water-dimer points from a 4,758-point close-range grid, not the 0.2 kcal/mol ≈ 70 cm⁻¹ v3 quoted — v3 understated it by roughly 8× ([flex-autoPES](https://par.nsf.gov/servlets/purl/10194876)). (ii) PySCF campaigns run as **N concurrent single-thread jobs, not one big threaded job**: PySCF's developers state that for a small system "you may not see much difference of speed" from extra threads and that "a system with several hundred orbitals should be big enough to see the difference" ([PySCF issue 1360](https://github.com/pyscf/pyscf/issues/1360)). Install from PyPI, not conda-forge, where an open bug lets a job asked for 4 threads consume 700 % CPU ([PySCF issue 2533](https://github.com/pyscf/pyscf/issues/2533)).

| Row ID | Tier | Method / Code | Delivers / accuracy | Conc. | Product | Limitations |
|---|---|---|---|---|---|---|
| **T2-10s** | 10 s | semi-empirical relaxed 1-D scan, 20–40 points — xtb/ORCA | well topology only; **no energy claim** | C | A | a tight-binding intermolecular curve is qualitative |
| **T2-1min** | 1 min | dense MLFF scan, 10³ points, to bound the well — MACE/AIMNet2 | the boundaries of the well, not its depth | **G** | A | **the one row where FP32 is legitimate**; use the boundaries, not the energies |
| **T2-30min** | 30 min | composite meta-GGA scan, ~200 points — ORCA r²SCAN-3c | relative energies ±1.5–3 kcal/mol `[E]` | **P** | A | no diffuse functions; the long-range tail is unreliable |
| **T2-1h** | 1 h | 2-D relaxed (R, θ) grid, 960 points, ωB97X-V/def2-TZVPP — ORCA | a 2-D surface, ±1–2 kcal/mol `[E]` | **P** | A | relaxed scans hide hysteresis; run both scan directions and compare |
| **T2-3h** | 3 h | 1-D sinc-DVR on 30–40 points — SciPy | band origins ±20–50 cm⁻¹ `[E]` | C | A | **a 40 × 40 DVR on a GPU is ~70× *slower* than the CPU `[M]`** |
| **T2-12h** | 12 h | **Δ-learning + active learning**: DFT/PIP base + CCSD(T)-F12 correction — ORCA + MOLPIPx | fitted surface, **RMS 3–10 cm⁻¹ `[E]`**, from 2,000 DFT + 300–800 CC points | **P** | A | the fitted surface's error where the acquisition function never looked is **not** bounded by the AL residual |
| **T2-1d** | 1 d | committee-uncertainty active learning with an explicit acquisition function — ORCA + NN committee | 300–800 actively selected points from a 2,000-point pool; surface RMS 5–20 cm⁻¹ `[E]` | **P** | A | **variance-maximisation alone plateaus an order of magnitude worse `[M]`**; a held-out validation grid is mandatory |
| **T2-3d** | 3 d | 3-D rigid-monomer DVR, matrix-free Lanczos, on 500 actively selected points | band origins ±5–20 cm⁻¹ `[E]` | C | A | basis 40 × 24 × 24 = 23,040; **the GPU gives ~1–2× here, i.e. nothing `[M]`** |
| **T2-1w** | 1 w | 6-D rigid-monomer variational treatment on a Δ-learned surface | band origins ±1–5 cm⁻¹ `[E]` | C | A | dense 3-D at 40³ needs 32.8 GB and does not fit 24 GB; matrix-free is mandatory |
| **T2-1mo** | 1 mo | full-dimensional flexible-monomer surface — autoPES/flex-autoPES or PIP | **9.1 cm⁻¹ (site–site, 4,758–11,311 points) `[M]`**, or 0.4–6.3 cm⁻¹ (PIP, 3–5 × 10⁴ points) `[M]` | C | A | **not a workstation tier**: 5 × 10⁴ CC points is 26–43 days here against 12–20 h on 1,024 HPC cores `[D]` |

**Expansion block — Table 2**

| Row ID | Core-h | Input / workflow | State-in | State-out | Frozen-mono | Mem / scratch | Licence | Setup 1? | Max benchmark error | Mitigation / notes |
|---|---|---|---|---|---|---|---|---|---|---|
| T2-10s | 0.022 | `xtb --opt vtight` along R | seed `.xyz` | scan `.xyz` — **D** | frozen-iso | <0.5 GB | free | Yes | — | nothing to parallelise at this size |
| T2-1min | 0.13 | ASE + MLFF, 10³ points, batched on the GPU | seed `.xyz`; model key | dense grid → HDF5 shard — **D** | frozen-iso | 1 GB VRAM | free | Yes (CPU) | S30L 7.31 kcal/mol | the 3090's 35.6 TFLOPS FP32 is the right number **only for this row** |
| T2-30min | 4 | **`parallel -j 16` single-rank, not `PAL16`** | previous point's `.gbw` (`! MORead`) | per-point `.gbw` + HDF5 shard — **D** | frozen-iso | 1 GB per job / 1 GB | ORCA academic | Yes | r²SCAN-3c ROT34 AMAX 1.5 % | v3 omitted the concurrency instruction on this row; it is now explicit |
| T2-1h | 8 | `seq 0 959 \| parallel -j 16 --joblog pes.log --eta ./pes_run.sh {}`; recover with `--resume-failed` | scan `.xyz` / `.gbw` | HDF5 `points/<method>/…` — **D** | frozen-iso | 2 GB per job / 3 GB | ORCA academic | Yes (one 6 h job) | — | neighbour `.gbw` reuse is free: each displaced point is a small perturbation of a converged one |
| T2-3h | 24 | SciPy sinc-DVR; **if JAX is used, `JAX_ENABLE_X64=True` on line 1** | fitted 1-D potential | eigenvalues — **R** (cheap to redo) | frozen-iso | 1 GB | free | Yes | — | JAX "by default enforces single-precision numbers" and the flag "only works on startup"; FP32 noise is ~10⁻¹–10⁰ cm⁻¹ `[D]` |
| T2-12h | 96 | 2,000 DFT points (`parallel -j 16`) + 300–800 CC points; fit with MOLPIPx for a differentiable GPU surface | HDF5 `delta_pairs(low, high)` | fitted Δ-surface + held-out residual — **R** | frozen-iso | 3 GB per job / 8 GB | free / ORCA academic | No | fit residual must be reported in cm⁻¹ | **was v3's 1 w row.** Cite the 200-CCSD(T)-point demonstration and the 208-PIP ethanol correction |
| T2-1d | 192 | committee of two neural surfaces; escalate on the weighted square energy difference | 2,000-point DFT pool; committee checkpoint | escalated points; updated committee — **D** | frozen-iso | 3 GB per job / 8 GB | free | No | H₂O–He: 472 points → 0.3253 cm⁻¹ `[M]` | **budget the held-out grid separately**; a 500-point surface with a 100-point held-out set is not a spectroscopic surface |
| T2-3d | 576 @ 8 cores | 500 DLPNO points at 15 min each = 125 core-h ⇒ **10.4 h wall at `-j 16` `[D]`**, or 29 min on 1,024 HPC cores | HDF5 geometry list; neighbour `.gbw` | 500 CC energies in HDF5; DVR eigenvalues — **D** | frozen-iso | 3 GB per job / 5–20 GB per point | ORCA academic | No | Inductiva RTX 3090 Lanczos: 0.67× at dim 10⁴, 4.2× at 10⁵ `[M]` | v3's "125 h on one rank" read as the row's cost; it is the *serial* number. **Correct optimisations are CUDA-graph capture of the fixed-shape matvec and block Lanczos, not a faster card** |
| T2-1w | 1,344 @ 8 cores | matrix-free Lanczos / block Davidson on the Δ-learned 6-D surface | Δ-surface | VRT manifold — **R** | frozen-iso | 8 GB / 20 GB | free | No | — | one matvec is ~14 µs of FP64 arithmetic against a 5–15 µs launch overhead `[D]` — this is why the GPU underdelivers here |
| T2-1mo | 5,760 @ 8 cores; 92,160 @ 128 | autoPES/flex-autoPES (needs **ORCA 3.0.1, Dalton 2.0, SAPT2016**), or PIP via MSA-2.0 | HDF5 full point set | analytic surface file, 500 MB – 5 GB | relaxed (full-dimensional) | 16 GB / 200 GB | autoPES licence **`n.a.`** | No | flex-autoPES (H₂O)₂ 9.1 cm⁻¹; stationary-point RMSE 7.6 cm⁻¹ vs CCSD(T) 1.8 `[M]` | **HPC only.** autoPES's grid is `NFP × 6 × 100/(100 − TEST PCT)`, proportional to the number of fit parameters, not to atom count; out of the box it is serial (`MAX SIM PT` default 1) |

### 13.3 Table 3 — Equilibrium geometry and B_e

**Every row here reports B_e, which is not an observable, and says so.** Converting to B₀ is Table 4's job, and ΔB_vib is 0.1–0.7 % of B_e — larger than the 0.1 % target — so **no row of this table alone can meet a B₀ specification, however good its equilibrium structure** (§3.0).

#### Table 3-O — ORCA track

| Row ID | Tier | Method / Code | Delivers / accuracy | Conc. | Product | Limitations |
|---|---|---|---|---|---|---|
| **T3O-10s** | 10 s | GFN2-xTB equilibrium structure — xtb/ORCA | **B_e ±3–15 % `[M]`** | C | A | intermonomer separations from tight binding are systematically off by 14 pm MAD on S22 |
| **T3O-1min** | 1 min | **recipe R1**: frozen high-level or experimental monomers + r²SCAN-3c intermolecular optimisation — ORCA | **B_e ±1–3 %; A to <0.2 % `[D]`** | C | A | no diffuse functions, so the intermolecular distance is the dominant error |
| **T3O-30min** | 30 min | ωB97X-V/def2-TZVPP full optimisation — ORCA | **B_e ±0.5–3 % `[E]`** | **P** | A | **BSSE contracts the complex: +4.1 pm ≈ 2.8 % in B at plain cc-pVTZ `[M]`** |
| **T3O-1h** | 1 h | ωB97X-V/jun-cc-pVTZ, diffuse functions in the basis — ORCA | **B_e ±0.5–2 % `[E]`** | **P** | A | diffuse sets on hydrogen cause near-linear dependence; the failure is loud, not silent |
| **T3O-3h** | 3 h | **recipe R2**: frozen CCSD(T)-class monomers + ωB97M-V/def2-QZVPP intermolecular + 3-leg counterpoise + VPT2 — ORCA | **B_e ±0.4–1.5 %; ±0.3–0.5 % semi-rigid; A to <0.2 % `[E]`**, plus ΔB_vib | **P** | A | quadruple zeta is the DFT complete-basis limit for non-covalent interactions — this is the end of the DFT road |
| **T3O-12h** | 12 h | **recipe R4: junChS (jun-cc-pVnZ CBS+CV composite)** — ORCA compound scripts | **B_e, MAE 0.13 % for ≤16 atoms `[M]`** | C | A | delivers **B_e only**; ChS is not BSSE-free; parameter-wise addition assumes near-transferable intramolecular coordinates |
| **T3O-1d** | 1 d | MPQC CCSD(T)-F12 numerical-gradient optimisation — ORCA | **B_e ±0.3–0.8 % floppy, ±0.15–0.5 % semi-rigid `[D]`** | **S** | A | **no analytic DLPNO gradient**: each gradient is 6N = 60 single points, 25 cycles ≈ 1,500 points ≈ 375 core-h. **Dominated by T3O-12h** |
| **T3O-3d** | 3 d | canonical CCSD(T)/cc-pVTZ-F12 (paired with CABS: OptRI [Yousaf & Peterson, J. Chem. Phys. 129, 184108 (2008)], JKFIT, MP2FIT), AUTOCI analytic gradients — ORCA (Setup 3) | **B_e ±0.3–1 % `[M]`, with a warning** | **S** | A | **fc-CCSD(T)/cc-pVQZ gave −6.56 % in A_e for a C₅H₂ isomer `[M]`**; frozen core is a −0.81 % bias in B_e |
| **T3O-1w** | 1 w | composite CCSD(T)/CBS + Δcore + ΔT + ΔQ (the ChS/HEAT family) — ORCA compound scripts or CFOUR | **B_e ±0.04 % semi-rigid closed-shell `[M]`; 1–2 % floppy `[D]`** | **S** | A | the 0.04 % figure is a semi-rigid closed-shell number and does not transfer to a floppy complex |
| **T3O-1mo** | 1 mo | explicitly correlated reference single points; F12 geometry **requires Molpro** | **B_e — no better than T3O-1w for the reported constants** | **S** | A | "F12 gradients are not available" is true in ORCA 6.1 and false in general. **Dominated by T3O-1w**; retained for method benchmarking |

**Expansion block — Table 3-O**

| Row ID | Core-h | Input / workflow | State-in | State-out | Frozen-mono | Mem / scratch | Licence | Setup 1? | Max benchmark error | Mitigation / notes |
|---|---|---|---|---|---|---|---|---|---|---|
| T3O-10s | 0.022 | `! XTB2 TightOpt`; or `xtb --opt vtight --strict` | seed `.xyz` | `xtbopt.xyz`, `.xtbw` — **R** | relaxed | <0.5 GB | free | Yes | pyrrole–Ne: B3LYP/6-311++G(d,p) 46.34 %; MP2/aVTZ 25.66 % `[M]` | escalate before quoting anything; use for search seeding and isotopologue bookkeeping |
| T3O-1min | 0.13 | `! r2SCAN-3c TightSCF DefGrid3` + §4.4 `%geom` + `%geom Constraints {…} end`. **No `D4`, no `gCP`** — both are inside the composite | `xtbopt.xyz`; `InHess XTB2` | `.xyz`, `.gbw`, `.opt` — **R** | **frozen-iso** | 1 GB per rank / 1 GB | ORCA academic | Yes | r²SCAN-3c ROT34 AMAX 1.5 % `[M]` | **strictly dominates the unconstrained r²SCAN-3c row: same cost, same B, A improved by ~1.5 pp.** Pareto: frontier |
| T3O-30min | 4 | `! wB97X-V def2-TZVPP def2/J RIJCOSX TightSCF DefGrid3` + §4.4 `%geom` + `InHess XTB2` | `s(1min).xyz` + `.gbw` (`! MORead`) + `.opt` | `.xyz`, `.gbw`, `.opt`; exported `.bas` for the CP triplet — **R** | relaxed | 2 GB per rank / 3 GB | ORCA academic | Yes | ωB97X-D rare-gas RMSD 36.34 pm `[M]` | do not counterpoise below triple zeta; at triple zeta and above use `BSSEOptimization.cmp` and report both structures |
| T3O-1h | 8 | `! wB97X-V jun-cc-pVTZ def2/J RIJCOSX TightSCF DefGrid3`; raise `%scf SThresh` if the overlap spectrum is ill-conditioned | TZ `.gbw` cascade | `.xyz`, `.gbw`, `.opt` — **R** | relaxed | 2 GB per rank / 4 GB | ORCA academic | Yes | as above | use calendar sets (jun-, may-) rather than aug-; jun-TZ carries ~27 % fewer functions. **Removed from v3's dominated list**: cc-pVTZ → cc-pVDZ-F12 (paired with CABS) cuts the CP discrepancy from 4.1 to 1.1 pm |
| T3O-3h | 24 | `! wB97M-V def2-QZVPP def2/J RIJCOSX TightSCF DefGrid3` + §4.4 `%geom` + constraints; then 3 × `! DLPNO-CCSD(T1) TightPNO cc-pVDZ-F12 (paired with CABS)`; then `! wB97X-V def2-TZVPP Freq VPT2 VeryTightSCF` | `s(30min).xyz` + `.gbw` (TZ→QZ projection) | `.xyz`, `.gbw`, `.opt`, `.hess` — the geometry that feeds Tables 4 and 5 — **R** | **frozen-iso** | 3 GB per rank / 6 GB | ORCA academic | Yes (at risk) | ωB97X-V A21 RMSD 0.58 pm; rare gases 7.91 pm `[M]` | **the best de novo accuracy-per-core-hour row in the document.** Report the residual gradient on the frozen coordinates |
| T3O-12h | 96 | R[fc-CCSD(T)/jun-cc-pVTZ] + ΔR[MP2/CBS(T→Q), n⁻³] + ΔR[MP2/CV, cc-pwCVTZ], parameter-wise | `s(3h).xyz` + `.gbw` | composite geometry; component energies — **D** (no MDCI restart) | frozen-iso or relaxed | 3 GB per rank / 8 GB | ORCA academic | No | ChS MAE 0.13 % in B_e, ≤16 atoms `[M]` | anchored on pyridine–H₂O at 7 h 47 min (cc-pVTZ) / 16 h 12 min (jun-cc-pVTZ) on 64 CPUs `[M]`. **Replaces v3's double-hybrid geometry row**, which had no vdW benchmark |
| T3O-1d | 192 | `! MPQC CCSD(T)-F12 TightPNO cc-pVDZ-F12 (paired with CABS) NumGrad Opt TightSCF` + §4.4 `%geom` | `s(3h).xyz` + `.gbw`; per-displacement neighbour `.gbw` | `.xyz`; 60 displacement single points per cycle — **D**, run them as independent jobs | relaxed | 3 GB per rank / 5–20 GB per point | ORCA academic | No | geometry maximum **`n.a.`** | **Dominated by T3O-12h**: 375 core-h for a floppy geometry no better than T3O-3h, against a 0.13 %-MAE composite at 6–20 h |
| T3O-3d | 576 @ 8 cores; 9,216 @ 128 | `! AUTOCI-CCSD(T) cc-pVTZ-F12 (paired with CABS: OptRI [Yousaf & Peterson, J. Chem. Phys. 129, 184108 (2008)], JKFIT, MP2FIT) Opt TightSCF`; AO-direct mandatory | `s(3h).xyz` + `.gbw` | `.xyz`, `.gbw` — **D** in ORCA, **R** in CFOUR | relaxed | 8 GB per rank / **99 GB four-external + 25 GB three-external if integral-conventional** | ORCA academic | No | −6.56 % in A_e `[M]` | ~596 basis functions sits at ORCA's 500–600 practical ceiling. **`fc` vs `ae` is not a detail: the core-correlation effect on A_e was 1.0 pp with inconsistent sign.** Prefer CFOUR (T3C-3d) or junChS-F12 (Molpro) |
| T3O-1w | 1,344 @ 8; 21,504 @ 128 | CCSD(T)/cc-pV∞Z + Δcore + ΔT + ΔQ, counterpoise-bracketed | CFOUR `JOBARC`; component energies | composite geometry; `FCMFINAL` — **R** in CFOUR | relaxed | 16 GB / 200 GB | CFOUR academic registration | No | composite MAE 0.04 %, SD 0.07 % — **semi-rigid closed-shell** `[M]` | **this row is the ChS/HEAT-class composite; the family names are given so a reader can find the literature.** For a floppy complex quote 1–2 % and route to Product B |
| T3O-1mo | 5,760 @ 8; 92,160 @ 128 | Molpro DF-CCSD(T)-F12 optimisation, or `! CCSD(T)-F12D/RI cc-pVTZ-F12` single points in ORCA | Molpro or ORCA geometry | reference geometry — **D** | relaxed | 32 GB / 500 GB | **Molpro commercial**; ORCA academic | No | as T3O-1w | **junChS-F12 (recipe R3) dominates this row**: "one order of magnitude faster than the CBS+CV counterparts" with SE100 MUE(r) 0.0011 Å `[M]` |

#### Table 3-C — CFOUR track

| Row ID | Tier | Method / Code | Delivers / accuracy | Conc. | Product | Limitations |
|---|---|---|---|---|---|---|
| **T3C-10s** | 10 s | — | — | — | — | **This track cannot fill this tier.** CFOUR has no semi-empirical or force-field engine and no DFT on its public feature list. Use `T3O-10s` |
| **T3C-1min** | 1 min | — | — | — | — | **Cannot fill.** No global optimiser; the cheapest meaningful CFOUR job is an SCF or MP2 single point. Use `T3O-1min` |
| **T3C-30min** | 30 min | MP2/cc-pVDZ optimisation, `COORDINATES=INTERNAL` with `*` on the variables | a structural sanity check; **no B_e claim** | C | A | **MP2 is not worth running in parallel in CFOUR** |
| **T3C-1h** | 1 h | CCSD(T)/cc-pVTZ single point with `PROPS=FIRST_ORDER` | dipole components, EFG → χ; no geometry | C | A | requires the EFG → nuclear-quadrupole-coupling conversion by hand |
| **T3C-3h** | 3 h | CCSD(T)/cc-pVTZ geometry optimisation, analytic gradients | **B_e, MAE 0.90 % `[M]`** at this basis | C | A | optimisation only in internal or XYZ2INT coordinates — **Cartesian input silently disables it** |
| **T3C-12h** | 12 h | CCSD(T)/cc-pVQZ optimisation | **B_e, MAE 0.43 % `[M]`** | C | A | frozen core at quadruple zeta is a **−0.81 % bias `[M]`**; add a core correction |
| **T3C-1d** | 1 d | all-electron CCSD(T)/cc-pCVQZ optimisation | **B_e, mean −0.037 %, MAE 0.164 % `[M]`** | C | A | core-valence quadruple zeta is expensive; the cheap alternative is fc/CBS(Q,5) + core/cc-pCVTZ at 0.107 % `[M]` |
| **T3C-3d** | 3 d | ChS composite in CFOUR: fc-CCSD(T)/TZ + ΔMP2/CBS + ΔMP2/CV | **B_e, MAE 0.13 % `[M]`** | C | A | plain ChS has no diffuse functions — **use jun-cc-pVnZ for a weak complex** |
| **T3C-1w** | 1 w | CBS+CV+fT+fQ composite | **B_e, MAE 0.04 % semi-rigid closed-shell `[M]`** | **S** | A | a semi-rigid figure; a floppy complex is 1–2 %. The ΔQ step is the cost driver |
| **T3C-1mo** | 1 mo | the above plus relativistic (`RELATIVISTIC=DPT2` or `X2C1E`) and the diagonal Born–Oppenheimer correction (`DBOC=ON`) | **B_e with small-correction closure** | **S** | A | DBOC is limited to HF/MP1/MP2/CCSD for RHF/UHF; **`ANHARM=FULLQUARTIC` is not in the public release** |

**Expansion block — Table 3-C**

| Row ID | Core-h | Input / workflow | State-in | State-out | Frozen-mono | Mem / scratch | Licence | Setup 1? | Max benchmark error | Mitigation / notes |
|---|---|---|---|---|---|---|---|---|---|---|
| T3C-10s / T3C-1min | — | — | — | — | — | — | — | No | — | **track gap, stated rather than hidden.** Escalate to the ORCA track |
| T3C-30min | 4 | `*CFOUR(CALC=MP2,BASIS=PVDZ,COORDINATES=INTERNAL, MEMORY_SIZE=32,MEM_UNIT=GB)` with `*` after the variables | `ZMAT` + `GENBAS` | `JOBARC`, `JAINDX`, `OPTARC` — **R** | relaxed | global 32 GB / 20 GB | CFOUR academic (wet signature) | **No** — licence does not scale to a class | — | `MEMORY_SIZE` is a **global** allocation defaulting to ≈762 MB; raise it or the job thrashes |
| T3C-1h | 8 | add `PROPS=FIRST_ORDER` | `JOBARC` | first-order properties — **R** | frozen-inc | 32 GB / 30 GB | CFOUR academic | No | χ conversion factor 234.96474 documented | ORCA cannot do this at CCSD(T): "an unrelaxed density for CCSD(T) is NOT available" |
| T3C-3h | 24 | `*` after the variables; `GEO_CONV=5`, RMS gradient 1e-5 E_h/bohr, `GEO_MAXCYC=50` | `ZMAT`, `JOBARC` | optimised `ZMAT`, `JOBARC` — **R** | relaxed or frozen-inc | 32 GB / 40 GB | CFOUR academic | No | fc-CCSD(T)/VTZ MAE 0.90 % in B_e `[M]` | **three-character variable names, single-space fields, no 0°/180° angles — dummy atoms for linear fragments** |
| T3C-12h | 96 | `BASIS=PVQZ` | `JOBARC` | `JOBARC` — **R** | relaxed | 32 GB / 80 GB | CFOUR academic | No | fc-CCSD(T)/VQZ MAE 0.812 %, max 2.701 % `[M]` | `ABCDTYPE=AOBASIS` + `CC_PROG=ECC` required for parallel execution |
| T3C-1d | 192 | `BASIS=PCVQZ`, `FROZEN_CORE=OFF` | `JOBARC` | `JOBARC` — **R** | relaxed | 32 GB / 150 GB | CFOUR academic | No | ae-CCSD(T)/cc-pCVQZ MAE 0.164 %, max 0.874 % `[M]` | the cheap alternative is fc/CBS(Q,5) + core/cc-pCVTZ — CBS extrapolation makes a small core-valence basis suffice |
| T3C-3d | 576 @ 8; 9,216 @ 128 | three legs, added parameter-wise | `JOBARC` per leg | composite geometry — **R** | frozen-inc | 32 GB / 200 GB | CFOUR academic | No | ChS MAE 0.13 % `[M]` | **report the extrapolation formula: n⁻³ vs n⁻⁵ is worth 3–5 mÅ, i.e. 0.20–0.34 % in B `[D]`** |
| T3C-1w | 1,344 @ 8; 21,504 @ 128 | add ΔT and ΔQ increments | `JOBARC`, `MOINTS`, `MOABCD` | composite geometry — **R** (CFOUR restarts CC) | frozen-inc | 64 GB / 400 GB | CFOUR academic | No | 0.04 % semi-rigid closed-shell `[M]` | **this is the tier CFOUR's CC restart makes reachable under a 48 h queue and ORCA does not** |
| T3C-1mo | 5,760 @ 8; 92,160 @ 128 | stacked `RELATIVISTIC=`, `DBOC=ON` | `JOBARC` | full small-correction set — **R** | frozen-inc | 64 GB / 500 GB | CFOUR academic | No | — | public CFOUR is **v2.1, July 2019** and lags the developers' version; GUINEA, FULLQUARTIC, CASSCF, Raman and MRCC-driven runs are unavailable |

### 13.4 Table 4 — Vibrational averaging: from B_e to B₀

**This is where the accuracy actually comes from, and it is the highest accuracy-per-core-hour item in the document.**

**The salvaged dynamical protocol, and the walk-back.** Classical molecular dynamics at jet temperature does not deliver absolute constants: at 5 K, k_BT = 3.48 cm⁻¹, so classical equipartition puts 1.74 cm⁻¹ into each mode against zero-point energies of 15 cm⁻¹ for a 30 cm⁻¹ intermolecular bend and 1500 cm⁻¹ for a 3000 cm⁻¹ X–H stretch. **A classical trajectory at 5 K is a frozen structure rattling in the harmonic bottom of the well and returns essentially B_e.** The zero-point elongation it misses is large: ΔR₀ᵉ = 0.361 Å for CH₃⁺–He, 0.155 Å for CH₃⁺–Ne, 0.038 Å for CH₃⁺–Ar `[M]` ([methyl-cation rare-gas complexes](https://arxiv.org/pdf/2009.05443.pdf)) — for a pseudo-diatomic that puts B₀ roughly 30 % below B_e in the He case. And at 5 K the classical barrier-crossing rate between equivalent minima is e^(−28.7) ≈ 3 × 10⁻¹³, so the trajectory is trapped in one well while the true ground state is delocalised.

**v3 concluded from this that classical averaging should be deleted and its output marked `n.a.` That was an over-correction and it is walked back.** Classical MD at the temperature the sampling is actually run at remains a legitimate **diagnostic**, and it is reinstated as one, with its output restricted to exactly three things: **(a) a basin count**, **(b) a bounded upper estimate of ΔB_vib**, and **(c) an explicit "zero-point energy not included" flag.** It may not emit an absolute constant. Rigid-monomer path-integral MD at 50 K, the salvaged protocol of v3, is retained unchanged for the cases where a quantum average is genuinely needed.

#### Table 4-O — ORCA track

| Row ID | Tier | Method / Code | Delivers / accuracy | Conc. | Product | Limitations |
|---|---|---|---|---|---|---|
| **T4O-10s** | 10 s | inertial defect and planar moments from any geometry | Δ = I_c − I_a − I_b, P_aa/P_bb/P_cc; **sign must be right** | C | A | qualitative; three subtractions on A, B, C |
| **T4O-1min** | 1 min | **recipe R6: semi-experimental anchoring** — scale the geometry to the measured parent, substitute masses — Kisiel suite | **B₀ ±0.03–0.1 % `[M]`** | C | **B** | **requires a measured parent or close analogue.** Not available for Product A |
| **T4O-30min** | 30 min | analytic DFT Hessian on the T3O-3h geometry; harmonic α_r | **ΔB_vib, ±0.1 % of B₀ at 20 % force-constant error `[D]`**; quartic distortion free | **S** | A | **analytic frequencies are not restartable** — must fit one wall-clock window |
| **T4O-1h** | 1 h | DFT VPT2 anharmonic force field, 49 analytic Hessians at N = 10 | **B₀ = B_e + ΔB_vib, ±0.3–0.5 % semi-rigid `[D]`**; α_r, quartic distortion, Watson parameters | **S** | A | **`!VPT2` accepts only analytic-Hessian methods — no double hybrids, no RI-JK, and no linear molecules** |
| **T4O-3h** | 3 h | VPT2 on three conformers; Boltzmann-averaged constants + Pickett export | B₀ per conformer plus vibrational satellites B_v | **S** | A | Boltzmann weights inherit the energy error; report the weights and their sensitivity |
| **T4O-12h** | 12 h | **isotopologue campaign from one force field** — `orca_vib` re-analysis | B₀ for 6–15 isotopologues at **zero additional electronic-structure cost — a 6–15× saving `[D]`** | C | **C** | the force field must be the *same*; re-analysis cannot fix a wrong geometry |
| **T4O-1d** | 1 d | rigid-monomer path-integral MD at 50 K on an MLFF surface, ORCA `%md` | **ΔB_vib with a factor-2 uncertainty**; basin count | **G** with a CPU driver | A | P > 18 beads at 50 K for ω_max = 600 cm⁻¹ (use 40 with the 2.2 safety factor); **full-dimensional PIMD at jet temperature is deleted** — P > 863 at 5 K |
| **T4O-3d** | 3 d | ⟨μ_αα⟩ from the Table 2 3-D surface — vibrational averaging of the inverse inertia tensor | **B₀ ±5–20 cm⁻¹-equivalent for the intermolecular modes `[E]`** | C | A | averages the **inverse** inertia tensor, not the constants — averaging B directly is wrong |
| **T4O-1w** | 1 w | diffusion Monte Carlo on the Δ-learned surface | ⟨μ_αα⟩, B₀ for the true ground state | **P** | A | walker ensembles are embarrassingly parallel and checkpointable; the surface's error is inherited |
| **T4O-1mo** | 1 mo | full VRT manifold and tunnelling splittings from the 6-D surface | band origins; **tunnelling splittings as estimates only** | C | A | splittings span 6 MHz to 279,650 MHz within one molecule — **factor of 3 at best** |

**Expansion block — Table 4-O**

| Row ID | Core-h | Input / workflow | State-in | State-out | Frozen-mono | Mem / scratch | Licence | Setup 1? | Max benchmark error | Mitigation / notes |
|---|---|---|---|---|---|---|---|---|---|---|
| T4O-10s | 0.022 | three subtractions | any `.xyz` | Δ, P_aa/P_bb/P_cc — **R** | — | negligible | free | Yes | — | **the sign of Δ must be right, always**; it probes the out-of-plane force field directly |
| T4O-1min | 0.13 | scale to the measured A, B, C, then substitute masses; **use r_e^SE, not r₀** | experimental parent constants + any `.xyz` | scaled geometry — **R** | frozen-iso | negligible | free | Yes | 0.03–0.06 % `[M]` | **the best cell in the document.** Add recipe R5 (per-bond regression) as a named sub-recipe, and **never apply the template to a B3LYP geometry — it nearly doubles the deviation** |
| T4O-30min | 4 | `! wB97M-V def2-QZVPP … Freq TightSCF DefGrid3 MORead` at the **identical level and geometry** | `s(3h).xyz` + `.gbw` | `.hess` (→ all isotopologues), quartic constants — **D** | inherits | 3 GB per rank / 6 GB | ORCA academic | Yes (at risk) | — | if it will not fit one window, switch to `NumFreq`, which **is** restartable via `%freq Restart true` |
| T4O-1h | 8 | `! B3LYP D4 def2-TZVPP VPT2` + `%pal nprocs 16 nprocs_group 2 end` + `%method Z_Tol 1e-14 end` + `%output Pickettname "x.txt" end` | same `.xyz` + `.gbw`; 49 analytic Hessians | `.hess` per displacement; α_r; a Pickett template — **D** | frozen-iso permitted | 3 GB per rank / 10 GB | ORCA academic | No | — | 49 = 6N − 11 at N = 10. **Skip for linear complexes.** Substituted hybrid force fields are permitted on the semi-rigid manifold only, and **any mode below ~100 cm⁻¹ is excluded from the hybrid treatment** |
| T4O-3h | 24 | three conformer `.xyz` + `.gbw` | Table 3 geometries | three `.hess`; B_v satellites — **D** | frozen-iso | 3 GB per rank / 12 GB | ORCA academic | No | — | satellites are the second-strongest features in a jet spectrum of a complex |
| T4O-12h | 96 | `for iso in …; do cp s5.hess iso_$iso.hess; orca_vib iso_$iso.hess; done`; or CFOUR `ISOMASS` + `xjoda` | one `.hess` or `JOBARC` | B₀ per isotopologue — **R** | inherits | 1 GB | ORCA academic | Yes | isotopologue shifts to 0.02–0.1 % `[M]` | **the second-highest-value reuse in the document.** Note that CFOUR's `ANHARM=VIBROT` cubic constants are **not** sufficient for isotopologues of lower symmetry — use `ANHARM=VPT2` |
| T4O-1d | 192 | `%md Restart IfExists end`; MLFF forces on the GPU, ORCA driver on one P-core | MLFF checkpoint; `.xyz`; `mdrestart` | `.mdrestart` every step; trajectory — **R** (chain 6 h jobs) | frozen-iso (rigid monomers) | 4 GB host / 2 GB VRAM | free | No | — | **classical MD is a diagnostic only**: emit basin count, a bounded ΔB_vib, and a "zero-point energy not included" flag |
| T4O-3d | 576 | Lanczos on the registered grid; average μ_αα, then invert | Table 2 3 d surface (HDF5 `dvr_grid`) | ⟨μ_αα⟩, B₀ — **R** | frozen-iso | 8 GB | free | No | — | average the **inverse** inertia tensor; the DVR is cheap to redo |
| T4O-1w | 1,344 | walker ensemble with periodic checkpoints | Table 2 12 h Δ-surface | walker checkpoint; ⟨μ_αα⟩ — **R** | frozen-iso | 8 GB | free | No | — | embarrassingly parallel; the natural Setup-3 row |
| T4O-1mo | 5,760 @ 8; 92,160 @ 128 | variational solution on the 6-D surface | Table 2 1 mo surface | VRT manifold, splittings — **D** | relaxed | 32 GB / 200 GB | free | No | splittings span 6 MHz – 279,650 MHz `[M]` | report the barrier, the reduced mass and the path, and flag the splitting as an estimate |

#### Table 4-C — CFOUR track

| Row ID | Tier | Method / Code | Delivers / accuracy | Conc. | Product | Limitations |
|---|---|---|---|---|---|---|
| **T4C-10s** | 10 s | — | — | — | — | **Cannot fill.** No cheap engine. Use `T4O-10s` |
| **T4C-1min** | 1 min | — | — | — | — | **Cannot fill.** No template or scaling machinery is documented. Use `T4O-1min` |
| **T4C-30min** | 30 min | SCF or MP2 harmonic frequencies, `VIB=EXACT` | ω at a low level; a sanity check | C | A | MP2 is not parallelised in CFOUR |
| **T4C-1h** | 1 h | CCSD(T) **analytic harmonic Hessian**, `VIB=EXACT` + `ABCDTYPE=AOBASIS`, `CC_PROG=ECC` | ω, IR intensities; produces `FCM`, `FCMINT`, `DIPDER` | C | A | no analytic second derivatives for ROHF-based coupled cluster |
| **T4C-3h** | 3 h | `ANHARM=VIBROT` — vibration–rotation interaction constants only | **α constants → B₀ from B_e** | C | A | only φ_nij with n totally symmetric; **no saving in C₁, which is the usual symmetry of a floppy complex**, and insufficient for lower-symmetry isotopologues |
| **T4C-12h** | 12 h | full `ANHARM=VPT2`, cc-pVTZ, 49 analytic Hessians at N = 10 | anharmonic ν, α, **quartic and sextic** distortion; oxirane agreement ~0.1 % rotational, 2–3 % quartic, 3–4 % sextic `[M]` | C | A | **GUINEA (deperturbed VPT2, anharmonic intensities) is not in the public release** |
| **T4C-1d** | 1 d | the above with `PROPS=FIRST_ORDER` and vibrationally averaged properties | ⟨A⟩ = A_e + Σ_r (∂A/∂Q_r)⟨Q_r⟩ + …; χ tensors | C | A | vibrational averaging of properties inherits the force field's error |
| **T4C-3d** | 3 d | `ANHARM=VPT2` at cc-pCVTZ or ANO1 with core correlation, queue-split | a semi-experimental-quality force field | **S** | A | job splitting is script-driven, not built in; `FD_PROJECT=OFF` in the parallel recipe |
| **T4C-1w** | 1 w | isotopologue force fields from the same field via `%isotopes` | B₀ for 6–15 isotopologues at CCSD(T) quality | C | **C** | `ANHARM=VIBROT` is insufficient here — the full `ANHARM=VPT2` field is required |
| **T4C-1mo** | 1 mo | the above plus `SPINROT=ON`, `DBOC=ON`, `RELATIVISTIC=DPT2` | the full composite spectroscopic-constant set for an isotopic campaign | **S** | A | DBOC limited to HF/MP1/MP2/CCSD for RHF/UHF; `ANHARM=FULLQUARTIC` unavailable publicly |

**Expansion block — Table 4-C**

| Row ID | Core-h | Input / workflow | State-in | State-out | Frozen-mono | Mem / scratch | Licence | Setup 1? | Max benchmark error | Mitigation / notes |
|---|---|---|---|---|---|---|---|---|---|---|
| T4C-10s / T4C-1min | — | — | — | — | — | — | — | No | — | **track gap, stated.** Use the ORCA track |
| T4C-30min | 4 | `VIB=EXACT`, `CALC=MP2` | `ZMAT` + `JOBARC` | `FCMFINAL` — **R** (finite-difference restart needs only `JOBARC` + `JAINDX`) | frozen-inc | global 32 GB / 20 GB | CFOUR academic | No | — | `VIB=ANALYTIC` is "in the current release not available"; the manual warns "Please do not use VIB=2!" |
| T4C-1h | 8 | `VIB=EXACT` + `ABCDTYPE=AOBASIS` + `CC_PROG=ECC` | converged `ZMAT`, `JOBARC` | `FCM`, `FCMINT`, `DIPDER`, `FCMFINAL` — **R** | frozen-inc | 32 GB / 60 GB | CFOUR academic | No | — | **ORCA cannot do this at any tier**: analytic Hessians for SCF only |
| T4C-3h | 24 | `VIB=EXACT, ANHARM=VIBROT` | `JOBARC`, `FCMFINAL` | α_r → B₀ — **R** | frozen-inc | 32 GB / 80 GB | CFOUR academic | No | oxirane rotational constants ~0.1 % `[M]` | **do not budget a symmetry discount for a C₁ complex** |
| T4C-12h | 96 | `VIB=EXACT, ANHARM=VPT2, ANH_STEPSIZ=50000, FD_PROJECT=ON` | `JOBARC` | full cubic + semidiagonal quartic field — **R** | frozen-inc | 32 GB / 150 GB | CFOUR academic | No | quartic 2–3 %, sextic 3–4 % `[M]` | 49 analytic Hessians against **176,400 ORCA/DLPNO single points** for the same object — a ratio of 36N² `[D]` |
| T4C-1d | 192 | add `PROPS=FIRST_ORDER` | `JOBARC` | ⟨A⟩, χ, dipole components — **R** | frozen-inc | 32 GB / 150 GB | CFOUR academic | No | — | conversion χ(kHz) = EFG(a.u.) × Q(mbarn) × 234.96474 |
| T4C-3d | 576 @ 8; 9,216 @ 128 | add `FREQ_ALGORITHM=PARALLEL, ANH_ALGORITHM=PARALLEL, FD_PROJECT=OFF`; process with `xjoda`, `xsymcor`, `xja2fja`, `xcubic` | `JOBARC` | queue-split force field — **R**; `FD_IRREP` is the decomposition axis | frozen-inc | 32 GB / 250 GB | CFOUR academic | No | — | **`FD_IRREP` gives no decomposition in C₁** — decompose by displacement instead |
| T4C-1w | 1,344 @ 8; 21,504 @ 128 | `%isotopes` per isotopologue, re-running `xjoda` against a saved `JOBARC` | one `JOBARC` + one force field | B₀ per isotopologue — **R** | frozen-inc | 32 GB / 250 GB | CFOUR academic | No | isotopologue shifts 0.02–0.1 % `[M]` | **one force field, many isotopologues, no new electronic structure** |
| T4C-1mo | 5,760 @ 8; 92,160 @ 128 | stacked `SPINROT=ON`, `DBOC=ON`, `RELATIVISTIC=` | `JOBARC` | the full constant set — **R** | frozen-inc | 64 GB / 500 GB | CFOUR academic | No | spin–rotation ~3 % for D₂O `[M]` | **CFOUR has no documented SPCAT export — the constants must be transcribed by hand or by a script (`n.a.`)** |

### 13.5 Table 5 — Interaction and binding energies

**Single track (ORCA).** *CFOUR can supply reference energies but has no DLPNO or F12, so the cost-effective rows here are all ORCA rows.* **D₀ is not an assignment observable** — it does not appear in the rotational Hamiltonian — so this table is post-assignment validation and conformer ranking, never a route to a constant.

| Row ID | Tier | Method / Code | Delivers / accuracy | Conc. | Product | Limitations |
|---|---|---|---|---|---|---|
| **T5-10s** | 10 s | GFN2-xTB interaction energy | **ΔE ±2–5 kcal/mol `[E]`** | C | A | screening only |
| **T5-1min** | 1 min | MLFF interaction energy — MACE/AIMNet2 | **ΔE, but errors of 3.5–7.3 kcal/mol on S30L `[M]`** | **G** | A | **larger than typical isomer separations — a filter with a wide window, never a ranking** (guard G4) |
| **T5-30min** | 30 min | r²SCAN-3c with the built-in D4 + gCP | **ΔE ±1–2 kcal/mol `[E]`** | **P** | A | **never add D4 or gCP — both are inside the composite** |
| **T5-1h** | 1 h | ωB97M-V/def2-TZVPP, three-leg Boys–Bernardi counterpoise | **ΔE ±0.5–1 kcal/mol `[E]`**; raw, CP and half-CP all reported | **P** | A | **never add D4 to a VV10 functional** |
| **T5-3h** | 3 h | **junChS-F12 composite energy** — ORCA F12 single points | **A14 MAX 0.11, MUE 0.06, RMSD 0.07 kJ/mol `[M]`** | C | A | requires F12 auxiliary and CABS bases; **no F12 gradient in ORCA** |
| **T5-12h** | 12 h | DLPNO-CCSD(T1)/TightPNO/cc-pVDZ-F12 (paired with CABS) with local energy decomposition | **ΔE ±0.2–0.5 kcal/mol `[E]`**, plus a physical decomposition | **S** | A | **no MDCI restart** — the job must fit one window |
| **T5-1d** | 1 d | PNO-space extrapolation CPS(6/7) | **ΔE ±0.1–0.2 kcal/mol `[M]`** (S66: TightPNO 0.20 → CPS(6/7) 0.11) | **S** | A | CPS is calibrated against tighter PNO calculations, **not against canonical CCSD(T)** — a convergence measure, not an accuracy measure |
| **T5-3d** | 3 d | canonical CCSD(T)-F12D/RI single point | **ΔE ±0.1–0.2 kcal/mol `[E]`** | **S** | A | **dominated by T5-3h** on cost: junChS-F12 delivers 0.06 kJ/mol MUE at a fraction of this |
| **T5-1w** | 1 w | CCSD(T)/CBS + Δcore + ΔT increments | **ΔE ±0.05–0.1 kcal/mol `[E]`** | **S** | A | thermochemical-composite territory; **HEAT, Wn, Gn and ccCA take a geometry as input and produce no rotational constant** — footnote only |
| **T5-1mo** | 1 mo | SAPT(DFT) or SAPT2+3 decomposition — Psi4 or autoPES | the physical decomposition: electrostatics, exchange, induction, dispersion | C | A | a decomposition is an interpretation, not an accuracy improvement |

**Expansion block — Table 5**

| Row ID | Core-h | Input / workflow | State-in | State-out | Frozen-mono | Mem / scratch | Licence | Setup 1? | Max benchmark error | Mitigation / notes |
|---|---|---|---|---|---|---|---|---|---|---|
| T5-10s | 0.022 | `! XTB2` on dimer and two monomers | ensemble `.xyz` | energies → HDF5 — **D** | frozen-inc | <0.5 GB | free | Yes | — | the three legs share one geometry |
| T5-1min | 0.13 | MLFF single points | ensemble `.xyz`; model key | energies + committee σ — **D** | frozen-inc | 1 GB VRAM | free | Yes (CPU) | S30L 7.31, PLA15 29.9 kcal/mol `[M]` | **guard G4 applies: audit ρ on 20 structures before culling** |
| T5-30min | 4 | `! r2SCAN-3c TightSCF DefGrid3`, three legs | `.xyz` + previous conformer `.gbw` | raw/CP/half-CP — **D** | frozen-inc | 1 GB per rank / 1 GB | ORCA academic | Yes | ROT34 AMAX 1.5 % `[M]` | one dispersion model per composite |
| T5-1h | 8 | ghost atoms with `:`; one exported `.bas` shared by all three legs | one dimer `.xyz`; **no cross-leg `.gbw` reuse dimer → monomer** | raw/CP/half-CP — **D** | frozen-inc | 2 GB per rank / 3 GB | ORCA academic | Yes | S66 CP reduces MAE from ~0.7 to ~0.2 kcal/mol at aVDZ `[M]` | full CP at double zeta, half-CP at triple zeta and above, CP-free with F12 |
| T5-3h | 24 | CCSD(T)-F12b/jun-cc-pVTZ + MP2-F12 CBS + MP2 CV, added | `s(3h).xyz` + `.gbw` | composite energy — **D** | frozen-inc | 3 GB per rank / 20 GB | ORCA academic | No | junChS-F12 A14 MAX 0.11 kJ/mol `[M]` | **replaces v3's 3 d F12 row.** Costs "no more than twice the underlying coupled-cluster step" |
| T5-12h | 96 | `! DLPNO-CCSD(T1) TightPNO cc-pVDZ-F12 (paired with CABS) cc-pVDZ-F12 (paired with CABS)/C def2/JK TightSCF` + `%mdci TCutPNO 1e-7 DoLED true StorageType Shared end` | `.xyz` + `.gbw` from T3O-3h | energies, LED terms — **D**, no MDCI restart | frozen-inc | 3 GB per rank / 5–20 GB | ORCA academic | No | TightPNO S66 outliers <0.3 kcal/mol `[M]` | `StorageType Shared` works only when all ranks are on one node — which is Setup 2's situation |
| T5-1d | 192 | two MDCI energies at consecutive `TCutPNO` exponents; E = E^X + 1.5(E^Y − E^X) | per-threshold `.gbw` | extrapolated energy — **D** | frozen-inc | 3 GB per rank / 20 GB | ORCA academic | No | S66 CPS(6/7) 0.11 kcal/mol `[M]` | **the exponents must be consecutive** — v3's "1e-5 → 1e-7" ladder was invalid; F = 1.5 "should NOT be changed" |
| T5-3d | 576 @ 8; 9,216 @ 128 | `! CCSD(T)-F12D/RI cc-pVTZ-F12 cc-pVTZ-F12-CABS` | `.xyz` + `.gbw` | reference energy — **D** in ORCA, **R** in CFOUR | frozen-inc | 8 GB / 100 GB | ORCA academic | No | — | **dominated by T5-3h** |
| T5-1w | 1,344 @ 8; 21,504 @ 128 | composite increments | `JOBARC`/`MOINTS`/`MOABCD` where CFOUR is used | reference energy + increments — **R** in CFOUR | frozen-inc | 16 GB / 200 GB | CFOUR academic | No | W4 MAD 0.066 kcal/mol vs ATcT `[M]` | for a *trimer*, state that D3/D4 is pairwise-additive and three-body terms carry 15–20 % `[M]` |
| T5-1mo | 5,760 @ 8; 92,160 @ 128 | Psi4 SAPT, or autoPES's SAPT(DFT) route | `.xyz` | decomposition table | frozen-inc | 32 GB / 200 GB | Psi4 free / autoPES **`n.a.`** | No | — | SAPT(DFT) scales N⁵ against CCSD(T)'s N⁷, which is why it can afford a full-dimensional grid |

---

## 14. Tables 6–10: secondary observables, large-amplitude motion, and the non-microwave regimes

### 14.1 Table 6 — Dipole components, quadrupole coupling, distortion and the free observables

**Dual track.** The split is sharp here: ORCA emits a Pickett/SPCAT file and quartic distortion at DFT level; CFOUR emits sextic distortion, nuclear spin–rotation, the diagonal Born–Oppenheimer correction and CCSD(T)-quality electric field gradients, and has no SPCAT export.

#### Table 6-O — ORCA track

| Row ID | Tier | Method / Code | Delivers / accuracy | Conc. | Product | Limitations |
|---|---|---|---|---|---|---|
| **T6O-10s** | 10 s | inertial defect, planar moments, dipolar coupling D ∝ r⁻³ from any geometry | Δ sign; P_aa > P_bb > P_cc ordering; D | C | A | qualitative, but **free and independently constraining** |
| **T6O-1min** | 1 min | GFN2-xTB dipole in the principal axis system | μ_a, μ_b, μ_c **signed**, ±0.5 D `[E]` | C | A | semi-empirical dipoles decide which branch types exist and are unreliable at this level |
| **T6O-30min** | 30 min | ωB97X-V/def2-TZVPP dipole and electric field gradient | **μ ±0.1–0.3 D `[E]`**; χ_aa, χ_bb − χ_cc ~10 % `[E]` | **P** | A | hybrid DFT does **not** currently meet the ±0.1 D per-component requirement for flexible or weakly bound species |
| **T6O-1h** | 1 h | MP2/6-311++G(2d,2p) electric field gradients | **χ to ~5 % `[M]`** | C | A | the field-gradient basis requirement is tighter than the energy's |
| **T6O-3h** | 3 h | quartic centrifugal distortion from the T4O-30min harmonic force field | **quartic constants to a factor of 2 `[M]`**, free | C | A | free only if the Hessian already exists; otherwise it is a Hessian job |
| **T6O-12h** | 12 h | DLPNO-CCSD unrelaxed-density multipoles and field gradients | μ and χ at coupled-cluster quality for closed shells | **S** | A | **"an unrelaxed density for CCSD(T) is NOT available"** — CCSD only |
| **T6O-1d** | 1 d | vibrational corrections to μ and χ from the VPT2 force field | ⟨μ⟩, ⟨χ⟩; **χ(D) vibrational correction is 1.7 % `[M]`** | **S** | A | a vibrational correction inherits the force field's error |
| **T6O-3d** | 3 d | Boltzmann-averaged constants across the conformer ensemble + Pickett/SPCAT export | an assignment-ready `.var` template | C | A | the Pickett export is "still being refined and extended" |
| **T6O-1w** | 1 w | vibrational satellites B_v for the three lowest modes | B_v; order of magnitude for intermolecular modes, 0.1 % for intramolecular | **S** | A | intermolecular satellites are exactly where VPT2 is least trustworthy |
| **T6O-1mo** | 1 mo | **— sextic distortion, spin–rotation and DBOC cannot be produced by this track** | — | — | — | **Track gap.** None is documented in ORCA. **Escalate to `T6C-1mo`** |

**Expansion block — Table 6-O.** Core-h follow the standard ladder (0.022, 0.13, 4, 8, 24, 96, 192, 576, 1,344, 5,760 at 8 cores). **State-in** for every row from 30 min upward is the T3O-3h geometry plus its `.gbw` (`! MORead`); **State-out** is the property block of the `.out` plus, from T6O-3d, `pickett.txt` — **D** for every analytic-property row, since analytic frequencies and MDCI are not restartable. **Frozen-mono**: inherits the geometry row's flag. **Mem/scratch** 2–3 GB per rank / 3–20 GB. **Licence** ORCA academic throughout. **Setup 1** yes to 3 h, no beyond. **Max benchmark error**: camphor-class dipole studies show an 0.08 D component deciding whether a branch exists `[M]`; χ at 5 % from MP2/6-311++G(2d,2p) `[M]`; χ(D) basis shift TZ→6Z is 9.9 kHz = 6.3 % `[M]`. **Mitigation**: report signed components in the principal axis system, never magnitudes; for χ, state the basis and whether a vibrational correction was applied.

#### Table 6-C — CFOUR track

| Row ID | Tier | Method / Code | Delivers / accuracy | Conc. | Product | Limitations |
|---|---|---|---|---|---|---|
| **T6C-10s** | 10 s | — | — | — | — | **Cannot fill.** No cheap engine. Use `T6O-10s` |
| **T6C-1min** | 1 min | — | — | — | — | **Cannot fill.** Use `T6O-1min` |
| **T6C-30min** | 30 min | SCF/MP2 `PROPS=FIRST_ORDER` | dipole, quadrupole, octopole; electric field gradients | C | A | MP2 is not parallelised in CFOUR |
| **T6C-1h** | 1 h | CCSD(T)/cc-pVTZ `PROPS=FIRST_ORDER` | **CCSD(T)-quality μ and EFG → χ**, converted by χ(kHz) = EFG × Q(mbarn) × 234.96474 | C | A | the conversion is manual; no SPCAT export |
| **T6C-3h** | 3 h | quartic distortion from the `T4C-1h` analytic Hessian | quartic constants; ~2–3 % on oxirane `[M]` | C | A | requires the analytic CCSD(T) Hessian first |
| **T6C-12h** | 12 h | **sextic centrifugal distortion** from `ANHARM=VPT2` | **sextic constants, 3–4 % on oxirane `[M]` — CFOUR only** | C | A | needs the full VPT2 field; **not available in ORCA at any tier** |
| **T6C-1d** | 1 d | nuclear spin–rotation, `SPINROT=ON` | C_aa, C_bb, C_cc; **~3 % demonstrated for D₂O `[M]`** | C | A | NMR-type input; one of the few places computation sits at experimental accuracy |
| **T6C-3d** | 3 d | vibrationally averaged properties, ⟨A⟩ = A_e + Σ_r (∂A/∂Q_r)⟨Q_r⟩ + … | ⟨μ⟩, ⟨χ⟩, ⟨A⟩ at coupled-cluster quality | **S** | A | inherits the anharmonic field's error |
| **T6C-1w** | 1 w | diagonal Born–Oppenheimer correction, `DBOC=ON` | the DBOC contribution to the constants | **S** | A | limited to HF, MP1, MP2 and CCSD for RHF/UHF |
| **T6C-1mo** | 1 mo | relativistic corrections, `RELATIVISTIC=DPT2` or `X2C1E`, stacked with the above | the closed small-correction set for a heavy-atom complex | **S** | A | **no SPCAT export — transcribe the constants by hand (`n.a.`)** |

**Expansion block — Table 6-C.** Core-h as the standard ladder; Setup-3 rows also quote the 128-core figure (T6C-3d 9,216; T6C-1w 21,504; T6C-1mo 92,160). **State-in** is `JOBARC` (+ `JAINDX`, and `MOINTS`/`MOABCD` for coupled-cluster restarts); **State-out** is `JOBARC` plus the printed property block — **R** throughout, because CFOUR restarts both coupled cluster and finite-difference frequencies, which is the reason these rows are reachable under a 48 h queue and their ORCA counterparts are not. **Frozen-mono** `frozen-inc`. **Mem** is a *global* `MEMORY_SIZE=32, MEM_UNIT=GB`, not per rank — the default is ≈762 MB and must be raised. **Licence** CFOUR academic, wet signature, two-year renewing term. **Setup 1: no, at every tier** — the licence does not scale to a class.

### 14.2 Table 7 — Internal rotation, tunnelling and large-amplitude motion

**Single track.** *CFOUR has no internal-rotation or tunnelling module documented; the reference-quality inputs it can supply are barrier heights, which are Table 5's business.*

**A binding correction from v3.** v3's Table 7 claimed V₃ to ±10 %, ±7–10 %, ±5–7 %, ±5 % and ±3–5 % across five rows **while all five cited the same benchmark, whose own spread is ±14 %** (ammonia–formic acid: computed span 168.3–212.8 cm⁻¹ against 195.18(7) measured). **Every accuracy cell in this table is now capped at ±14 %, the only supported in-domain figure, and the tiers are differentiated by *coverage* — one-dimensional against two-dimensional against conformer-resolved — not by a number the benchmark does not support.**

| Row ID | Tier | Method / Code | Delivers / accuracy | Conc. | Product | Limitations |
|---|---|---|---|---|---|---|
| **T7-10s** | 10 s | GFN2-xTB relaxed torsional scan | the existence and rough height of a barrier; **no V₃ claim** | C | A | tight binding misplaces torsional barriers routinely |
| **T7-1min** | 1 min | MLFF torsional scan, dense | the shape of the one-dimensional path | **G** | A | use the shape, not the height |
| **T7-30min** | 30 min | r²SCAN-3c relaxed 1-D scan, ~36 points | **V₃ ±14 % `[M]`**, one-dimensional | **P** | A | a one-dimensional path through a multi-dimensional barrier is a lower bound on the true barrier |
| **T7-1h** | 1 h | ωB97X-V/def2-TZVPP relaxed 1-D scan | **V₃ ±14 % `[M]`**, one-dimensional, better electronic structure | **P** | A | same coverage limitation; the improvement is in the energy, not the path |
| **T7-3h** | 3 h | 2-D (τ, R) relaxed scan | **V₃ ±14 % `[M]`**, two-dimensional coupling captured | **P** | A | the coupling to the intermolecular stretch is often what the one-dimensional path misses |
| **T7-12h** | 12 h | 2-D scan + one-dimensional torsional Schrödinger solution → A/E splittings | A/E splittings and the reduced barrier s; **±14 % on V₃ propagates steeply into the splitting** | C | A | the splitting depends exponentially on V₃, so a 14 % barrier error is not a 14 % splitting error |
| **T7-1d** | 1 d | conformer-resolved barriers across the Table 1 ensemble | **V₃ ±14 % `[M]`** per conformer | **P** | A | ranking conformers by barrier inherits the energy ranking's error |
| **T7-3d** | 3 d | MPQC CCSD(T)-F12 single points along the converged path | the barrier at coupled-cluster quality, still **±14 %** against experiment | **S** | A | **the cap is set by the benchmark spread, not by the electronic structure** |
| **T7-1w** | 1 w | tunnelling splitting from an instanton or WKB treatment on the fitted path | **an estimate only — factor of 3 at best** | C | A | splittings span 6 MHz to 279,650 MHz within one molecule `[M]` |
| **T7-1mo** | 1 mo | full VRT treatment on the Table 2 6-D surface | band origins and splittings for the coupled manifold | C | A | **no tier in this document reliably delivers a tunnelling splitting**; report the barrier, the reduced mass and the path |

**Expansion block — Table 7.** Core-h as the standard ladder. **State-in**: the T3O-3h or T3O-12h geometry plus `.gbw`; scan rows consume the previous point's `.gbw` by `! MORead` (free, and each point is a small perturbation). **State-out**: `.xyz` + `.gbw` per scan step, the fitted path, and the effective one-dimensional potential — **D**, because a relaxed scan is a set of independent optimisations and is decomposed rather than checkpointed. **Frozen-mono**: `frozen-iso` for the intermolecular torsion; `relaxed` if the barrier involves an internal rotor of one monomer. **Mem/scratch** 1–3 GB per rank / 1–20 GB. **Licence** ORCA academic. **Setup 1** yes to 3 h. **Max benchmark error: ±14 %, ammonia–formic acid, computed span 168.3–212.8 cm⁻¹ against 195.18(7) `[M]` — the same figure for every row, and the accuracy cells are capped at it accordingly.** **Mitigation**: report the barrier with the reduced mass and the path; run the relaxed scan in both directions and report the hysteresis; state whether the path is one- or two-dimensional.

### 14.3 Table 8 — Infrared, terahertz and far-infrared

**Dual track**, because the anharmonic force field is the product and only CFOUR can produce a CCSD(T)-quality one.

#### Table 8-O — ORCA track

| Row ID | Tier | Method / Code | Delivers / accuracy | Conc. | Product | Limitations |
|---|---|---|---|---|---|---|
| **T8O-10s** | 10 s | GFN2-xTB harmonic frequencies | mode ordering only | C | A | semi-empirical frequencies are qualitative |
| **T8O-1min** | 1 min | MLFF harmonic frequencies | mode ordering, fast, on the GPU | **G** | A | float32 noise sits near the soft-mode frequencies |
| **T8O-30min** | 30 min | r²SCAN-3c analytic harmonic Hessian | **ω ±25–40 cm⁻¹ `[M]`** (aRMSD 35 cm⁻¹) | **S** | A | harmonic; intermolecular modes are the least harmonic thing in the system |
| **T8O-1h** | 1 h | ωB97X-V/def2-TZVPP analytic harmonic Hessian + IR intensities | **ω ±20–35 cm⁻¹ `[E]`**, intensities | **S** | A | analytic Hessians are **not restartable** |
| **T8O-3h** | 3 h | DFT VPT2 fundamentals | **ν ±15–25 cm⁻¹ `[E]`** | **S** | A | **no linear molecules**; VPT2 is sensitive to numerical noise |
| **T8O-12h** | 12 h | VPT2 with a substituted hybrid force field: high-level harmonic + low-level anharmonic | **ν ±10–20 cm⁻¹ `[E]`** at no extra anharmonic cost | **S** | A | **hybrid fields degrade for low-symmetry systems and fail catastrophically under large-amplitude motion** — exclude every mode below ~100 cm⁻¹ |
| **T8O-1d** | 1 d | MLFF molecular dynamics with a dipole surface → IR from the dipole autocorrelation | a full spectrum including anharmonic couplings | **G** | A | **the Fourier resolution bound is Δν̃ = 1/(cT): 2 ps → 17 cm⁻¹, 4 ps → 8 cm⁻¹ `[D]`** — and band positions are limited by *sampling* long before they are limited by that bound |
| **T8O-3d** | 3 d | band origins from the Table 2 3-D DVR | **±5–20 cm⁻¹ `[E]`** for the intermolecular manifold | C | A | only the modes the surface spans |
| **T8O-1w** | 1 w | 6-D variational band origins on the Δ-learned surface | **±1–5 cm⁻¹ `[E]`** | C | A | the fit residual is the floor: report it in cm⁻¹ |
| **T8O-1mo** | 1 mo | **— a CCSD(T)-quality anharmonic force field is not reachable in this track** | — | — | — | **Track gap.** `!VPT2` accepts only analytic-Hessian methods, and the DLPNO alternative needs 176,400 single points at N = 10. **Escalate to `T8C-1mo`** |

#### Table 8-C — CFOUR track

| Row ID | Tier | Method / Code | Delivers / accuracy | Conc. | Product | Limitations |
|---|---|---|---|---|---|---|
| **T8C-10s** | 10 s | — | — | — | — | **Cannot fill** — CFOUR has no semi-empirical or force-field engine. Use `T8O-10s` |
| **T8C-1min** | 1 min | — | — | — | — | **Cannot fill** — no machine-learned or force-field path. Use `T8O-1min` |
| **T8C-30min** | 30 min | SCF/MP2 harmonic frequencies, `VIB=EXACT` | ω at a low level | C | A | MP2 not parallelised |
| **T8C-1h** | 1 h | CCSD(T) analytic harmonic Hessian | **ω, IR intensities at coupled-cluster quality** | C | A | no ROHF-based analytic second derivatives |
| **T8C-3h** | 3 h | `ANHARM=VIBROT` | α constants only, not fundamentals | C | A | insufficient for a full spectrum |
| **T8C-12h** | 12 h | `ANHARM=VPT2`, cc-pVTZ | **fundamentals; cyclopentadiene CCSD(T)/ANO within ~0–19 cm⁻¹, ethylene ~10 cm⁻¹ `[M]`** | C | A | GUINEA (deperturbed VPT2, anharmonic intensities) is not public |
| **T8C-1d** | 1 d | the above plus resonance analysis and two-quantum transition intensities | overtones and combination bands | C | A | resonance treatment is where VPT2 is most delicate |
| **T8C-3d** | 3 d | `ANHARM=VPT2` at cc-pCVTZ/ANO1 with core correlation, queue-split | **ν within ~5–10 cm⁻¹ `[E]`** | **S** | A | script-driven job splitting; `FD_PROJECT=OFF` in the parallel recipe |
| **T8C-1w** | 1 w | isotopologue spectra from the same force field | full spectra for 6–15 isotopologues | C | **C** | `ANHARM=VPT2`, not `VIBROT`, is required |
| **T8C-1mo** | 1 mo | the above plus relativistic and DBOC corrections | the closed set for a heavy-atom complex | **S** | A | `ANHARM=FULLQUARTIC` unavailable publicly |

**Expansion block — Tables 8-O and 8-C.** Core-h follow the standard ladder, with 128-core figures for the 3 d, 1 w and 1 mo rows (9,216 / 21,504 / 92,160). **State-in**: the geometry and `.gbw` (ORCA) or `JOBARC` (CFOUR) from the corresponding Table 3 row. **State-out**: `.hess` / `FCMFINAL`, which feeds Tables 4 and 6 and every isotopologue — ORCA analytic-frequency rows are **D** (not restartable) while every CFOUR row is **R**. **Frozen-mono** inherits. **Mem** 3 GB per rank (ORCA) or a 32 GB global allocation (CFOUR); scratch 6–250 GB. **Licence** ORCA academic / CFOUR academic wet-signature. **Setup 1** yes to 3 h on the ORCA track, never on the CFOUR track. **Max benchmark error**: r²SCAN-3c aRMSD 35 cm⁻¹ `[M]`; small-basis C–H stretches too high "by 10–20 cm⁻¹ or more" `[M]`; water VPT2+K overtones within 1.4 cm⁻¹ `[M]`. **Mitigation**: state which modes were treated harmonically and which anharmonically, and exclude every mode below ~100 cm⁻¹ from a hybrid force field.

### 14.4 Table 9 — Raman

**Single track (ORCA).** *Raman intensities are not in CFOUR's public release.*

| Row ID | Tier | Method / Code | Delivers / accuracy | Conc. | Product | Limitations |
|---|---|---|---|---|---|---|
| **T9-10s** | 10 s | GFN2-xTB polarizability | qualitative activity only | C | A | semi-empirical polarizabilities are indicative |
| **T9-1min** | 1 min | MLFF polarizability surface, where available | activity pattern | **G** | A | polarizability models are less mature than energy models |
| **T9-30min** | 30 min | r²SCAN-3c numerical Raman intensities | **ω ±25–40 cm⁻¹ `[M]`**, activities to a factor of ~2 `[E]` | **S** | A | Raman intensities need polarizability derivatives, hence extra displacements |
| **T9-1h** | 1 h | ωB97X-V/def2-TZVPP Raman | **ω ±20–35 cm⁻¹ `[E]`** | **S** | A | diffuse functions matter more for polarizabilities than for energies |
| **T9-3h** | 3 h | Raman with an augmented basis | intensities with converged polarizabilities | **S** | A | near-linear dependence risk from diffuse sets |
| **T9-12h** | 12 h | VPT2 anharmonic Raman | **ν ±15–25 cm⁻¹ `[E]`** | **S** | A | anharmonic intensities are the least reliable output of VPT2 |
| **T9-1d** | 1 d | MLFF MD with a polarizability surface → Raman from the polarizability autocorrelation | a full spectrum | **G** | A | **the polarizability-surface fitting error is not benchmarked for this class — `n.a.`; planning value a factor of 2 in intensity `[E]`** |
| **T9-3d** | 3 d | Placzek-approximation decomposition into isotropic and anisotropic parts | depolarisation ratios | C | A | **the decomposition's validity for a floppy complex is not established — `n.a.`; treat as qualitative `[E]`** |
| **T9-1w** | 1 w | Raman band origins from the Table 2 surface | intermolecular Raman activity | C | A | only the modes the surface spans |
| **T9-1mo** | 1 mo | resonance Raman or a full polarizability surface | the complete activity map | **S** | A | **dominated for every microwave observable**; retain only if Raman is itself the deliverable |

**Expansion block — Table 9.** Core-h as the standard ladder. **State-in**: geometry plus `.gbw`; the polarizability derivative rows additionally consume the `.hess`. **State-out**: `.hess` with Raman activities; the MD rows emit trajectories and a polarizability time series — **D** for analytic rows, **R** for MD (`%md Restart IfExists`). **Frozen-mono** inherits. **Mem** 2–4 GB per rank / 3–40 GB. **Licence** ORCA academic. **Setup 1** yes to 3 h. **Max benchmark error** r²SCAN-3c aRMSD 35 cm⁻¹ `[M]`, and **the same figure may not certify five different bands** — the tiers here are differentiated by coverage and by whether anharmonicity is included, per Rule 3. **Mitigation**: report activities as ratios, not absolutes, and state the polarizability basis.

### 14.5 Table 10 — Nuclear magnetic resonance, ultraviolet–visible and mass spectrometry

**Single track (ORCA / QCxMS).** *These are outside the microwave critical path and are retained for completeness and for the teaching tier.*

**A correction from v3.** v3's 1 w row stated "at 5–15 s per MS step this is 14,000–42,000 core-hours" while the same row's core-hour cell read 1,344 — **a self-contradiction of a factor of 10–30, and the row was infeasible as specified.** It is re-costed below: the trajectory count is reduced by a factor of 20 and the row is marked Setup-3-only with honest core-hours.

| Row ID | Tier | Method / Code | Delivers / accuracy | Conc. | Product | Limitations |
|---|---|---|---|---|---|---|
| **T10-10s** | 10 s | GFN2-xTB electronic gap | a qualitative excitation estimate | C | A | not a spectroscopic prediction |
| **T10-1min** | 1 min | sTDA/sTDDFT on the DFT density | absorption band positions to ~0.3–0.5 eV `[E]` | **P** | A | simplified TDDFT is a screening tool |
| **T10-30min** | 30 min | TD-DFT (ωB97X-D4/def2-TZVP), 10 roots | vertical excitations ±0.2–0.3 eV `[E]` | **P** | A | charge-transfer states in a complex need a range-separated functional and still drift |
| **T10-1h** | 1 h | GIAO NMR shieldings at DFT | ¹H and ¹³C shifts, referenced | **P** | A | shifts in a weakly bound complex are dominated by conformational averaging, not by the shielding calculation |
| **T10-3h** | 3 h | NMR with vibrational corrections from the existing force field | corrected shifts | **S** | A | inherits the force field's error |
| **T10-12h** | 12 h | STEOM-DLPNO-CCSD excitations | excitations ±0.1–0.2 eV `[E]` | **S** | A | **no MDCI restart** — must fit one window |
| **T10-1d** | 1 d | QCxMS trajectory ensemble, ~500 trajectories | a fragmentation pattern, qualitative | C | A | **single-core trajectories: run as 16 concurrent single-rank jobs, and drop to 15 when a GPU feeder is running** |
| **T10-3d** | 3 d | QCxMS with ~2,000 trajectories | a converged fragmentation pattern | C | A | statistical convergence is the binding constraint, not the electronic structure |
| **T10-1w** | 1 w | QCxMS at a higher level, **~2,500 trajectories (re-costed, was ~50,000)** | branching ratios | C | A | **v3's specification was infeasible by 10–30×; this row is Setup-3-only and its core-hours are quoted at 128 cores** |
| **T10-1mo** | 1 mo | full multi-reference or high-level excited-state treatment | reference excitation energies | **S** | A | **dominated for every microwave observable** |

**Expansion block — Table 10.** Core-h: standard ladder to 3 d; **T10-1w is 21,504 core-h at 128 cores `[D]`, not 1,344** — the arithmetic is 2,500 trajectories × ~1,000 steps × 5–15 s ÷ 3,600, and the row does not fit Setup 2 at all. **State-in**: geometry plus `.gbw`; QCxMS consumes a geometry and a charge/multiplicity sidecar. **State-out**: `.cis` eigenvectors (TD-DFT, consumed by `orca_plot` and STEOM), shielding tensors, trajectory files — **D** throughout; QCxMS trajectories are independent and are the decomposition unit. **Frozen-mono** `relaxed`. **Mem** 2–8 GB per rank / 3–100 GB. **Licence** ORCA academic; QCxMS free. **Setup 1** yes to 1 h; the 1 d and 3 d rows fit only as ensembles of independent short jobs, which is exactly what they are. **Max benchmark error**: not benchmarked in domain for weakly bound complexes — **`n.a.`; planning values as stated, all `[E]`.** **Mitigation**: for NMR, average over the Boltzmann ensemble before comparing to experiment; for QCxMS, report the trajectory count and the seed.

---

## 15. The Pareto frontier, dominated rows, and the two use cases

### 15.1 Dominated rows, collected

A row is **dominated** when another row delivers the same observable at least as well for less. The list below is the shortest path to a good choice in this document, and it is the reason §15 sits early rather than at position 15 of 24.

| Row | Dominated by | Reason |
|---|---|---|
| `T3O-1d` (DLPNO numerical-gradient optimisation) | **`T3O-12h` (junChS)** | 375 core-hours of single points for a floppy geometry no better than `T3O-3h`, against a composite with a **0.13 % MAE in B_e `[M]`** at 6–20 h |
| `T3O-3d` (canonical CCSD(T)/cc-pVTZ-F12 (paired with CABS: OptRI [Yousaf & Peterson, J. Chem. Phys. 129, 184108 (2008)], JKFIT, MP2FIT)) | **`T3C-3d` or recipe R3 (junChS-F12)** | fc-CCSD(T)/cc-pVQZ gave **−6.56 % in A_e** for a C₅H₂ isomer `[M]`; junChS-F12 is "one order of magnitude faster than the CBS+CV counterparts" and reaches 0.0011 Å MUE on SE100 |
| `T3O-1mo` (F12 reference points) | **`T3O-1w`** | no improvement in the reported constants |
| unconstrained r²SCAN-3c geometry | **`T3O-1min` (recipe R1, frozen monomers)** | identical cost, identical B, **A improved by ~1.5 percentage points `[D]`** |
| `T5-3d` (CCSD(T)-F12D single point) | **`T5-3h` (junChS-F12 energy)** | the same F12 physics with composite bookkeeping, at 0.06 kJ/mol MUE `[M]` |
| `T1-1mo` (exhaustive union search) | **the concurrent two-device union of §8A.2** | running GOAT on both devices from the same start delivers much of the same diversity for **zero additional wall time** |
| `T9-1mo`, `T10-1mo` | every microwave row | these deliver nothing that appears in the rotational Hamiltonian |
| **any row applying an additive diffuse-function correction** | **any row with diffuse functions in the basis** | MAE degrades from 1.52 % to **12.74 % `[M]`** |

**Removed from v3's dominated list.** The diffuse-triple-zeta geometry row (`T3O-1h`). v3 declared it dominated on the grounds that "the constants barely move"; that is contradicted by the counterpoise evidence of §4.7 — going from cc-pVTZ to cc-pVDZ-F12 (paired with CABS) cuts the counterpoise discrepancy from 4.1 pm to 1.1 pm, a ~2 % versus ~0.8 % effect on B.

### 15.2 The frontier depends on the use case

Two use cases rank the tiers in opposite orders.

**Assignment.** You want the narrowest window per hour. The frontier is `T3O-1min` → `T3O-3h` → `T3O-12h` → `T4O-1h`, and it stops there: above 12 h the electronic structure is no longer the limiting term, ΔB_vib is. **If a measured parent exists, `T4O-1min` beats everything above it by an order of magnitude at one minute of cost.**

**Characterisation.** You have an assignment and want a defensible structure and energetics. The frontier runs `T3O-12h` → `T3C-12h` → `T4C-12h` → `T5-3h`, and the CFOUR track is not optional: sextic distortion, spin–rotation and a CCSD(T)-quality anharmonic field exist only there.

### 15.3 Non-monotonicity, restated

**Accuracy is not monotonic in cost.** Three documented instances: canonical CCSD(T)/cc-pVQZ underperforming a double hybrid by 6 percentage points in A_e for a C₅H₂ isomer; adding core-valence functions making agreement *worse* in that same study; and counterpoise correction degrading optimised geometries at double zeta. A tier table is a budget ladder, not an accuracy ladder, and every row that is not on the frontier says so in its notes.

---

## 16. Failure modes and mandatory guards

Version 1 of this document was entirely constructive: every row said what to run and what to do if it was slow, and no row said when its output was invalid. This section is the gate. **A result that fails any applicable criterion is discarded, not reported with a caveat.** The six failure modes in 16.1 are listed first because three of them corrupt results silently rather than crashing.

### 16.1 The six silent failure modes

**(1) SCF "near convergence" acceptance.** ORCA distinguishes complete convergence, near convergence and no convergence, where "near SCF convergence is defined as being not completely converged but: deltaE < 3e-3; MaxP < 1e-2 and RMSP < 1e-3", with default `MaxIter` = 125, and for a geometry optimisation near-convergence at one step does not necessarily abort the run ([ORCA Input Library, SCF convergence issues](https://sites.google.com/site/orcainputlibrary/scf-convergence-issues)). **A ΔE tolerance of 3 × 10⁻³ Eh is 1.9 kcal/mol — roughly half the water-dimer binding energy.** A weakly bound complex whose SCF near-converges at a few optimisation steps produces a plausible-looking geometry built on unreliable gradients. *Guard:* grep every output for near-convergence warnings and reject the run; use `TightSCF` minimum, `VeryTightSCF` for anything numerically differentiated; raise `MaxIter`; use `SlowConv` or `%scf DIISMaxEq 20 end` for stubborn cases rather than accepting a near-converged step.

**(2) Grid non-invariance under rotation.** "Because the atom-centered integration grids used in most quantum chemistry packages are anchored to the Cartesian axes, DFT energies typically lack invariance with respect to rigid-body rotations", and "first and second derivatives of the DFT energy with respect to nuclear displacements also lack rotational invariance", so "both the geometries of stationary points and the values of harmonic vibrational frequencies depend on the molecular orientation". The worked example: 2-butyne at B97-D/def2-TZVP on a pruned (75,302) grid gives a methyl rocking mode that **varies from 31i to 29 cm⁻¹ upon rotation**, so "the very nature of this stationary point … depends" on orientation, while the energy varies by only 0.01 kcal/mol ([Bootsma & Wheeler](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60c74474ee301c02d6c7916e/original/popular-integration-grids-can-result-in-large-errors-in-dft-computed-free-energies.pdf)). **An imaginary frequency that depends on molecular orientation is exactly the failure a van der Waals complex will hit**, because its softest modes are in that range. *Guard:* `DefGrid3` for every frequency calculation and every soft-mode analysis; re-run any structure with a mode below 50 cm⁻¹ in a rotated orientation and require the frequency to be stable; there is no `DEFGRID4`, so no protocol may depend on one.

**(3) Loose default optimisation thresholds.** `!Opt` leaves a residual displacement of roughly 3.6 pm, which is **2.1 % in B**; `!TightOpt` 1.2 pm and 0.69 %; `!VeryTightOpt` 0.36 pm and **0.21 %** — still above the 0.1 % target. *Guard:* the corrected `%geom` block of Section 4.4 (TolE 1e-7, TolRMSG 3e-6, TolMaxG 1e-5, TolRMSD 5e-5, TolMaxD 1e-4) on every row that reports a rotational constant, plus **reporting the final maximum gradient and the softest force constant** so the residual displacement Δr = g/k can be checked by the reader.

**(4) Linear dependence with diffuse sets.** Diffuse functions on hydrogen are a common source of near-linear dependence, and AutoAux with `aug-` sets can itself produce a near-linearly dependent fitting basis. *Guard:* monitor the smallest overlap eigenvalue; control with `%scf SThresh` and the ORCA 6 companion `DiffSThresh`; report the cutoff and the number of discarded combinations; prefer calendar sets (jun-, may-), which carry about 27 % fewer functions than `aug-` at triple zeta with comparable van der Waals curves ([Papajak & Truhlar](https://pubs.acs.org/doi/10.1021/ct200106a)); enable pivoted-Cholesky removal of redundant auxiliary functions. If more than a few combinations are discarded, repeat in a calendar basis rather than accepting the result.

**(5) Frozen-core defaults.** This is not a detail: the core-correlation effect on A_e was **1.0 percentage point with inconsistent sign across isomers**, and adding core-valence functions made agreement worse in one case ([Fortenberry *et al.*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10537648/)). ORCA's defaults freeze ten electrons for Al–Ca ([ORCA frozen core](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/frozencore.html)). *Guard:* a mandatory `fc`/`ae` field on every correlated row, with a core-valence basis whenever `ae` is used. Every correlated row in Tables 1–10 carries this field.

**(6) GPU floating-point non-determinism.** Run-to-run variability from floating-point non-associativity is real and quantified: the metric V_s(f) = 1 − |f_ND/f_D| is zero only for bitwise-identical implementations, and experiments over 100 arrays of 10⁶ FP64 elements with 10⁴ repeats per array, plus 5 × 10⁵-sum sweeps on NVIDIA GPUs and 10⁴ runs on an H100, show non-zero variability from atomic-order effects even in plain summation ([Kiran *et al.*](https://arxiv.org/html/2408.05148v3)). AIMNet2 additionally runs in float32 with about 4 × 10⁻⁶ Eh precision, **coarser than the gradient threshold the corrected `%geom` block demands**. *Guard:* FP64, or FP32 with FP64 accumulation, for any geometry that feeds a rotational constant; record kernel-determinism flags and the checkpoint hash; and — the binding rule — **a machine-learned geometry is a pre-optimisation only, and a DFT re-optimisation is mandatory before any A, B or C is quoted.**

### 16.2 Validation and rejection criteria

**Benchmark comparison.** Before a tier is used in production on a new system class it is validated against reference data for that observable: interaction and relative energies against counterpoise-corrected CCSD(T)/CBS on S66-class systems *and* on the weakly bound rare-gas cases that bracket the low end of the energy scale ([S66](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=98bccb93809e66f5eaf70568c6b9a6b9c80539d1); [Ar₂ benchmark potential](https://pubmed.ncbi.nlm.nih.gov/20831315/)); frequencies against anharmonic reference data with the mid-IR and the sub-200 cm⁻¹ region **scored separately**; rotational constants against experimental B₀; excitation energies against UCCSDT- or CC3-class references ([*JCP* 161, 144120](https://pubs.aip.org/aip/jcp/article/161/14/144120/3316843/Assessment-of-the-similarity-transformed-equation); [QUEST](https://arxiv.org/pdf/2001.00416v1.pdf)); shieldings against coupled-cluster values or referenced experimental shifts; electron-ionisation spectra against library spectra by cosine similarity with the scoring convention stated. The metric is named with the result, and **a tier whose measured error exceeds its class interval is re-classified, not re-labelled**. In-domain and out-of-domain statistics are reported in separate columns, never merged.

**Single-reference diagnostics.** Every CCSD(T), MPQC CCSD(T)-F12, EOM-CCSD and STEOM-CCSD result carries its T1 and D1 diagnostics and, where available, the percentage of the atomisation energy carried by the perturbative triples. Coupled cluster degrades systematically for large, highly polarizable, small-gap π systems where the perturbative triples overbind, and in that regime cross-validation against fixed-node diffusion Monte Carlo or rank-reduced CCSDT(Q) is required ([*JCP* 162, 114112](https://pubs.aip.org/jcp/article/162/11/114112/3339921/On-the-applicability-of-CCSD-T-for-dispersion); [arXiv:2410.12603](https://arxiv.org/abs/2410.12603)).

**Spin contamination.** Charge and multiplicity are stated for every calculation; unrestricted references for all open-shell species, which here means the whole mass-spectrometry column of Table 10 (charge +1, multiplicity 2). ⟨S²⟩ is recorded, and a deviation above 10 % of the expected value triggers rejection or recomputation. Open-shell MPQC CCSD(T)-F12 uses the same keyword as the closed-shell case but is designed for **high-spin** open shells, and open-shell singlets "may give qualitatively wrong results" ([MDCI](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/mdci.html)); such systems are routed elsewhere.

**Imaginary frequencies.** A minimum has no imaginary frequencies; a transition structure has exactly one, connected to its endpoints by an IRC or NEB relaxation. Modes below roughly 15i cm⁻¹ are treated as numerical artefacts and the structure is regridded with `DefGrid3` and tighter SCF and recomputed; anything larger is rejected. Given failure mode (2), a soft imaginary mode must also be shown to be orientation-independent before it is accepted as physical.

**Machine-learned uncertainty.** Element membership is necessary but not sufficient: an in-domain element set can still produce a geometry far outside the training distribution. Every production prediction carries a committee uncertainty — the `aimnet2-nse` checkpoint ships as a four-member ensemble ([model card](https://huggingface.co/isayevlab/aimnet2-nse)) and MACE4IRmol is published as an uncertainty-aware ensemble ([arXiv:2508.19118](https://arxiv.org/abs/2508.19118)). Thresholds: per-atom force uncertainty above roughly 10 % of the mean force magnitude, or energy uncertainty above the tier's target interval, rejects the frame and escalates it. Charge and spin are checked, not just elements, because MACE-OFF models are explicitly limited to neutral, non-radical, non-reactive systems ([MACE-OFF](https://pmc.ncbi.nlm.nih.gov/articles/PMC12123624/)). A trajectory with more than 5 % of frames above threshold is discarded in full rather than filtered.

**Basis and SCF criteria.** `TightSCF` or better for energies, `VeryTightSCF` or `ExtremeSCF` for anything numerically differentiated; the ladder is Sloppy, Loose, Medium, Strong, Tight, VeryTight, Extreme ([ORCA 6.1, SCF](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/scf.html)). Geometry optimisation uses the Section 4.4 block for rotational constants and `TightOpt` at minimum for energetics. Grids are `DefGrid2` or `DefGrid3`.

**Counterpoise and PNO consistency.** An interaction energy is reported with its raw value, its counterpoise-corrected value and their difference. **If the counterpoise correction exceeds 25 % of the raw interaction energy the basis is inadequate for that system and the calculation is repeated in a larger basis rather than reported.** Local coupled-cluster results carry their numerical TCutPNO value, not a label, and any high-tier result carries either TightPNO with a stated threshold or a CPS extrapolation with the spread quoted as the uncertainty.

**Dynamics convergence.** A dynamic spectrum is reported only with: the equilibration protocol and length, the production ensemble and thermostat coupling, the timestep, the total production time, the number of independent replicates, the autocorrelation depth and resulting nominal resolution, the window function, the quantum correction factor, and whether cross-correlation terms were included. A spectrum whose band positions shift between replicates by more than the tier's target interval is not converged and is not reported.

**Cross-tier consistency.** Adjacent tiers computing the same observable should agree within the looser of the two target intervals. Disagreement beyond that is a signal to investigate, not evidence that the higher tier is right.

**Bead- and grid-convergence gates for the dynamical rows.** A path-integral result reports ⟨μ⟩ convergence against bead count, not only energy convergence, because ⟨μ⟩ converges more slowly. A DVR result reports convergence of ⟨μ⟩ under both doubling the number of points at fixed box length and extending the box at fixed spacing, because ⟨μ⟩ weights the tail of the wavefunction. A run that shows only E₀ convergence has not demonstrated that its rotational constants are converged.

**Symmetry and embedding gates.** Eckart alignment is applied within one symmetry-distinct basin or declared `n.a.`; nuclear spin statistical weights are supplied by the user and their derivation stated, because `molsym` handles point groups rather than molecular symmetry groups and PGOPHER requires user-supplied weights.

### 16.3 Summary of rejection triggers

A result is discarded if: the SCF fails to converge, near-converges, or converges to an orbital-space saddle point; T1 or D1 exceeds the single-reference threshold; ⟨S²⟩ deviates by more than 10 % of its expected value; a minimum has any imaginary frequency above 15i cm⁻¹, or a soft imaginary mode that changes on rotation of the input orientation; a transition structure has other than exactly one imaginary mode, or its IRC fails to connect the intended endpoints; a machine-learned uncertainty exceeds the Section 16.2 threshold; the counterpoise correction exceeds 25 % of the raw interaction energy; more than a few basis-function combinations are removed for linear dependence; the autocorrelation function is unconverged between replicates; a bead- or grid-convergence test was run on E₀ only; a rotational constant is quoted from a geometry that was not re-optimised at a quantum-chemical level with the Section 4.4 thresholds; or two adjacent tiers disagree beyond their target intervals with no explanation.

---

## 17. Validation protocol: the six-system working set

A method is not validated by a mean error on S66. It is validated by reproducing measured constants on systems in the actual target class. **These six systems are chosen so that between them they exercise every observable this document claims to predict**, and every constant below is an experimental value from the cited primary source.

| # | System | Measured rotational constants (MHz) | Secondary observables | What it tests |
|---|---|---|---|---|
| 1 | **Ar–ketene (H₂CCO–Ar)**, A₁ state | A = 10447.9248(10); B = 1918.0138(16); C = 1606.7642(15) | μ_a = 0.125(3) D, μ_b = 1.369(2) D; hydrogen-atom tunnelling splits a- and b-type lines into two states, quenched in HDCCO–Ar; R_cm = 3.589(1) Å, θ_cm = 83.3(6)°; δ_J = 16.0856(70) kHz, δ_K = −152.24(23) kHz, h_K = 1.562(64) kHz | Rare gas + tunnelling + measured dipole components + four structurally fitted isotopologues: A, B, C, μ components and a tunnelling prediction in one system ([NIST / Gillies *et al.*](https://www.nist.gov/publications/rotational-spectra-structure-internal-dynamics-and-electric-dipole-moment-argon-0)) |
| 2 | **Ar–oxazole (C₃H₃ON·Ar)** | A = 5012.89486(14); B = 1398.428151(32); C = 1388.952841(31) | χ_aa = 2.3032(6), χ_bb = −4.0526(8), χ_cc = 1.7494(4) MHz; D_J = 5.52411(28) kHz, D_JK = 37.1990(30) kHz, D_K = −35.922(28) kHz; κ = −0.99; r_s(Ar–ring) 3.447 Å against r₀ 3.458 Å; b- and c-type only, no a-type; 103 components fitted, σ = 0.70 kHz, 3–21 GHz | Rare gas + full ¹⁴N quadrupole tensor + near-prolate (B − C = 9.48 MHz) so it stresses any near-symmetric-top failure boundary; also a documented topology ambiguity, "geometries 1 and 4 could not be distinguished experimentally" ([Kraka, Cremer, Spoerel, Merke, Stahl & Dreizler](https://s3.smu.edu/dedman/catco/publications/pdf/JPhysChem_99_12466_1995.pdf)) |
| 3 | **H₂CO–H³⁵Cl** | B = 2687.856(23); C = 2527.412(23); **A not determined, fixed at 42 GHz** | eQq_aa = −41.424(14), eQq_bb = 14.106(19), eQq_cc = 27.318(19) MHz; ³⁷Cl eQq_aa = −32.678(11) MHz; Δ_J = 0.0105(12), Δ_JK = −0.223(10) MHz; Δ_K, δ_J, δ_K indeterminate and set to zero; k_σ = 0.069(8) mdyn Å⁻¹, ω_σ = 85(4) cm⁻¹; four isotopologues; the hydrogen bond is 13° from linear | A hydrogen-bonded dimer with a large chlorine quadrupole tensor and an **undeterminable A** — the canonical case where theory must supply a parameter, and therefore the test case for the prior-disclosure rule of Section 20.3 ([Fraser, Gillies, Zozom, Lovas & Suenram](https://www.sciencedirect.com/science/article/pii/0022285287900919/pdf)) |
| 4 | **Water dimer (H₂O)₂ / (D₂O)₂** | (H₂O)₂: A = 227,580.432 ± 0.50; (B+C)/2 = 6155 (K = 2, J = 2→3) and 6144 (K = 2, J = 3→4). (D₂O)₂: A = 120,327.492(32) | 1→4 tunnelling splitting −70,128.436 ± 0.10 MHz; acceptor switching 279,650 MHz (H₂O)₂ against 53,000 MHz (D₂O)₂; K = 0 donor–acceptor interchange 19,526.73 against 1,172.23 MHz; K = 0 donor tunnelling 6 MHz; D₀ = 1105 ± 10 and 1244 ± 10 cm⁻¹; r(O–O) = 2.98 ± 0.04 Å | The tunnelling stress test: nine orders of magnitude of splittings in one molecule plus a large H/D isotope effect. **Any method claiming to predict tunnelling must reproduce the (H₂O)₂/(D₂O)₂ ratio before an absolute value is quoted** ([Mukhopadhyay, Cole & Saykally](https://escholarship.org/content/qt1j70w8wt/qt1j70w8wt.pdf)) |
| 5 | **NH₃–HCOOH** | rotational constants as published in the cited work | V₃ = 195.18(7) cm⁻¹ experimental, against a computed span of 168.3–212.8 cm⁻¹ (±14 %) | The internal-rotation test, and the one case in the set where methods demonstrably disagree with one another rather than with experiment ([Roehling, Hill, Daly & Kukolich](https://experts.arizona.edu/en/publications/ammonia-formic-acid-complex-internal-rotation-analysis-calculatio/)) |
| 6 | **C₆H₆–HCN against Ar₃–HCN** | as published in the cited work | R_cm = 3.96 Å against 3.47 Å; χ projected on the figure axis −4.223(4) against −1.143(2) MHz; D_J = 10.374(2) kHz and D_JK = 173.16(1) kHz for Ar₃–HCN, showing "large centrifugal distortion including higher-order terms" | A **matched topology pair**: two genuinely different binding topologies differing by 0.49 Å in R_cm, i.e. roughly 25 % in B. Tests whether a cheap method can do the job it is actually needed for — discriminating topologies — without pretending to 0.1 % ([Gutowsky, Arunan *et al.*](https://pubs.aip.org/aip/jcp/article/103/10/3917/481163/Rotational-spectra-and-structures-of-the-C6H6-HCN)) |

**Two named literature sets supplement the six.** The 16-complex semi-experimental benchmark of *J. Phys. Chem. A* 2018 and the 21-dimer optimisation benchmark of *J. Chem. Phys.* 162, 174106 are used for statistics; the six systems above are used for diagnosis, because each one isolates a specific failure.

**Exclusion rule.** Any constant whose published value was influenced by a theoretical prior — fixed at a computed value, regularised toward one, or fitted with a computed parameter held constant — is **excluded from the validation set**, or, where it cannot be excluded (system 3, where A is fixed at 42 GHz), the affected parameter is excluded and the remaining parameters are used. Validating theory against a constant that theory helped produce is circular.

**Reporting rule.** For each system the validation report gives, per observable: the computed value, the measured value, the signed difference in the natural unit **and in MHz**, the percentage difference, and the tier and full method string that produced it. A single summary statistic across the six is not an acceptable substitute.

---

### 17.5 A defensible prediction interval: split-conformal, replacing the Bayesian anchor

**The problem with v3's statistical anchor.** v3 leaned on a Bayesian analysis of theoretical rotational constants for its search-window statistics. That source explicitly excludes this document's target class: "we have chosen to exclude classes of molecules with complicated electronic structure and nuclear motion. Examples include weakly bound or nonrigid molecules … (e.g., van der Waals complexes, large amplitude motion)", adding "For these systems, it is very likely that the low-cost methods investigated here would fail" ([Lee & McCarthy](https://par.nsf.gov/servlets/purl/10149706)). **v3's uncertainty model was calibrated on a population that excludes v3's subject matter.**

**The replacement.** Conformal prediction "is a technique for constructing prediction intervals that attain valid coverage in finite samples, without making distributional assumptions" ([Romano, Patterson & Candès](https://arxiv.org/pdf/1905.03222)); it has been transferred to molecular-property models ([conformal prediction for molecular properties](https://arxiv.org/pdf/2304.00970.pdf)), to machine-learned interatomic potentials ([conformal MLIP uncertainty](https://arxiv.org/html/2510.00721v1)) and to model-agnostic structure–activity models ([model-agnostic conformal QSAR](https://pmc.ncbi.nlm.nih.gov/articles/PMC13390030/)).

**The protocol for this document.** For tier row *t* and observable *B*:

1. Take the calibration set = the six-system working set of §17 plus the 16-complex semi-experimental benchmark, **held out from any fitting**.
2. Compute the absolute relative residual s_i = |B_calc,i − B_exp,i| / B_exp,i for every calibration system **at that row's level**.
3. The 90 % conformal half-width is the ⌈0.9(n+1)⌉-th smallest s_i. With n = 22 that is the 21st order statistic.
4. Emit `B_pred × (1 ± q̂₀.₉₀)`.

This is finite-sample valid without assuming normality, which is the right property for a benchmark set that is small and heavy-tailed. **Stratify by two groups only — semi-rigid and floppy — because conformal coverage degrades with the number of strata at this n.**

**What it changes.** It replaces "±0.3–0.5 %" — a `[D]` band with no coverage guarantee — with a number the document can defend as a coverage statement. **It is the single most rigorous upgrade available to §3 and §4, and it is open work: the calibration set has not yet been built** (§21.3, roadmap item 2). Until it is, the bands in §3.1 stand as `[D]` estimates and are flagged as such.

---

## 18. Deliverable specification

**The output of this document is not a table of numbers in a paper. It is a `.par` + `.int` pair that runs unmodified through SPCAT, plus the resulting `.cat` truncated to 2–20 GHz at 2 K.** A method that cannot produce that has not produced anything usable at the bench.

**Required contents of the `.par`:** A, B, C; the five quartic centrifugal distortion constants with the Watson reduction (A or S) and the axis representation stated; any quadrupole coupling constants with the correct parameter codes; and the source of every value, since a `.par` mixing computed and measured parameters must say which is which.

**Required contents of the `.int`:** signed μ_a, μ_b, μ_c, entered in the documented form `1 x.xxx /mua`, with the temperature as the ninth parameter on line 2 ([Kisiel, CRIB sheet for SPFIT/SPCAT](http://info.ifpan.edu.pl/~kisiel/asym/pickett/crib.htm)).

**The `.cat` format** is fixed-width Fortran `[F13.4, 2F8.4, I2, F10.4, I3, I7, I4, 12I2]`:

| Field | Columns | Meaning |
|---|---|---|
| FREQ | 1–13 | frequency in MHz |
| ERR | 14–21 | estimated uncertainty, MHz |
| LGINT | 22–29 | log₁₀ of the integrated intensity, nm²·MHz |
| DR | 30–31 | degrees of freedom in the rotational partition function (0 atom, 2 linear, 3 nonlinear) |
| ELO | 32–41 | lower-state energy, cm⁻¹ |
| GUP | 42–44 | upper-state degeneracy |
| TAG | 45–51 | species tag; a negative tag flags a laboratory-measured frequency |
| QNFMT | 52–55 | quantum-number format code |
| QN | 56–79 | quanta, upper state first (characters 1–12), lower state from character 14 |

([Pickett, SPFIT/SPCAT documentation](https://spec.jpl.nasa.gov/ftp/pub/calpgm/spinv.pdf); [CDMS mirror](https://cdms.astro.uni-koeln.de/classic/predictions/pickett/spinv.html)). Quanta above 9 are encoded 10–19 as a0–a9 up to z9 = 259, and 100–109 as A0–A9 up to Z9 = 359. QNFMT = Q × 100 + H × 10 + NQN, with Q mod 5 = 3 for asymmetric rotors, +11 for multi-state fits and +20 for two spins coupled to I_tot; asymmetric-top orderings include N,K_a,K_c,v,J,F and N,K_a,K_c,J,F₁,F.

**The traps that produce a silently wrong file**, all documented: parameter codes such as `110010000 = 1.5 χ_aa` for a prolate rotor, `110030000 = 1.5 χ_cc` for oblate and `110610000 = χ_ab`; spins rounded **up** (1/2 → 1, 3/2 → 2); nearly all quartic distortion constants carrying the **opposite sign** to the usual convention, except S-reduction d₁ and d₂; and misuse of ERPAR in the `.par`, documented as "TRAP 3" ([Kisiel, CRIB sheet](http://info.ifpan.edu.pl/~kisiel/asym/pickett/crib.htm)). The role of each file is also set out in the Cologne teaching material ([SPIN course, First Steps with SPFIT/SPCAT](https://spin.astro.uni-koeln.de/chapter/SPFITSPCATUniverse/)). SPFIT consumes a `.par` and a `.lin` and writes `.bak`, `.par`, `.fit` and `.var`; SPCAT consumes the `.var` plus the `.int` and writes `.out` and `.cat`, optionally `.egy` and `.str`, supporting up to 999 vibrational states and 9 spins.

**Tooling, named correctly.** The Python package that exists is **pyckett**, a wrapper around SPFIT/SPCAT with readers `parvar_to_dict`, `int_to_dict`, `lin_to_df`, `cat_to_df`, `egy_to_df` and `erhamlines_to_df`, and command-line tools `pyckett_auto`, `pyckett_add`, `pyckett_omit`, `pyckett_uncertainties`, `pyckett_qrot`, `pyckett_report`, `pyckett_duplicates` and `pyckett_pmix` ([pyckett on PyPI](https://pypi.org/project/pyckett/)); [PySpecTools](https://github.com/laserkelvin/PySpecTools) is the other established Python layer. **PGOPHER** is the main alternative front end: it simulates and fits linear, symmetric and asymmetric tops, reads plain text, JCAMP-DX, HITRAN and Pickett's CALPGM files, and fits line positions, intensities, energy levels, common differences or band contours ([Western, PGOPHER](https://pgopher.chm.bris.ac.uk/Help/PGOPHERaccepted.pdf)). **AABS** consumes `.LIN` and `.RES` files and is the standard interactive assignment front end ([Kisiel, internal-rotation programs](http://info.ifpan.edu.pl/~kisiel/introt/introt.htm)).

**A deliverable checklist for every reported species:** the `.par` and `.int`; the `.cat` over 2–20 GHz at 2 K; the dark-branch flags from Stage D of Section 13.1; the inertial defect and the three planar moments; the Stage E deduplication audit log; the final maximum gradient and softest force constant for the geometry; and the `fc`/`ae` field with the basis used.

---

## 19. The teaching tier, corrected

### 19.1 GitHub is not the only option, and v3's constraint was false

v3 framed the teaching tier around GitHub Actions and Codespaces and derived a binding constraint from the 20-job Free-tier concurrency limit. **That constraint is an artefact of the platform choice, not of the problem.**

**[ChemCompute](https://chemcompute.org/) provides free, class-scale compute to undergraduate classes** on four clusters: Jetstream ("A cloud cluster consisting of up to 10 nodes (10 cores each)"), **Expanse at SDSC (728 nodes, 128 cores per node)**, **Bridges2 at PSC (488 nodes, 128 cores per node)** and **Delta at NCSA (AMD EPYC CPU nodes and NVIDIA A100, A40 and H200 GPUs)** — with Expanse, Bridges2 and Delta "Available for use to registered users and **undergraduate classes**", running GAMESS, Psi4 and PySCF, and the whole service framed as being available "without the hassle of compiling, installing, and maintaining software and hardware". **This removes the concurrency constraint entirely, makes the 12 h and 1 d tiers reachable in a teaching context, and puts real GPUs in students' hands** — which ties directly to the gpu4pyscf correction of §8.2.

**Published lab material exists and should be cited rather than reinvented.** [Psi4Education](https://psicode.org/posts/psi4education/) offers "a suite of free, open-source lab activities", has "partnered with ChemCompute to enable you to run Psi4Education Jupyter notebook labs using FREE compute resources", and requires no local install: "You do not have to install or configure any software on your own computer." Emory's undergraduate quantum-chemistry lab publishes its full notebook and assignment set under Creative Commons ([Course-QuantumChemistryLab](https://github.com/fevangelista/Course-QuantumChemistryLab)).

**The 50-student claim needs a caveat v3 did not have.** The most detailed published account of remote computational-chemistry teaching reports that virtual-desktop delivery "beyond about 30" participants had not been extensively tested, describes autoscaling browser-based JupyterHub on Kubernetes for ~80 participants, and prices the alternative at **"roughly a dollar a day"** for a 2-core, 4 GB server — negligible "for a one-day workshop with 100 participants" ([UCLouvain, virtual computational-chemistry teaching laboratories](https://dial.uclouvain.be/pr/boreal/object/boreal:254784/datastream/PDF_01/view)). **A 50-seat class is fine, the realistic operating point is that paper's 38-student case study, and the cloud fallback costs about $50 a day for the whole class** — small enough that the licence-driven GitHub design is a choice, not a necessity.

**The licence constraint is real and unchanged.** ORCA may not be placed in a shared or public image (§11.1); CFOUR is worse and must not appear in a teaching stack at all (§9.2). **Teach with PySCF, Psi4 or xtb.**

**GitHub remains a good fallback**, and its limits are unchanged: 4 CPU / 16 GB public runners, free and unlimited minutes; a 6 h job cap; a 256-job matrix; 20 concurrent jobs on Free; 500 MB of artifacts against a 90-day default retention; a 10 GB cache. **v3's claim that a 4-CPU runner delivers "roughly 0.3–0.4× the throughput of Setup 2's eight P-cores" is unsourced `[E]` and it decides which tiers are teachable. It is flagged as requiring measurement** — one xtb job and one Psi4 job on a public runner, timed, is a ten-minute experiment (§21.2).

### 19.2 The 90-minute laboratory, revised

v3's design is good and is kept: **zero points for the value of B, 25 points for a four-term error budget.** It teaches judgement rather than software operation. Two blocks are added and one is moved out.

| Minutes | Block | What it teaches |
|---|---|---|
| 0–10 | launch; xtb constants | the mechanics, once |
| 10–20 | signed error against the measurement | that a signed error is information and an absolute error is not |
| 20–30 | window and candidate-line count | that a prediction is a window, and how many lines are in it |
| 30–45 | re-optimise; Δr = g/k | that "converged" is a claim about a threshold |
| **45–55** | **grid and threshold sensitivity (new)** | re-run at two grids and two convergence thresholds and plot ΔB against the setting. A published DFT-practice lab includes exactly this — students "understand that the choice of an integration grid can have a significant impact" ([The Devil in the Details](https://s3-eu-west-1.amazonaws.com/itempdf74155353254prod/10187756/The_Devil_in_the_Details__What_Everybody_Should_Know_When_Running_DFT_Calculations_v1.pdf)) |
| **55–65** | **counterpoise (new)** | optimise at cc-pVTZ with and without counterpoise and watch R move by ~4 pm, i.e. ~2 % in B ([Plumley & Dannenberg](https://pmc.ncbi.nlm.nih.gov/articles/PMC3073166/)). **The cheapest "your answer depends on something you did not think about" moment available** |
| 65–78 | the free observables | inertial defect, planar moments, dipole branch types |
| 78–88 | the four-term error budget | the graded deliverable |
| 88–90 | commit and reproduce | provenance |

The reproducibility Actions run moves to homework: it costs five minutes of class time and teaches provenance, not chemistry.

---

## 20. Reproducibility and provenance

### 20.1 The timing statement

**The ten wall-clock tiers in this document are estimated design targets, not measurements on the hardware of Section 8.** They are anchored on published benchmarks — an r²SCAN-3c SCF timing for a 153-atom, 7,155-AO complex on four cores ([r²SCAN-3c](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60c752f6bb8c1a21633dbf6c/original/r2scan-3c-an-efficient-swiss-army-knife-composite-electronic-structure-method.pdf)), a canonical CCSD/cc-pVTZ benzene timing on 16 cores ([PySCF benchmarks](https://pyscf.org/benchmark.html)), DLPNO-CCSD(T1) timings for 63- and 90-atom systems ([Nagy *et al.*, *Chem. Sci.*](https://real.mtak.hu/205919/1/state-of-the-art-local-correlation-methods-enable-affordable-gold-standard-quantum-chemistry-for-up-to-hundreds-of-atoms.pdf)), an F12 overhead measurement ([Pavošević & Neese, SI](https://arxiv.org/pdf/2008.03237.pdf)), ORCA's own statements on numerical-gradient and molecular-dynamics step cost ([ORCA optimisations](https://www.faccts.de/docs/orca/6.0/manual/contents/typical/optimizations.html); [ORCA molecular dynamics](https://www.faccts.de/docs/orca/6.0/manual/contents/detailed/moldyn.html)), QCxMS trajectory defaults ([QCxMS run documentation](https://xtb-docs.readthedocs.io/en/latest/qcxms_doc/qcxms_run.html)), and MACE-OFF23 throughput on an A100 ([*JACS* 2024](https://pubs.acs.org/doi/10.1021/jacs.4c07099)) — then scaled to the 5–10 atom system size. Rows dominated by numerical differentiation publish their explicit single-point counts so the estimate can be audited.

**Anyone using these tiers as a scheduling contract must measure them locally.** The protocol: three named reference complexes spanning the interaction classes (a rare-gas dimer, a hydrogen-bonded dimer, a mixed dispersion/induction pair); the exact atom and basis-function counts; the core count and thread or MPI configuration; three repeat runs with the median and spread reported; and a statement of whether the number quoted is wall-clock or core-hours. **The mandatory minimum is one single point on the actual complex at the tier's level, wall time recorded, with every point count in the tier rescaled by the measured value.**

### 20.2 What every reported result records

**Software.** ORCA version and build, including compiler, MPI implementation, BLAS library and whether the shared-memory or MPI build was used; OPI version; `xtb` version, and for g-xTB the specific preprint binary release, since it currently ships as modified `xtb 6.7.1` binaries rather than a mainline feature ([grimme-lab/g-xtb](https://github.com/grimme-lab/g-xtb)); QCxMS or QCxMS2 version, stated unambiguously since they are different codes with different engine lists; and the version of any external variational, path-integral or diffusion Monte Carlo code.

**Model checkpoints.** The exact checkpoint, not the family: MACE-OFF24(M) — note this is the only MACE-OFF24 variant, the small/medium/large trio existing only for MACE-OFF23, so "MACE-OFF24(S)" and "MACE-OFF24(L)" are not offered as alternatives anywhere in this document ([MACE-OFF](https://pmc.ncbi.nlm.nih.gov/articles/PMC12123624/); [ACEsuit/mace-off](https://github.com/ACEsuit/mace-off)) — MACE-OMOL-0 with its size variant, MACE-POLAR-1 with its variant (`polar-1-s`, `-m` or `-l`, requiring `mace-torch >= 0.3.16` plus `graph_electrostatics` — [MACE documentation](https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html)), MACE-MDP, MACE4IRmol, and the AIMNet2 checkpoint. `mace-torch`, PyTorch, CUDA and cuEquivariance versions are recorded alongside, because inference cost and, on reduced-precision paths, numerical results depend on them ([Speeding up MACE](https://arxiv.org/html/2510.23621v1)). Training-data provenance is stated correctly: MACE-OFF23 is trained on SPICE v1 at ωB97M-D3(BJ)/def2-TZVPPD (neutral formal-charge subset, ion pairs removed) augmented with QMugs 50–90-atom molecules and water clusters, and MACE-OFF24(M) adds SPICE v2 configurations with a 6.0 Å cutoff. **QM9 is not part of the training data.**

**Numerical settings.** SCF convergence level with its energy and density thresholds; integration grid with any `%method` overrides; the five geometry convergence thresholds as numbers; `fc` versus `ae` with the basis; PNO thresholds as numerical TCutPNO values rather than labels; extrapolation family, cardinal numbers, separate SCF and correlation formulae and their exponents; auxiliary-basis provenance including AutoAux settings; molecular-dynamics timestep, thermostat type and coupling time, equilibration and production lengths, and box composition; autocorrelation depth, window function and quantum correction factor; bead count and the ω_max used to derive it; DVR grid spacing, box length and basis size; the temperature of any Boltzmann weighting with the free-energy model and low-frequency treatment; and the numerical precision of any machine-learned inference together with kernel-determinism flags and the checkpoint hash.

**Seeds.** Recorded for every stochastic component: GOAT sampling, molecular-dynamics initial velocities, machine-learned trajectory initialisation, diffusion Monte Carlo walker initialisation, and QCxMS conformer and internal-excess-energy sampling.

**Data and code.** The routing logic, the counterpoise driver, the extrapolation scripts, the external-tool wrappers of Section 10 and the analysis pipeline are software components and are archived as such, with a repository, a commit hash and an archival DOI, together with the input and output files for the reference complexes used in the timing protocol.

### 20.3 Priors, disclosure, and the independence of experimental constants

Rotational fits of floppy complexes are frequently ill-conditioned: in H₂CO–HCl the A constant could not be determined and was held fixed at 42 GHz, while Δ_K, δ_J and δ_K were indeterminate and set to zero ([Fraser *et al.*](https://www.sciencedirect.com/science/article/pii/0022285287900919/pdf)). Fixing A at a computed value is already an infinitely sharp prior, and nobody objects to it; a stated prior with a propagated uncertainty is more honest than that, not less. **But the value of an experimental rotational constant is that it is independent of theory**, and its downstream uses — semi-experimental equilibrium structures, benchmark sets for method development, catalogue entries used to identify species in the interstellar medium — all depend on that independence. A constant regularised toward a MPQC CCSD(T)-F12 value cannot be used to test MPQC CCSD(T)-F12, and the contamination is invisible three papers later. It also breaks error propagation, because Kraitchman coordinates and their Costain errors are derived from measured moments of inertia, so a theory-shrunk B silently biases an r_s structure presented as experimental.

**The rule, in four parts.** (1) Priors may be used to generate search windows, to seed fits and to break degeneracies during assignment. (2) Any parameter whose reported value is influenced by a prior is flagged in the published table, exactly as fixed parameters conventionally are. (3) Constants entering catalogues, semi-experimental structures or method benchmarks come from an unregularised least-squares fit. (4) A regularised value and an unregularised value never appear in the same column.

The defensible use of Bayesian methods here is the inverse direction — characterising *theory* error so that search windows are statistically justified. Lee and McCarthy ran 6,916 optimisations to establish "a hierarchy of accuracy and uncertainty" among low-cost methods, finding B3LYP and MP2 worse than the Minnesota functionals and ωB97X-D, and used those uncertainties to simulate spectra and cross-correlate against broadband data ([Lee & McCarthy, *JPCA* 2020](https://pubmed.ncbi.nlm.nih.gov/31910016/)). Probabilistic deep learning has been applied to molecule identification with dropout as approximate Bayesian sampling over about 83,000 molecules ([*JPCA* 2020](https://pubs.acs.org/doi/10.1021/acs.jpca.0c01376)). **Bayes on the theory side of the interface, not on the experimental side.**

---

## 21. Hard limits, and the development roadmap

v3 carried a single fifteen-item list of withdrawn claims plus three aspirational items. That reads as an apology and it mixes two different kinds of thing. It is split here: **the hard limits a reader must know, and a roadmap of what would remove them.**

### 21.1 Seven hard limits

1. **No tier delivers B₀ to 0.1 % de novo.** B_e reaches 0.13 % by composite `[M]`; ΔB_vib is 0.1–0.7 % of B_e and is the limiting term. **0.1 % is reached only through Product B or C, which require a measurement.**
2. **No tier reliably delivers a tunnelling splitting.** Splittings span 6 MHz to 279,650 MHz within one molecule `[M]`. Report the barrier, the reduced mass and the path, and label the splitting an estimate — factor of 3 at best.
3. **No V₃ claim tighter than ±14 %** is supported by any in-domain benchmark. Every Table 7 accuracy cell is capped there.
4. **No completeness proof over binding topologies exists from any stochastic search.** GOAT and CREST are both "keep going until nothing new turns up" heuristics; the only completeness argument available is hand enumeration for a rigid or near-rigid complex.
5. **Foundation machine-learned force fields cannot rank van der Waals isomers.** Interaction-energy errors of 3.5–7.3 kcal/mol on S30L and 29.9 on PLA15 `[M]` exceed typical isomer separations. They enumerate; they do not judge. **No MLFF geometry may support a Product-A absolute constant** — though MLFF geometries remain valid for Product-C differences and for qualitative inertial-defect and planar-moment checks.
6. **Dipole components are not predicted to the ±0.1 D the experiment needs** by hybrid DFT for flexible or weakly bound species, and an 0.08 D component decides whether a branch exists.
7. **This document's statistical anchor excludes its own subject matter.** The Bayesian analysis v3 leaned on for search-window statistics explicitly excludes "weakly bound or nonrigid molecules … van der Waals complexes, large amplitude motion", adding "For these systems, it is very likely that the low-cost methods investigated here would fail" ([Lee & McCarthy](https://par.nsf.gov/servlets/purl/10149706)). **v4 replaces it with the split-conformal protocol of §17.5.**

### 21.2 Values that gate a decision and are not measured

Under Rule 7, a `[D]` or `[E]` value may not be the sole support for a hardware exclusion, a routing gate or an accuracy claim. Five currently are. **Each is stated here as an open measurement rather than buried in a cell.**

| # | Value | Tag | What it gates | The measurement that settles it |
|---|---|---|---|---|
| 1 | GPU crossover at **~50–90 basis functions against 8 P-cores** | `[D]` (de-rated from a 32-core measurement) | Step 2b of the routing procedure | the matched protocol of §8.4, on three systems spanning the crossover |
| 2 | Semi-rigid **0.3–0.5 %** and floppy **1–2 %** B₀ bands | `[D]` | every Product-A accuracy claim | the six-system working set of §17 plus the split-conformal procedure of §17.5 |
| 3 | Teaching-runner throughput **0.3–0.4× of 8 P-cores** | `[E]`, unsourced in v3 | which tiers are teachable | one xtb job and one Psi4 job on a 4-CPU public runner, timed |
| 4 | **DLPNO 15 min/point (×/÷ 3)** | `[E]`, extrapolated from published anchors on different molecules and bases | every PES and geometry cost in Tables 2, 3 and 5 | one DLPNO-CCSD(T1)/TightPNO/cc-pVDZ-F12 (paired with CABS) single point on the actual complex, wall time recorded, all point counts rescaled |
| 5 | **g-xTB 3N–6N numerical-gradient penalty** | `[D]` conditional on there being no analytic gradient | a two-tier demotion of every g-xTB row | time one g-xTB gradient through `ExtOpt` |

Three further quantities are honestly unknown and are recorded as such: **measured bandwidth contention** between ORCA and a concurrent GPU worker on this box; **measured thermal clock loss** with the GPU loaded; and the **MLFF-to-DFT rank correlation** for this complex class, which guard G4 requires to be measured per system rather than assumed.

### 21.3 Roadmap

Six things would materially change this document, in order of expected value.

1. **Run the §8.4 matched benchmark** and replace the `[E]` 5–30× GPU projection with a measured number. This is a half-day of work and it retires open item 1.
2. **Build the split-conformal calibration set** (§17.5) from the six-system working set plus the 16-complex semi-experimental benchmark, and replace every `±x–y %` band with a coverage-valid interval.
3. **Measure the frozen-monomer residual gradient** on three complexes spanning dispersion-bound to hydrogen-bonded, to calibrate when the deformation channel forces a relaxed-monomer optimisation (§9A.7 rule 8).
4. **Acquire a Molpro licence, or accept the ChS ceiling.** junChS-F12 is an order of magnitude cheaper than ChS for better accuracy, and its geometry route needs Molpro. Without it, `T3O-12h` (junChS in ORCA) is the ceiling.
5. **Run a grid-convergence line item** — one complex, DEFGRID2 against DEFGRID3, reporting ΔR and ΔB. **No published study quantifies grid effects on optimised *intermolecular distances***, and grid sensitivity is documented to be worst for noncovalent interactions ([Herbert](https://www.asc.ohio-state.edu/herbert.44/reprints/ARCC_20_1.pdf)). Until it exists, the DEFGRID3 requirement is asserted rather than demonstrated.
6. **Instrument the pipeline's provenance log** (§8A.5) from day one. Retro-fitting an audit trail after a campaign is how a guide's bias becomes a reported number.

---

## 22. Conference record and provenance of this revision

v4 incorporates the second eight-agent conference. Each agent's report is retained in `mm/conference2/` and the twenty adopted improvements are indexed in the "Changes in this revision" table at the head of this document.

| Agent | Scope | Principal contribution to v4 |
|---|---|---|
| 1 | GPU electronic structure | overturned Ruling 9; the crossover table, the fair-comparison protocol, the corrected routing rule, and 14 numbered corrections, all applied |
| 2 | Potential-energy surfaces and throughput | published point counts, PySCF's parallelism model, MPS, active learning, DVR-on-GPU measurements, execution recipes, the re-tiered Table 2 |
| 3 | Codes and acquisition | the ORCA-versus-CFOUR acquisition and capability tables, the ZMAT worked example, and the two ten-tier tracks |
| 4 | Conformer and isomer search | the GOAT-versus-CREST verdict and comparison table, the six-step union protocol, and the MLFF-GOAT recipe |
| 5 | Integration and adversarial audit | named the spec-to-conclusion failure and found fifteen further instances; the six over-corrections walked back; the information architecture, the decision card, and eighteen internal inconsistencies now fixed |
| 6 | State persistence and job chaining | the master state inventory, the canonical pipeline and drivers, the HDF5 store, and dangerous-reuse rules D1–D5 |
| 7 | Composite and combined methods | the frozen-monomer verdict, the propagation arithmetic, ChS/junChS/junChS-F12, the R1–R9 menu, and twelve protocol rules |
| 8 | Heterogeneous orchestration | the contention budget, scout-and-anchor, the Parsl configuration, the costed nine-atom example, and integrity guards G1–G7 |

**The chair's own position, recorded.** v3's GPU ruling was mine and it was wrong. The error was not a missing citation; it was a method — dividing a specification and emitting a conclusion about application performance without checking a benchmark. The provenance-tagging rule of §12.5 exists because that method is repeatable, and because a wrong number that arrives with arithmetic attached is harder to dislodge than a wrong number that arrives as an opinion. Readers of v3 should treat every `[D]`-tagged value in this revision as a hypothesis about their own machine until they have measured it.

---

## Appendix A. Large-amplitude-motion integrations, retained from v3

The tier tables of §13 and §14 were rebuilt for v4 and their superseded rows removed. The methodological prose beneath those tables was not superseded and is retained here in full: the two-stage deduplication protocol, the corrected DVR treatment, the salvaged path-integral protocol, and the internal-rotation and tunnelling discussion. Row IDs in §13–§14 point back to these subsections.

### A.1 Conformer search, global optimisation and the two-stage deduplication protocol

The search problem for a shallow intermolecular surface is finding the true global minimum among near-degenerate conformers, and the pruning problem is doing it without merging genuinely distinct species. The LAM document's "Spectroscopic Override" — merge if energies agree to 0.5 kcal/mol and A, B, C agree to 1.5 % — is **CREST's CREGEN with the RMSD test deleted and the remaining threshold loosened**, and it is deleted here.

**CREGEN already deduplicates on rotational constants.** Verified defaults: `--ethr` **0.05 kcal/mol**; `--bthr` **0.01 (= 1 %)**, "dynamically adjusted between this value and 2.5%, based on an anisotropy of the rotational constants in the ensemble"; `--rthr` **0.125 Å**; `--ewin` "depending on the application (e.g., 6 kcal/mol conformational searches …)" ([CREST keyword documentation](https://crest-lab.github.io/crest-docs/page/documentation/keywords.html)). CREGEN classifies pairs sequentially on energy, rotational-constant difference and Cartesian RMSD ([CREST, *J. Chem. Phys.* 160, 114110](https://pubs.aip.org/aip/jcp/article/160/11/114110/3278084/CREST-A-program-for-the-exploration-of-low-energy)), and the reason it uses rotational constants at all is that "rotational constants remain unaffected by atom permutations", so duplicates and rotamers show minimal |ΔB| while "true conformers should exhibit notable differences". The 1.5 % override sits inside CREST's own 1.0–2.5 % adaptive band, but CREST applies that band as one of three ANDed criteria precisely so that a loose rotational-constant window cannot merge distinct structures on its own. **Deleting the RMSD test and then loosening the remaining criterion is the worst of both worlds.**

Neither a single 1.5 % override nor a single 0.1 % threshold is defensible, because they answer different questions. A deduplication threshold must be **looser** than the method error of the level generating the ensemble, or genuine conformers are merged; a reporting threshold must be **tighter** than the experimental resolution, or distinct species are presented as one. Applying 0.1 % to a machine-learned ensemble whose own constants are wrong by 1–2 % is not rigour, it is noise amplification, and it will split one conformer into dozens of duplicates. Foundation models cannot rank these structures anyway: MACE-POLAR-1-L, -M and MACE-OMOL-0 give 3.52, 4.78 and 7.31 kcal/mol errors on S30L non-covalent interactions, orders of magnitude above a 0.05 kcal/mol dedup threshold ([MACE-POLAR-1](https://arxiv.org/html/2602.19411v1)).

#### The binding two-stage protocol

- **Stage A — generation-level deduplication, loose, on the machine-learned or semi-empirical ensemble.** CREGEN defaults exactly: `--ethr 0.05`, `--bthr 0.01` (dynamic to 0.025), `--rthr 0.125`, `--ewin 6.0`. Do not tighten. Purpose: remove genuine duplicates before expensive re-optimisation. Expected survivors: tens to low hundreds.
- **Stage B — reporting-level deduplication, tight, only after QM re-optimisation.** Re-optimise every Stage A survivor at the tier's QM level, then merge two structures only if **all three** hold: |ΔE| ≤ 0.05 kcal/mol **AND** |ΔA|/A, |ΔB|/B, |ΔC|/C ≤ 0.1 % **AND** heavy-atom RMSD ≤ 0.125 Å. **The AND is essential — an OR would merge distinct conformers that happen to be isoenergetic, and this is the single most common implementation error.**
- **Stage C — degeneracy bookkeeping.** Collapse enantiomers and equivalent rotamers, recording g_i via `CONFDEGEN AUTO`, which detects and records conformer degeneracy from RMSD and writes the full rotamer ensemble ([ORCA GOAT manual](https://www.faccts.de/docs/orca/6.0/manual/contents/typical/GOAT.html)). Enantiomers have identical inertia tensors and identical |μ_α| and are indistinguishable in an achiral microwave experiment; CREST already treats them as "a special case of rotamers" ([CREST docs, context](https://crest-lab.github.io/crest-docs/page/overview/context.html)). **g_i enters the Boltzmann weight, not the line list.**
- **Stage D — spectroscopic annotation.** Tag every surviving conformer with signed μ_a, μ_b, μ_c in the principal axis system and flag any component with |μ_α| < 0.1 D as **dark** in that selection rule. Dark conformers are excluded from the predicted line list and retained in the thermodynamic ensemble. If two structures agree on A, B, C to 0.1 % but differ in dipole components by more than the method's own dipole error, that is a signal that the geometries are not in fact the same — investigate rather than merge.
- **Stage E — audit log.** Record, for every merge, which criterion fired and the pre-merge values. This log is a deliverable.

A worked warning on why 1.5 % is too loose: in the 4-fluorothreonine study "the rotational constants of several conformers are quite similar one another" and assignment depended on predictions accurate to about 0.3 % — 8.2, 3.2 and 3.2 MHz in A₀, B₀ and C₀ ([Bringing machine-learning enhanced quantum chemistry to rotational spectroscopy](https://iris.unibs.it/retrieve/5d344a42-4240-478d-8312-08caabc252c7/Chemistry%20A%20European%20J%20-%202023%20-%20Barone.pdf)). A 1.5 % merge window would have collapsed that conformer family into one structure. At the other end, CP-FTMW line accuracy is about 2 kHz ([PMC9105391](https://pmc.ncbi.nlm.nih.gov/articles/PMC9105391/)), so 1.5 % of a 3 GHz constant is 45 MHz — over twenty thousand times the measurement precision.

**Search engines.** GOAT is a basin-hopping/minima-hopping hybrid needing no metadynamics and therefore far fewer gradients than CREST, costs about 100 × N_atoms geometry optimisations, runs 8 workers by default, offers `GOAT-ENTROPY` until the conformational entropy converges below 0.1 cal/(mol·K), and — decisively here — **works with any ORCA method including `!ExtOpt` external engines and DFT**, which matters on a shallow surface where semi-empirical ordering is unreliable ([ORCA GOAT manual](https://www.faccts.de/docs/orca/6.0/manual/contents/typical/GOAT.html); [de Souza, *Angew. Chem.* 2025](https://pubmed.ncbi.nlm.nih.gov/39959942/)). CREST's iMTD-GC metadynamics explores flat surfaces more aggressively and is the right cross-validation reference; run both and take the union ([CREST, *JCP* 160, 114110](https://pubs.aip.org/aip/jcp/article/160/11/114110/3278084/CREST-A-program-for-the-exploration-of-low-energy)). ABCluster's artificial-bee-colony swarm optimisation is the right tool for rigid-monomer packing, benchmarked on TIP4P water clusters to N = 20 with all global minima located and applied to microhydration, methanol microsolvation, nonpolar and ion–aromatic clusters ([ABCluster, *PCCP* 2016](https://pubmed.ncbi.nlm.nih.gov/26738568/); [*PCCP* 2015](https://pubmed.ncbi.nlm.nih.gov/26327507/); [ABCluster theory docs](https://zhjun-sci.com/abcluster/doc/theory.html)). Thermal molecular dynamics is **not** recommended as a primary search: at jet temperatures no barrier is crossed, and at temperatures high enough to cross barriers the complex dissociates.

**Table 1 — Conformer search and global optimisation**

### A.2 Intermolecular potential surfaces, scanning, and the corrected DVR

This is the flagship high-tier route of the document, and the LAM document's Methodology 3 is retained here with five corrections.

**Correction 1 — the citation.** Colbert and Miller is *J. Chem. Phys.* **96**(3), 1982–1991, 1 February **1992** ([Colbert & Miller, full text](https://scispace.com/pdf/a-novel-discrete-variable-representation-for-quantum-3r3q5zgsqw.pdf)). Widely used secondary notes render the year as "1982", which is the starting page number ([MCTDH numerical methods notes](https://www.pci.uni-heidelberg.de/tc/usr/mctdh/lit/NumericalMethods.pdf)); anyone copying that produces a broken reference.

**Correction 2 — the Cartesian sinc form is wrong for Jacobi coordinates.** The three verified kinetic-energy matrices are: on the infinite interval with grid x_i = iΔx, T_ii′ = (ħ²/2mΔx²) × {π²/3 for i = i′; 2(−1)^(i−i′)/(i−i′)² otherwise}; on the radial half-line with r_i = iΔr and r = 0 excluded, T_ii′ = (ħ²/2mΔr²) × {π²/3; (−1)^(i−i′)[1/(i−i′)² − 1/(i+i′)²]}; and on a finite box built on a sine basis with N − 1 interior points. The potential is diagonal and the quadrature weights are uniform.

In Jacobi coordinates the atom–rigid-rotor Hamiltonian is

H = −(ħ²/2μ) R⁻¹ (∂²/∂R²) R + ħ² l̂²/(2μR²) + V(R, r, θ) + H_monomer,  μ = m_A m_BC/(m_A + m_BC),

with ψ = r⁻¹R⁻¹ Σ φ_vj(r) Φ_a(R̂, r̂) χ_va(R) ([Hutson, *An introduction to the dynamics of van der Waals molecules*](https://durham-repository.worktribe.com/OutputFile/1172254)). **The R⁻¹(∂²/∂R²)R sandwich is exactly the substitution that converts the R² dR volume element into a flat dR one.** Make the substitution and solve for χ = Rψ, and the radial half-line form is legitimate. Omit it and you are solving a Hamiltonian missing a −ħ²/(μR)∂/∂R term. **The angular problem is not Cartesian at all:** the angular kinetic energy is the centrifugal operator ħ²l̂²/(2μR²), diagonal in associated Legendre functions, with the potential expanded as V = Σ_λ V_λ(R,r)P_λ(cos θ) and V_λ evaluated by Gauss–Legendre quadrature needing at least λ + 1 points, only even λ surviving for a homonuclear partner ([Hutson](https://durham-repository.worktribe.com/OutputFile/1172254)). A sinc grid in θ has the wrong weight, the wrong boundary behaviour at the poles, and destroys the l(l+1) eigenvalues. For two polyatomic fragments the canonical formulation is Ĥ = Ĥ_A + Ĥ_B + Ĥ_INT with the G-matrix G_ij^F = δ_ij/m_i − 1/M_F, diagonal in Jacobi coordinates, and an angular basis of coupled **Wigner D-functions** ([Brocks, van der Avoird, Sutcliffe & Tennyson](https://www.ucl.ac.uk/mathematical-physical-sciences/sites/mathematical_physical_sciences/files/22.pdf)) — which is what the current state of the art for the water dimer uses ([Wang, Yang, Carrington & Zhang, *JCP* 163, 144308 (2025)](https://pubmed.ncbi.nlm.nih.gov/41070798/)).

**Correction 3 — never build the dense Hamiltonian.** With N points per dimension in f dimensions the direct-product basis has N^f functions:

**A three-dimensional sinc-DVR at 40 points per dimension is 64,000 basis functions and a 32.8 GB dense Hamiltonian, which exceeds both the RTX 3090's 24 GB and, once eigenvectors and workspace are added, the workstation's 64 GB.** Full diagonalisation would be about 2.6 × 10¹⁵ flop, roughly five hours at a realistic sustained 150 GFLOPS FP64, and the GPU cannot help because its FP64 rate is 0.556 TFLOPS and the matrix does not fit in 24 GB. The Hamiltonian is a sum of Kronecker products plus a diagonal potential, so a matrix–vector product costs f·N^(f+1) operations and needs only a handful of vectors: **matrix-free Lanczos or Davidson is mandatory practice above two dimensions**, and the field's methodological literature is organised around exactly that point ([Carrington, *Iterative methods for computing vibrational spectra*, *Mathematics* 6, 13 (2018)](https://www.mdpi.com/2227-7390/6/1/13)).

**Correction 4 — direct-product DVR is limited to about five or six degrees of freedom**, with effort ∝ f·N^(f+1), and the wall is vector storage rather than flops ([MCTDH review, Heidelberg](https://www.pci.uni-heidelberg.de/tc/usr/mctdh/lit/rev.pdf)). Beyond that: MCTDH for "typically four to twelve" degrees of freedom, with multi-layer variants further ([MCTDH review](https://www.pci.uni-heidelberg.de/tc/usr/mctdh/lit/rev.pdf); [multilayer MCTDH](https://pubs.acs.org/doi/10.1021/acs.jpca.5b03256)); MULTIMODE for full-dimensional VSCF/VCI ([CentAUR](https://centaur.reading.ac.uk/11699/)); diffusion Monte Carlo, whose cost is dimension-agnostic, above about six degrees of freedom (Section 13.4).

**Correction 5 — "constrained monomer relaxation at each grid point" is deleted.** It rested on analytic MPQC CCSD(T)-F12 gradients, which ORCA 6.1 does not have, and the SAPT surface developers reject gradient-based grid construction on principle ([Metz & Szalewicz](https://par.nsf.gov/servlets/purl/10194876)). Use rigid-monomer single points, or the flexible-monomer autoPES protocol, or move the relaxation to CFOUR or Molpro where analytic derivatives exist.

**The Δ-learning route, which is the highest-value use of this tier.** For a 5–10 atom complex the foundation-model paradigm is the wrong tool: the task is not generalising across chemical space but fitting one surface to spectroscopic accuracy. Published accuracies in the units that matter:

For scale, the CO₂–H₂O authors note that generic force fields give RMS errors of "tens to hundreds of cm⁻¹" and that "even a 10 cm⁻¹ RMS fitting error would be considered large in the long-range part of the potential". **That is the yardstick for this tier — cm⁻¹, not kcal/mol.** Reference-point counts: 10⁴–5 × 10⁴ CCSD(T)-F12 points for a rigid- or semi-rigid-monomer dimer; a four-body water term needed only 2,119 symmetry-unique points ([arXiv:2107.05881](http://arxiv.org/abs/2107.05881)).

**Table 2 — Intermolecular potential surface construction and scanning**

**A note on JAX, correcting the LAM document.** JAX and the GPU are retained for **DVR Hamiltonian diagonalisation and search-space pruning**, which are genuinely dense linear-algebra and combinatorial problems, and are **not** used to replace SPFIT. A single SPFIT least-squares fit already costs about 20–30 ms — AUTOFIT evaluates 35–50 triples per second per core — and the eleven hours quoted in the LAM document is 10⁷ of those fits, with the workload I/O-bound and weak-scaling efficiency around 0.2 at eight cores ([HS-AUTOFIT](https://cris.unibo.it/retrieve/e1dcb339-6b53-7715-e053-1705fe0a6cc9/electronics-10-02251-v2.pdf); [pategroup AUTOFIT](https://github.com/pategroup/bband_scripts/tree/master/autofit)). **"Hours to milliseconds" conflates the combinatorial triples search with the diagonalisation, and the correct statement is that autodiff reduces the per-fit cost of an already-millisecond operation.** The effort belongs in pruning: narrowing the window from ±120 MHz to ±12 MHz removes about three orders of magnitude of candidate triples, which is the same factor, obtained by improving the chemistry rather than the linear algebra. Dipole-based elimination of dark transitions and χ-tensor pre-filtering do the same job. Note also that the RTX 3090 contributes little to the DVR side in double precision: its FP64 rate is 0.556 TFLOPS and Lanczos on a Kronecker-structured Hamiltonian is bandwidth-bound on small vectors.

### A.3 Vibrational averaging: from B_e to B₀, and the salvaged path-integral protocol

This is where the accuracy actually comes from, and it is the highest accuracy-per-CPU-hour item in the document.

**The salvaged dynamical protocol, replacing LAM Methodology 2.** Classical molecular dynamics at the experimental temperature is rejected. At 5 K, k_BT = 3.48 cm⁻¹, so classical equipartition puts 1.74 cm⁻¹ into each mode, against zero-point energies of 15 cm⁻¹ for a 30 cm⁻¹ intermolecular bend, 100 cm⁻¹ for a 200 cm⁻¹ libration and 1500 cm⁻¹ for a 3000 cm⁻¹ X–H stretch. **A classical trajectory at 5 K is a frozen structure rattling in the harmonic bottom of the well and returns essentially B_e, not B₀.** The zero-point elongation it misses is large: ΔR₀ᵉ = 0.361 Å for CH₃⁺–He (1.817 → 2.178 Å), 0.155 Å for CH₃⁺–Ne, 0.038 Å for CH₃⁺–Ar, 0.025 Å for CH₃⁺–Kr ([vibrational analysis of methyl cation–rare gas complexes, arXiv:2009.05443](https://arxiv.org/pdf/2009.05443.pdf)) — for a pseudo-diatomic with B ∝ R⁻², the He case puts B₀ roughly 30 % below B_e. Running the trajectory hotter to mimic zero-point energy is an uncontrolled fudge, because every mode then gets the same k_BT regardless of ω whereas the true zero-point gives every mode ħω/2. And at 5 K the classical barrier-crossing rate between equivalent minima is negligible — a 100 cm⁻¹ barrier at k_BT = 3.48 cm⁻¹ is a Boltzmann factor of e^(−28.7) ≈ 3 × 10⁻¹³ — so the trajectory is trapped in one well and samples a symmetry-broken distribution while the true ground state is delocalised over all equivalent wells.

The full-dimensional path-integral fix is correct in principle and unaffordable in practice at jet temperature (Section 5.4). **The restricted form that survives, and is binding on every dynamical-averaging row:**

Mitigations that reduce the bead count further, in order of preference: GLE-thermostatted path integrals (PIGLET), which converged the same quantities with 8 replicas where standard PIMD needed 64 ([FHI-aims PIMD tutorial](https://nomad-laboratory.de/meetings/FHI-aims-2014/uploads/Meeting/Tutorial-PIMD.pdf)); ring-polymer contraction, computing fast forces on all beads and slow forces on a subset ([OpenMM RPMD plugin](https://docs.openmm.org/latest/userguide/library/09_rpmd_plugin.html)); and the rigid-monomer constraint itself. Production implementations are available in [i-PI](https://docs.ipi-code.org/features.html) with worked examples in the [atomistic cookbook path-integrals tutorial](https://atomistic-cookbook.org/examples/path-integrals/path-integrals.html), and rigid-rotor path-integral formulations exist for exactly this purpose ([path-integral simulations of rigid rotors](https://www.lmp.uni-saarland.de/wp-content/uploads/2017/09/JPhyCoMa99.pdf)).

**Diffusion Monte Carlo** is the dimension-agnostic alternative above about six degrees of freedom. It gives an essentially unbiased ground-state energy and, via descendant weighting, an unbiased sampling of |Ψ₀|² from which ⟨μ_αα⟩ — the right quantity — can be accumulated with no basis, no coordinate choice and no kinetic-energy-operator derivation. Statistical scaling, from H₉O₄⁺: guided DMC gives σ(E_ref) ≈ 560 cm⁻¹ at 1,000 walkers falling to ≈ 90 cm⁻¹ at 40,000, consistent with N_w^(−1/2); unguided DMC has roughly twice the guided σ at the same ensemble size; a single unguided run with 10⁶ walkers gave E₀ = 23,405 cm⁻¹, within 1 cm⁻¹ of the guided 40,000-walker result; a time step of 1.0 a.u. delivers 5–10 cm⁻¹ accuracy with at least 10,000 steps, and descendant weighting needs τ_DW ≈ 250 a.u.; guided DMC makes systems up to about 20 atoms feasible ([McCoy review, NSF PAR](https://par.nsf.gov/servlets/purl/10348784)). Excited states require a nodal surface: the practical fix builds nodes from asymmetric-top rigid-rotor wavefunctions evaluated with the system's own averaged constants, validated against converged variational results for H₂D⁺, and the results are **sensitive to the embedding scheme** — the DMC analogue of the Eckart problem, and it must be reported rather than hidden ([Petit, Wellen & McCoy, *JCP* 138, 034105 (2013)](https://pubs.aip.org/aip/jcp/article/138/3/034105/192485/Using-fixed-node-diffusion-Monte-Carlo-to); [Petit & McCoy, *JCP* 136, 074101 (2012)](https://pubs.aip.org/aip/jcp/article/136/7/074101/190734/Unraveling-rotation-vibration-mixing-in-highly); [McCoy group DMC pages](https://mccoygroup.github.io/References/References/Monte%20Carlo%20Methods/DMC.html)). A 40,000-walker, 10,000-step run is 4 × 10⁸ potential calls, so it requires a fitted surface or a machine-learned potential and is embarrassingly parallel.

**Table 4 — Vibrational averaging: B₀, ΔB_vib and the semi-experimental route**

### A.4 Internal rotation, tunnelling splittings and large-amplitude observables

**A methyl rotor splits every line into A and E components, doubling the line count and defeating naive pattern matching unless it is anticipated.** That is the practical reason this table exists.

**Axis systems and programs.** The three standard frames for a methyl-top Hamiltonian are the principal axis method (PAM, the rotor treated as a perturbation in the principal inertial frame), the internal axis method (IAM, frame rotated so the rotor axis coincides with a coordinate axis, removing the leading Coriolis coupling), and the rho axis method (RAM, frame aligned with the ρ vector, which is the workhorse for low barriers). The programs an experimentalist will actually use are catalogued and distributed on the [Kisiel internal-rotation programs page](http://info.ifpan.edu.pl/~kisiel/introt/introt.htm):

A three-program cross-comparison on common test systems is available in the [RWTH Aachen group's publication record](https://www.chemie.rwth-aachen.de/cms/chemie/forschung/uebersichtsseite-publikationen/publikationen/~mdfct/details/?file=64088&lidx=1), and the pairwise aixPAM/XIAM comparison for the methylanisoles is in [Ferres *et al.*](https://hal.science/hal-03183072/document).

**How V₃ is obtained and how good it is.** Relaxed torsional scan on a 10° grid with all other coordinates optimised, as in the [m-methylanisole study](https://hal.science/hal-03183072/document); fit to V = Σ_n (V_3n/2)(1 − cos 3nτ); combine V₃ with the internal-rotation constant F (F₀ ≈ 158–161 GHz for a methyl top — 159.808(80) GHz fitted for 2-methylthiophene, 160.898 and 160.855 GHz calculated for the two m-methylanisole conformers) to form the reduced barrier s = 4V₃/(9F). The A–E splitting is a steep function of s, collapsing towards zero for high barriers and reaching GHz scale for low ones.

Values from the [2-methylthiophene study, *Spectroscopy Journal* 1(1):5](https://www.mdpi.com/2813-446X/1/1/5) and the [m-methylanisole study](https://hal.science/hal-03183072/document); the 2-methylthiophene authors summarise the situation as "many levels of theory yield values with less than 10 cm⁻¹ deviation". **The document's position: ±10 cm⁻¹ or roughly 10 % on V₃ is what a relaxed DFT scan buys, which is ample for identifying the torsional regime and starting a fit, and insufficient for predicting a splitting pattern blind**, because a low barrier amplifies a 10 % error in V₃ into an order-of-magnitude error in the splitting. Anchors across the regime: V₃ = 11.21745(2) cm⁻¹ ([Obenchain *et al.*](https://bib-pubdb1.desy.de/record/454603/files/The%20low%20barrier%20methyl%20internal%20rotation%20in%20the_Obenchain_submission.pdf)); 36.63 and 55.77 cm⁻¹ for the methylanisoles; 49.374548(1) cm⁻¹ for p-methylanisole ([HAL hal-03183097](https://hal.science/hal-03183097v1/document)); 106.4456(8) cm⁻¹ for 4-methyl-2-nitrophenol ([*Molecules* 28, 2153](https://www.mdpi.com/1420-3049/28/5/2153)); 151.492(34) cm⁻¹ for vinyl acetate, fitted with XIAM, ERHAM and BELGI-C_s together ([Nguyen *et al.*](https://hal.science/hal-03183158/document)); 195.18(7) cm⁻¹ for ammonia–formic acid against a computed span of 168.3–212.8 cm⁻¹, a ±14 % spread ([Roehling, Hill, Daly & Kukolich](https://experts.arizona.edu/en/publications/ammonia-formic-acid-complex-internal-rotation-analysis-calculatio/)); ≈439.15 cm⁻¹ for the two equivalent tops of 2,5-dimethylfuran ([HAL hal-03183074](https://hal.science/hal-03183074/document)); and 565.1(5) cm⁻¹, where splittings of only about 51 MHz were estimated and B3LYP-D3(BJ)/def2-TZVP underestimated the barrier ([Semantic Scholar PDF](https://pdfs.semanticscholar.org/0be3/8225f41fd1700efaa6b69c19ef4e763d2784.pdf)). For scale on what a good fit achieves: 2,4-dimethylpyrrole was fitted with XIAM and BELGI-C_s-2Tops-hyperfine over 1,561 lines, "both achieved measurement accuracy of 4 kHz" ([HAL hal-04745405](https://hal.science/hal-04745405v1/document)).

**The binding protocol: compute V₃ from a relaxed scan at the best affordable level, convert to a predicted splitting to establish the order of magnitude, then fit V₃ to the observed splittings with XIAM — or BELGI/RAM36 if s is small — rather than trusting the computed value.** The computed number's job is to start the fit and distinguish conformers, not to predict the spectrum.

**Tunnelling splittings are the most under-served observable in the document, and the honest entry is that no tier reliably delivers them de novo.** Ar–ketene: "Tunneling of the hydrogen or deuterium atoms splits the a- and b-type rotational transitions … into two states", and "this internal motion appears to be quenched for HDCCO-Ar where only one state is observed" ([NIST / Gillies *et al.*](https://www.nist.gov/publications/rotational-spectra-structure-internal-dynamics-and-electric-dipole-moment-argon-0)). The water dimer spans nine orders of magnitude in one molecule: the largest 1→4 splitting is −70,128.436 ± 0.10 MHz, ground-state acceptor switching is 279,650 MHz for (H₂O)₂ against 53,000 MHz for (D₂O)₂, K = 0 donor–acceptor interchange is 19,526.73 against 1,172.23 MHz, and K = 0 donor tunnelling is 6 MHz ([Mukhopadhyay, Cole & Saykally](https://escholarship.org/content/qt1j70w8wt/qt1j70w8wt.pdf)). **Required accuracy is a factor of three — the correct order of magnitude — and even that is not routinely met.** Any method claiming to predict tunnelling must reproduce the (H₂O)₂/(D₂O)₂ ratio, which is the reason the water dimer is system 4 of the validation set in Section 17.

**Table 7 — Internal rotation, tunnelling splittings and large-amplitude observables**

---

## 23. References

Every reference below is cited inline in the body of this document and is listed with a resolvable URL. The list is rebuilt mechanically from the inline citations at each revision, so an entry exists here if and only if it is used above; the ordering is alphabetical by anchor text so that an inline citation can be located directly. Sources added in v4 are predominantly the GPU, concurrency, state-persistence, composite-method and conformer-search literature of the second conference.

1. [2-methylthiophene study, Spectroscopy Journal 1(1):5](https://www.mdpi.com/2813-446X/1/1/5) — mdpi.com
2. [`oet_aimnet2` readme](https://raw.githubusercontent.com/faccts/orca-external-tools/main/readmes/aimnet2.md) — raw.githubusercontent.com
3. [`oet_mace` readme](https://raw.githubusercontent.com/faccts/orca-external-tools/main/readmes/mace.md) — raw.githubusercontent.com
4. [ABCluster theory docs](https://zhjun-sci.com/abcluster/doc/theory.html) — zhjun-sci.com
5. [ABCluster, PCCP 2016](https://pubmed.ncbi.nlm.nih.gov/26738568/) — pubmed.ncbi.nlm.nih.gov
6. [ACEsuit/mace](https://github.com/acesuit/mace) — github.com
7. [ACEsuit/mace-off](https://github.com/ACEsuit/mace-off) — github.com
8. [acetylene semi-experimental structure study](https://pubmed.ncbi.nlm.nih.gov/21322673/) — pubmed.ncbi.nlm.nih.gov
9. [active-learning PES](https://chemrxiv.org/engage/chemrxiv/article-details/675b9e3bf9980725cfe8476a) — chemrxiv.org
10. [AiiDA 1.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC7479590/) — pmc.ncbi.nlm.nih.gov
11. [Alessandrini & Puzzarini, JPCA 2021](https://cris.unibo.it/retrieve/handle/11585/868614/e1dcb339-596b-7715-e053-1705fe0a6cc9/acs.jpca.1c07828_templaggio.pdf) — cris.unibo.it
12. [Alessandrini & Puzzarini, Lego-brick](https://ricerca.sns.it/retrieve/fe3d5821-f41e-48ed-927f-34be686e050b/acs.jpca.1c07828.pdf) — ricerca.sns.it
13. [Alessandrini et al., PCCP 2023](https://pubs.rsc.org/en/content/articlepdf/2023/cp/d3cp03984f) — pubs.rsc.org
14. [Altun, Neese & Bistoni](https://pmc.ncbi.nlm.nih.gov/articles/PMC7586325/) — pmc.ncbi.nlm.nih.gov
15. [Andrews, Taleb-Bendiab, LaBarge, Hillig & Kuczkowski, JCP 85, 3180](https://pubs.aip.org/aip/jcp/article/85/6/3180/219289/Rotational-spectrum-H-19F-spin-spin-and-D-nuclear) — pubs.aip.org
16. [artifact retention](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/remove-workflow-artifacts) — docs.github.com
17. [arXiv:1704.02697](https://arxiv.org/pdf/1704.02697) — arxiv.org
18. [arXiv:2107.05881](http://arxiv.org/abs/2107.05881) — arxiv.org
19. [arXiv:2410.12603](https://arxiv.org/abs/2410.12603) — arxiv.org
20. [arXiv:2508.03405](https://arxiv.org/html/2508.03405v1) — arxiv.org
21. [arXiv:2508.19118](https://arxiv.org/abs/2508.19118) — arxiv.org
22. [Ar₂ benchmark potential](https://pubmed.ncbi.nlm.nih.gov/20831315/) — pubmed.ncbi.nlm.nih.gov
23. [ASE database documentation](https://wiki.fysik.dtu.dk/ase/ase/db/db.html) — wiki.fysik.dtu.dk
24. [ASE optimize docs](https://ase-lib.org/_sources/ase/optimize.rst) — ase-lib.org
25. [ASE Trajectory](https://wiki.fysik.dtu.dk/ase/ase/io/trajectory.html) — wiki.fysik.dtu.dk
26. [atomistic cookbook path-integrals tutorial](https://atomistic-cookbook.org/examples/path-integrals/path-integrals.html) — atomistic-cookbook.org
27. [AUTOCI](https://orca-manual.mpi-muelheim.mpg.de/_sources/contents/modelchemistries/autoci.md) — orca-manual.mpi-muelheim.mpg.de
28. [autoPES manual](https://www.physics.udel.edu/~szalewic/SAPT/autoPES_manual.pdf) — physics.udel.edu
29. [benchmarks README](https://raw.githubusercontent.com/pyscf/gpu4pyscf/master/benchmarks/README.md) — raw.githubusercontent.com
30. [Bologna 74-isotopologue benchmark](https://cris.unibo.it/bitstream/11585/656295.2/1/Benchmark_paperS.pdf) — cris.unibo.it
31. [Bootsma & Wheeler](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60c74474ee301c02d6c7916e/original/popular-integration-grids-can-result-in-large-errors-in-dft-computed-free-energies.pdf) — chemrxiv.org
32. [BrianQC](https://www.brianqc.com/benchmarks) — brianqc.com
33. [Bringing machine-learning enhanced quantum chemistry to rotational spectroscopy](https://iris.unibs.it/retrieve/5d344a42-4240-478d-8312-08caabc252c7/Chemistry%20A%20European%20J%20-%202023%20-%20Barone.pdf) — iris.unibs.it
34. [Brocks, van der Avoird, Sutcliffe & Tennyson](https://www.ucl.ac.uk/mathematical-physical-sciences/sites/mathematical_physical_sciences/files/22.pdf) — ucl.ac.uk
35. [BSSEOptimization.cmp](https://github.com/ORCAQuantumChemistry/CompoundScripts/blob/main/GeometryOptimization/BSSEOptimization.cmp) — github.com
36. [Bunker, The Molecular Symmetry Group: a personal view](https://d197for5662m48.cloudfront.net/documents/publicationstatus/231737/preprint_pdf/3347adf8e8b88c35e3b80cce4047b4fd.pdf) — d197for5662m48.cloudfront.net
37. [Burns, Marshall & Sherrill](https://pubs.acs.org/doi/10.1021/ct400149j) — pubs.acs.org
38. [Carrington, Iterative methods for computing vibrational spectra, Mathematics 6, 13 (2018)](https://www.mdpi.com/2227-7390/6/1/13) — mdpi.com
39. [Cartesian coordinates](https://cfour.uni-mainz.de/cfour/index.php?n=Main.UseOfCartesianCoordinates) — cfour.uni-mainz.de
40. [Cazzoli et al., D₂O Lamb-dip](https://hal.science/hal-00604410/document) — hal.science
41. [CCL mailing list, 6 July 2016](https://server.ccl.net/chemistry/resources/messages/2016/07/06.003-dir/index.html) — server.ccl.net
42. [CDMS mirror](https://cdms.astro.uni-koeln.de/classic/predictions/pickett/spinv.html) — cdms.astro.uni-koeln.de
43. [CentAUR](https://centaur.reading.ac.uk/11699/) — centaur.reading.ac.uk
44. [CFOUR analytic energy derivatives](https://cfour.uni-mainz.de/cfour/index.php?n=Main.AnalyticEnergyDerivatives) — cfour.uni-mainz.de
45. [CFOUR download](https://cfour.uni-mainz.de/cfour/index.php?n=Main.Download) — cfour.uni-mainz.de
46. [CFOUR home](https://cfour.uni-mainz.de/cfour/index.php?n=Main.HomePage) — cfour.uni-mainz.de
47. [CFOUR installation](https://cfour.uni-mainz.de/cfour/index.php?n=Main.Installation) — cfour.uni-mainz.de
48. [CFOUR licence](https://cfour.uni-mainz.de/cfour/index.php?n=Main.MainLicense) — cfour.uni-mainz.de
49. [CFOUR overview paper](https://par.nsf.gov/servlets/purl/10177577) — par.nsf.gov
50. [CFOUR restarts](https://cfour.uni-mainz.de/cfour/index.php?n=Main.Non-standardFileHandlingAndRestartCalculations) — cfour.uni-mainz.de
51. [CHANGELOG](https://raw.githubusercontent.com/pyscf/gpu4pyscf/master/CHANGELOG) — raw.githubusercontent.com
52. [ChemCompute](https://chemcompute.org/) — chemcompute.org
53. [chemeurope, Eckart conditions](https://www.chemeurope.com/en/encyclopedia/Eckart_conditions.html) — chemeurope.com
54. [Classroom + Codespaces](https://docs.github.com/en/education/manage-coursework-with-github-classroom/integrate-github-classroom-with-an-ide/using-github-codespaces-with-github-classroom) — docs.github.com
55. [Codespaces automatic deletion](https://docs.github.com/en/codespaces/setting-your-user-preferences/configuring-automatic-deletion-of-your-codespaces) — docs.github.com
56. [Codespaces billing](https://docs.github.com/en/billing/managing-billing-for-your-products/about-billing-for-github-codespaces) — docs.github.com
57. [Codespaces timeout](https://docs.github.com/en/codespaces/setting-your-user-preferences/setting-your-timeout-period-for-github-codespaces) — docs.github.com
58. [Colbert & Miller, full text](https://scispace.com/pdf/a-novel-discrete-variable-representation-for-quantum-3r3q5zgsqw.pdf) — scispace.com
59. [CompoundScripts EnergyExtrapolation](https://github.com/ORCAQuantumChemistry/CompoundScripts/tree/main/EnergyExtrapolation) — github.com
60. [conformal MLIP uncertainty](https://arxiv.org/html/2510.00721v1) — arxiv.org
61. [conformal prediction for molecular properties](https://arxiv.org/pdf/2304.00970.pdf) — arxiv.org
62. [conformer table, UVa](https://uvadoc.uva.es/bitstream/handle/10324/6112/TFG-G592.pdf?sequence=1) — uvadoc.uva.es
63. [Costain](https://pubs.aip.org/aip/jcp/article/29/4/864/205101/Determination-of-Molecular-Structures-from-Ground) — pubs.aip.org
64. [Course-QuantumChemistryLab](https://github.com/fevangelista/Course-QuantumChemistryLab) — github.com
65. [Covalent executors](https://docs.covalent.xyz/docs/features/executors/) — docs.covalent.xyz
66. [CREST docs, context](https://crest-lab.github.io/crest-docs/page/overview/context.html) — crest-lab.github.io
67. [CREST input files](https://crest-lab.github.io/crest-docs/page/documentation/inputfiles.html) — crest-lab.github.io
68. [CREST keyword documentation](https://crest-lab.github.io/crest-docs/page/documentation/keywords.html) — crest-lab.github.io
69. [CREST NCI example](https://crest-lab.github.io/crest-docs/page/examples/example_3.html) — crest-lab.github.io
70. [CREST workflows](https://crest-lab.github.io/crest-docs/page/overview/workflows.html) — crest-lab.github.io
71. [CREST, J. Chem. Phys. 160, 114110](https://pubs.aip.org/aip/jcp/article/160/11/114110/3278084/CREST-A-program-for-the-exploration-of-low-energy) — pubs.aip.org
72. [CSC ORCA docs](https://docs.csc.fi/apps/orca/) — docs.csc.fi
73. [Czakó, Mátyus and Császár, JPCA 113, 11665 (2009)](https://www2.sci.u-szeged.hu/czako/papers/JPCA_H2O_113_11665_2009.pdf) — www2.sci.u-szeged.hu
74. [de Souza, Angew. Chem. 2025](https://pubmed.ncbi.nlm.nih.gov/39959942/) — pubmed.ncbi.nlm.nih.gov
75. [deformation-energy assessment](https://pubs.aip.org/aip/jcp/article/158/24/244106/2899786/A-quantitative-assessment-of-deformation-energy-in) — pubs.aip.org
76. [del Río, Mortensen & Jacobsen](https://arxiv.org/pdf/1808.08588) — arxiv.org
77. [Demaison et al., JCP 154, 194302 (2021)](https://pubs.aip.org/aip/jcp/article/154/19/194302/565922/How-accurate-is-the-determination-of-equilibrium) — pubs.aip.org
78. [df_pyscf_qchem](https://raw.githubusercontent.com/pyscf/gpu4pyscf/master/benchmarks/df/df_pyscf_qchem.md) — raw.githubusercontent.com
79. [Dohmen, Fedosov & Obenchain, PCCP 2023](https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04067k) — pubs.rsc.org
80. [Eckart conditions](https://en.wikipedia.org/wiki/Eckart_conditions) — en.wikipedia.org
81. [ETH Zürich HPC docs](https://docs.hpc.ethz.ch/software/chemistry/orca/) — docs.hpc.ethz.ch
82. [External Methods — OPI 2.0 docs](https://www.faccts.de/docs/opi/2.0/docs/contents/notebooks/extopt.html) — faccts.de
83. [FACCTs](https://www.faccts.de/orca/) — faccts.de
84. [FACCTs LinkedIn, 2026-05-11](https://www.linkedin.com/posts/faccts_orca-xtb-gxtb-activity-7452978014499495937-tiY3) — linkedin.com
85. [faccts/opi](https://github.com/faccts/opi) — github.com
86. [faccts/orca-external-tools](https://github.com/faccts/orca-external-tools) — github.com
87. [facebook/OMol25](https://huggingface.co/facebook/OMol25) — huggingface.co
88. [Fajen et al.](https://www.arxiv.org/abs/2512.01055) — arxiv.org
89. [Fatima et al., electric dipole moments from Stark effect in supersonic expansion](https://pmc.ncbi.nlm.nih.gov/articles/PMC9961461/) — pmc.ncbi.nlm.nih.gov
90. [features](https://cfour.uni-mainz.de/cfour/index.php?n=Main.Features) — cfour.uni-mainz.de
91. [FHI-aims PIMD tutorial](https://nomad-laboratory.de/meetings/FHI-aims-2014/uploads/Meeting/Tutorial-PIMD.pdf) — nomad-laboratory.de
92. [focal-point study](https://par.nsf.gov/servlets/purl/10566327) — par.nsf.gov
93. [Fortenberry et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10537648/) — pmc.ncbi.nlm.nih.gov
94. [Fraser, Gillies, Zozom, Lovas & Suenram 1987](https://www.sciencedirect.com/science/article/pii/0022285287900919/pdf) — sciencedirect.com
95. [GA102 whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf) — nvidia.com
96. [Gaussian pricing](https://gaussian.com/pricing/) — gaussian.com
97. [GeForce RTX 30 series specifications](https://en.wikipedia.org/wiki/GeForce_RTX_30_series) — en.wikipedia.org
98. [GitHub Actions limits](https://docs.github.com/en/actions/reference/limits) — docs.github.com
99. [GitHub-hosted runners](https://docs.github.com/en/actions/using-github-hosted-runners/using-github-hosted-runners/about-github-hosted-runners) — docs.github.com
100. [Glick, Kumawat & Sherrill, J. Chem. Phys. 162, 174106 (2025)](https://pubs.aip.org/aip/jcp/article/162/17/174106/3345631/Evaluating-wavefunction-methods-the-counterpoise) — pubs.aip.org
101. [GNU parallel](https://www.gnu.org/software/parallel/man.html) — gnu.org
102. [Goodlett and Kitzmiller, J. Chem. Phys. 161, 024107 (2024)](https://pubs.aip.org/aip/jcp/article/161/2/024107/3302915/MolSym-A-Python-package-for-handling-symmetry-in) — pubs.aip.org
103. [Gordon et al., PCCP 2018](https://pubs.rsc.org/en/content/articlelanding/2018/cp/c8cp01102h) — pubs.rsc.org
104. [gpu4pyscf benchmarks](https://raw.githubusercontent.com/pyscf/gpu4pyscf/master/benchmarks/scf/scf_pyscf_qchem.md) — raw.githubusercontent.com
105. [gpu4pyscf README](https://github.com/pyscf/gpu4pyscf) — github.com
106. [GPW GPU implementation](https://arxiv.org/abs/2603.24881) — arxiv.org
107. [grimme-lab/g-xtb](https://github.com/grimme-lab/g-xtb) — github.com
108. [Groner](https://www.sciencedirect.com/science/article/abs/pii/S0022285216300959) — sciencedirect.com
109. [Guo, Riplinger, Becker, Liakos, Minenkov, Cavallo & Neese, JCP 148, 011101 (2018)](https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/1.5011798/13764376/011101_1_online.pdf) — pubs.aip.org
110. [Gutowsky, Arunan et al. 1995](https://pubs.aip.org/aip/jcp/article/103/10/3917/481163/Rotational-spectra-and-structures-of-the-C6H6-HCN) — pubs.aip.org
111. [h5py Datasets](https://docs.h5py.org/en/stable/high/dataset.html) — docs.h5py.org
112. [Hait & Head-Gordon, arXiv:1709.05075](https://arxiv.org/abs/1709.05075) — arxiv.org
113. [HAL hal-03183074](https://hal.science/hal-03183074/document) — hal.science
114. [HAL hal-03183097](https://hal.science/hal-03183097v1/document) — hal.science
115. [HAL hal-04745405](https://hal.science/hal-04745405v1/document) — hal.science
116. [Herbert](https://www.asc.ohio-state.edu/herbert.44/reprints/ARCC_20_1.pdf) — asc.ohio-state.edu
117. [Herbert, JCP 161, 054114](https://www.asc.ohio-state.edu/herbert.44/reprints/JCP_161_054114.pdf) — asc.ohio-state.edu
118. [Herbert, JCTC 18, 6742](https://www.asc.ohio-state.edu/herbert.44/reprints/JCTC_18_6742.pdf) — asc.ohio-state.edu
119. [HFIP⋯Rg](https://pubs.acs.org/doi/10.1021/acs.jpca.1c03757) — pubs.acs.org
120. [How to VPT2](https://par.nsf.gov/servlets/purl/10284833) — par.nsf.gov
121. [HS-AUTOFIT, Electronics 10, 2251](https://cris.unibo.it/retrieve/e1dcb339-6b53-7715-e053-1705fe0a6cc9/electronics-10-02251-v2.pdf) — cris.unibo.it
122. [Hutson, An introduction to the dynamics of van der Waals molecules](https://durham-repository.worktribe.com/OutputFile/1172254) — durham-repository.worktribe.com
123. [i-PI](https://docs.ipi-code.org/features.html) — docs.ipi-code.org
124. [initial guess](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/initialguess.html) — faccts.de
125. [Intel ARK, i7-13700K](https://ark.intel.com/content/www/us/en/ark/products/230500/intel-core-i7-13700k-processor-30m-cache-up-to-5-40-ghz.html) — ark.intel.com
126. [Intel Core i7-13700K datasheet](https://media.distrelec.com/Web/Downloads/_t/ds/BX8071513700K_eng_tds.pdf) — media.distrelec.com
127. [Intel oneMKL Developer Guide](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2023-0/managing-performance-with-heterogeneous-cores.html) — intel.com
128. [Intel support article 000089918](https://www.intel.com/content/www/us/en/support/articles/000089918/processors.html) — intel.com
129. [J. K. G. Watson, J. Mol. Spectrosc. 1973, journal listing](https://scispace.com/journals/journal-of-molecular-spectroscopy-2bqujokn/1973) — scispace.com
130. [J. Phys. Chem. A](https://pubs.acs.org/doi/10.1021/jp049955k) — pubs.acs.org
131. [JACS 2024](https://pubs.acs.org/doi/10.1021/jacs.4c07099) — pubs.acs.org
132. [JAX sharp bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) — docs.jax.dev
133. [JCP 161, 144120](https://pubs.aip.org/aip/jcp/article/161/14/144120/3316843/Assessment-of-the-similarity-transformed-equation) — pubs.aip.org
134. [JCP 162, 114112](https://pubs.aip.org/jcp/article/162/11/114112/3339921/On-the-applicability-of-CCSD-T-for-dispersion) — pubs.aip.org
135. [JPCA 2020](https://pubs.acs.org/doi/10.1021/acs.jpca.0c01376) — pubs.acs.org
136. [Judge et al.](https://arxiv.org/html/2409.07190v1) — arxiv.org
137. [jun-ChS](https://cris.unibo.it/retrieve/fdcbe2fd-290c-49a7-88cc-01c515d136bd/Extension%20of%20the%20%E2%80%9Ccheap%E2%80%9D%20composite%20approach.pdf) — cris.unibo.it
138. [junChS-F12](https://cris.unibo.it/retrieve/handle/11585/868585/ae4939e6-d216-426d-9d79-edb47b92c82c/junChS-F12.pdf) — cris.unibo.it
139. [Kesharwani et al., arXiv:2111.01882](https://arxiv.org/pdf/2111.01882) — arxiv.org
140. [Kesharwani, Martin et al., PCCP 2022](https://pubs.rsc.org/en/content/articlehtml/2022/cp/d2cp03938a) — pubs.rsc.org
141. [Kiran et al.](https://arxiv.org/html/2408.05148v3) — arxiv.org
142. [Kisiel et al., PCCP 2003](https://pubs.rsc.org/en/content/articlelanding/2003/cp/b212029a) — pubs.rsc.org
143. [Kisiel internal-rotation programs page](http://info.ifpan.edu.pl/~kisiel/introt/introt.htm) — info.ifpan.edu.pl
144. [Kisiel, CRIB sheet for SPFIT/SPCAT](http://info.ifpan.edu.pl/~kisiel/asym/pickett/crib.htm) — info.ifpan.edu.pl
145. [Kisiel, structural programs](http://info.ifpan.edu.pl/~kisiel/struct/struct.htm) — info.ifpan.edu.pl
146. [Kraka, Cremer, Spoerel, Merke, Stahl & Dreizler, J. Phys. Chem. 99, 12466 (1995)](https://s3.smu.edu/dedman/catco/publications/pdf/JPhysChem_99_12466_1995.pdf) — s3.smu.edu
147. [larger runners](https://docs.github.com/en/actions/using-github-hosted-runners/using-larger-runners/about-larger-runners) — docs.github.com
148. [later r_m^(1)/r_m^(2) refinements](https://pubmed.ncbi.nlm.nih.gov/11336516/) — pubmed.ncbi.nlm.nih.gov
149. [LEDE-CREST](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/646b83b4b3dd6a65308e7595/original/a-variant-on-the-crest-algorithm-for-non-covalent-clusters-of-flexible-molecules.pdf) — chemrxiv.org
150. [Lee & McCarthy, Bayesian analysis](https://par.nsf.gov/servlets/purl/10149706) — par.nsf.gov
151. [Lee & McCarthy, JPCA 2020](https://pubmed.ncbi.nlm.nih.gov/31910016/) — pubmed.ncbi.nlm.nih.gov
152. [Li, Zhang, Sun & Chan](https://arxiv.org/html/2407.09700v1) — arxiv.org
153. [LibintX](https://arxiv.org/html/2405.01834v2) — arxiv.org
154. [Longuet-Higgins, Mol. Phys. 6, 445 (1963)](https://ui.adsabs.harvard.edu/abs/1963MolPh...6..445L/abstract) — ui.adsabs.harvard.edu
155. [m-methylanisole study](https://hal.science/hal-03183072/document) — hal.science
156. [MACE documentation](https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html) — mace-docs.readthedocs.io
157. [MACE-OFF](https://pmc.ncbi.nlm.nih.gov/articles/PMC12123624/) — pmc.ncbi.nlm.nih.gov
158. [MACE-POLAR-1](https://arxiv.org/html/2602.19411v1) — arxiv.org
159. [Maya HTT knowledge base](https://help.mayahtt.com/kb/topics/how_to_resolve_intel_mpi_performance_issues_on_windows_with_hybrid_cpu_architectures.html) — help.mayahtt.com
160. [McCoy group DMC pages](https://mccoygroup.github.io/References/References/Monte%20Carlo%20Methods/DMC.html) — mccoygroup.github.io
161. [McCoy review, NSF PAR](https://par.nsf.gov/servlets/purl/10348784) — par.nsf.gov
162. [McNaughton et al., PCCP 2017](https://pubs.rsc.org/en/content/articlelanding/2017/cp/c6cp07487a) — pubs.rsc.org
163. [MCTDH numerical methods notes](https://www.pci.uni-heidelberg.de/tc/usr/mctdh/lit/NumericalMethods.pdf) — pci.uni-heidelberg.de
164. [MCTDH review, Heidelberg](https://www.pci.uni-heidelberg.de/tc/usr/mctdh/lit/rev.pdf) — pci.uni-heidelberg.de
165. [Melli et al., PubMed 36149341](https://pubmed.ncbi.nlm.nih.gov/36149341/) — pubmed.ncbi.nlm.nih.gov
166. [metal complexes](https://www.brianqc.com/metal-complexes) — brianqc.com
167. [Metz & Szalewicz](https://par.nsf.gov/servlets/purl/10194876) — par.nsf.gov
168. [MIG supported GPUs](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html) — docs.nvidia.com
169. [model card](https://huggingface.co/isayevlab/aimnet2-nse) — huggingface.co
170. [model-agnostic conformal QSAR](https://pmc.ncbi.nlm.nih.gov/articles/PMC13390030/) — pmc.ncbi.nlm.nih.gov
171. [molecular geometry input](https://cfour.uni-mainz.de/cfour/index.php?n=Main.MolecularGeometryInput) — cfour.uni-mainz.de
172. [Molecules 26, 5162](https://www.mdpi.com/1420-3049/26/17/5162) — mdpi.com
173. [Molecules 28, 2153](https://www.mdpi.com/1420-3049/28/5/2153) — mdpi.com
174. [Molpro product catalogue](https://www.molpro.net/info/products.php) — molpro.net
175. [Mones, Ortner & Csányi](https://pmc.ncbi.nlm.nih.gov/articles/PMC6143621/) — pmc.ncbi.nlm.nih.gov
176. [MPI mirror](https://orca-manual.mpi-muelheim.mpg.de/contents/structurereactivity/optimizations.html) — orca-manual.mpi-muelheim.mpg.de
177. [MPS overview](https://docs.nvidia.com/deploy/pdf/CUDA_Multi_Process_Service_Overview.pdf) — docs.nvidia.com
178. [MRCC registration](https://www.mrcc.hu/index.php/getting-started/registration) — mrcc.hu
179. [Mukhopadhyay, Cole & Saykally, Chem. Phys. Lett. 633, 13 (2015)](https://escholarship.org/content/qt1j70w8wt/qt1j70w8wt.pdf) — escholarship.org
180. [multilayer MCTDH](https://pubs.acs.org/doi/10.1021/acs.jpca.5b03256) — pubs.acs.org
181. [Nagy et al., Chem. Sci.](https://real.mtak.hu/205919/1/state-of-the-art-local-correlation-methods-enable-affordable-gold-standard-quantum-chemistry-for-up-to-hundreds-of-atoms.pdf) — real.mtak.hu
182. [Nano-LEGO](https://pmc.ncbi.nlm.nih.gov/articles/PMC10291548/) — pmc.ncbi.nlm.nih.gov
183. [NASymmetry/MolSym](https://github.com/NASymmetry/MolSym) — github.com
184. [NERSC job policy](https://docs.nersc.gov/jobs/policy/) — docs.nersc.gov
185. [Nguyen et al.](https://hal.science/hal-03183158/document) — hal.science
186. [NIST / Gillies et al., Ar–ketene](https://www.nist.gov/publications/rotational-spectra-structure-internal-dynamics-and-electric-dipole-moment-argon-0) — nist.gov
187. [NIST CCCBDB moment-of-inertia conversion](https://cccbdb.nist.gov/convertmomintx.asp) — cccbdb.nist.gov
188. [NIST CODATA second radiation constant](https://physics.nist.gov/cgi-bin/cuu/Value?c22ndrc) — physics.nist.gov
189. [Numerical and exact kinetic energy operator using Eckart conditions with one or several reference geometries: HONO](https://dugi-doc.udg.edu/bitstream/handle/10256/16593/026355.pdf?sequence=1&isAllowed=y) — dugi-doc.udg.edu
190. [NVIDIA MPS architecture](https://docs.nvidia.com/deploy/mps/architecture.html) — docs.nvidia.com
191. [Obenchain et al.](https://bib-pubdb1.desy.de/record/454603/files/The%20low%20barrier%20methyl%20internal%20rotation%20in%20the_Obenchain_submission.pdf) — bib-pubdb1.desy.de
192. [Oka](https://okaionfactory.web.illinois.edu/publications/PDF/oka194.pdf) — okaionfactory.web.illinois.edu
193. [Open MPI issue #11345](https://github.com/open-mpi/ompi/issues/11345) — github.com
194. [OpenMM RPMD plugin](https://docs.openmm.org/latest/userguide/library/09_rpmd_plugin.html) — docs.openmm.org
195. [OPI paper](https://pubmed.ncbi.nlm.nih.gov/41885262/) — pubmed.ncbi.nlm.nih.gov
196. [ORCA 6.1 counterpoise manual](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/counterpoise.html) — faccts.de
197. [ORCA 6.1 extrapolation tutorial](https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extrapol.html) — faccts.de
198. [ORCA 6.1 manual, frequencies](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/frequencies.html) — faccts.de
199. [ORCA 6.1 manual, optimizing with external methods](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations.html) — faccts.de
200. [ORCA 6.1 manual, VPT2](https://www.faccts.de/docs/orca/6.1/manual/contents/spectroscopyproperties/vpt2.html) — faccts.de
201. [ORCA 6.1 MDCI manual](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/mdci.html) — faccts.de
202. [ORCA 6.1 tutorial, optimisation with external methods](https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extopt.html) — faccts.de
203. [ORCA 6.1, SCF](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/scf.html) — faccts.de
204. [ORCA dispersion manual](https://orca-manual.mpi-muelheim.mpg.de/contents/modelchemistries/dispersioncorrections.html) — orca-manual.mpi-muelheim.mpg.de
205. [ORCA End User License Agreement](https://www.hpc.unipr.it/dokuwiki/lib/exe/fetch.php?media=calcoloscientifico:cluster:softwareapplicativo:orca4.0.x-eula.pdf) — hpc.unipr.it
206. [ORCA EULA, June 2021 copy](https://hpc.hku.hk/wp-content/uploads/document/orca-eula_2021.pdf) — hpc.hku.hk
207. [ORCA frozen core](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/frozencore.html) — faccts.de
208. [ORCA GOAT manual](https://www.faccts.de/docs/orca/6.0/manual/contents/typical/GOAT.html) — faccts.de
209. [ORCA GOAT manual](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/goat.html) — faccts.de
210. [ORCA Input Library, SCF convergence issues](https://sites.google.com/site/orcainputlibrary/scf-convergence-issues) — sites.google.com
211. [ORCA Input Library, setting up ORCA](https://sites.google.com/site/orcainputlibrary/setting-up-orca) — sites.google.com
212. [ORCA molecular dynamics](https://www.faccts.de/docs/orca/6.0/manual/contents/detailed/moldyn.html) — faccts.de
213. [ORCA multiscale tutorial](https://www.faccts.de/docs/orca/6.1/tutorials/multi/basics-otheroniom.html) — faccts.de
214. [ORCA numerical gradients](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/numericalgradients.html) — faccts.de
215. [ORCA numerical integration](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/numericalintegration.html) — faccts.de
216. [ORCA optimisations](https://www.faccts.de/docs/orca/6.0/manual/contents/typical/optimizations.html) — faccts.de
217. [ORCA parallel manual](https://orca-manual.mpi-muelheim.mpg.de/contents/essentialelements/parallel.html) — orca-manual.mpi-muelheim.mpg.de
218. [ORCA QM/MM general](https://orca-manual.mpi-muelheim.mpg.de/contents/multiscalesimulations/qmmm-general.html) — orca-manual.mpi-muelheim.mpg.de
219. [ORCA VPT2](https://orca-manual.mpi-muelheim.mpg.de/contents/spectroscopyproperties/vpt2.html) — orca-manual.mpi-muelheim.mpg.de
220. [OSTI report](https://www.osti.gov/servlets/purl/2578595) — osti.gov
221. [Papajak & Truhlar](https://pubs.acs.org/doi/10.1021/ct200106a) — pubs.acs.org
222. [Parsl example configs](https://parsl.readthedocs.io/en/stable/userguide/configuration/examples.html) — parsl.readthedocs.io
223. [Parsl HighThroughputExecutor](https://parsl.readthedocs.io/en/stable/stubs/parsl.executors.HighThroughputExecutor.html) — parsl.readthedocs.io
224. [pategroup AUTOFIT](https://github.com/pategroup/bband_scripts/tree/master/autofit) — github.com
225. [Path Integral Methods in Atomistic Modelling, Eq. 3.28](https://pure.mpg.de/rest/items/item_3702731_1/component/file_3702732/content) — pure.mpg.de
226. [path-integral simulations of rigid rotors](https://www.lmp.uni-saarland.de/wp-content/uploads/2017/09/JPhyCoMa99.pdf) — lmp.uni-saarland.de
227. [Pavošević & Neese, SI](https://arxiv.org/pdf/2008.03237.pdf) — arxiv.org
228. [PCCP 2015](https://pubmed.ncbi.nlm.nih.gov/26327507/) — pubmed.ncbi.nlm.nih.gov
229. [PCWorld, quoting Intel](https://www.pcworld.com/article/545013/intel-alder-lake-to-offer-8-p-core-only-model-and-have-avx512-too.html) — pcworld.com
230. [Periodica Polytechnica](https://pp.bme.hu/ch/article/download/2806/1911/6564) — pp.bme.hu
231. [PERUN HPC](https://wiki.perun.tuke.sk/env/orca/) — wiki.perun.tuke.sk
232. [PetaChem](http://www.petachem.com/performance.html) — petachem.com
233. [Petit & McCoy, JCP 136, 074101 (2012)](https://pubs.aip.org/aip/jcp/article/136/7/074101/190734/Unraveling-rotation-vibration-mixing-in-highly) — pubs.aip.org
234. [Petit, Wellen & McCoy, JCP 138, 034105 (2013)](https://pubs.aip.org/aip/jcp/article/138/3/034105/192485/Using-fixed-node-diffusion-Monte-Carlo-to) — pubs.aip.org
235. [PGOPHER linear-molecule nucleus help](https://pgopher.chm.bris.ac.uk/Help/linearnucleus.htm) — pgopher.chm.bris.ac.uk
236. [PGOPHER release notes](https://pgopher.chm.bris.ac.uk/download/old/9.0.101/Help/bugs.htm) — pgopher.chm.bris.ac.uk
237. [Pickett, SPFIT/SPCAT documentation](https://spec.jpl.nasa.gov/ftp/pub/calpgm/spinv.pdf) — spec.jpl.nasa.gov
238. [Plumley & Dannenberg](https://pmc.ncbi.nlm.nih.gov/articles/PMC3073166/) — pmc.ncbi.nlm.nih.gov
239. [PMC9105391](https://pmc.ncbi.nlm.nih.gov/articles/PMC9105391/) — pmc.ncbi.nlm.nih.gov
240. [proline methyl ester](https://pubs.rsc.org/en/content/articlehtml/2025/cp/d5cp00898k) — pubs.rsc.org
241. [Psi4Education](https://psicode.org/posts/psi4education/) — psicode.org
242. [psicode.org](https://psicode.org/) — psicode.org
243. [PubMed 41081418](https://pubmed.ncbi.nlm.nih.gov/41081418/) — pubmed.ncbi.nlm.nih.gov
244. [Puzzarini & Stanton](https://pubs.rsc.org/en/content/articlehtml/2023/cp/d2cp04706c) — pubs.rsc.org
245. [Puzzarini & Stanton, PCCP 25, 1421 (2023)](https://pubs.rsc.org/en/content/articlepdf/2023/cp/d2cp04706c) — pubs.rsc.org
246. [Puzzarini et al., oxirane](https://pmc.ncbi.nlm.nih.gov/articles/PMC4630858/) — pmc.ncbi.nlm.nih.gov
247. [Puzzarini group composite review](https://pmc.ncbi.nlm.nih.gov/articles/PMC9863398/) — pmc.ncbi.nlm.nih.gov
248. [pyckett on PyPI](https://pypi.org/project/pyckett/) — pypi.org
249. [PySCF benchmarks](https://pyscf.org/benchmark.html) — pyscf.org
250. [PySCF issue 1360](https://github.com/pyscf/pyscf/issues/1360) — github.com
251. [PySCF issue 2533](https://github.com/pyscf/pyscf/issues/2533) — github.com
252. [PySCF lib API](https://pyscf.org/pyscf_api_docs/pyscf.lib.html) — pyscf.org
253. [PySCF SCF](https://pyscf.org/user/scf.html) — pyscf.org
254. [PySCF's GPU page](https://pyscf.org/user/gpu.html) — pyscf.org
255. [pyscf.org](https://pyscf.org/about.html) — pyscf.org
256. [PySpecTools](https://github.com/laserkelvin/PySpecTools) — github.com
257. [Q-Chem forum](https://talk.q-chem.com/t/q-chem-5-4-2-and-brianqc-1-2-1-wall-clock-time-discrepancy/538) — talk.q-chem.com
258. [QCFractal managers](https://docs.qcarchive.molssi.org/admin_guide/managers/index.html) — docs.qcarchive.molssi.org
259. [QCSchema specification components](https://molssi-qc-schema.readthedocs.io/en/latest/spec_components.html) — molssi-qc-schema.readthedocs.io
260. [QCxMS run documentation](https://xtb-docs.readthedocs.io/en/latest/qcxms_doc/qcxms_run.html) — xtb-docs.readthedocs.io
261. [QUEST](https://arxiv.org/pdf/2001.00416v1.pdf) — arxiv.org
262. [QUICK docs](https://quick-docs.readthedocs.io/en/latest/performance.html) — quick-docs.readthedocs.io
263. [RAARR, arXiv:1812.06221](https://arxiv.org/pdf/1812.06221.pdf) — arxiv.org
264. [racer benchmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC12977065/) — pmc.ncbi.nlm.nih.gov
265. [Rambus](https://www.rambus.com/blogs/pci-express-4/) — rambus.com
266. [Roehling, Hill, Daly & Kukolich](https://experts.arizona.edu/en/publications/ammonia-formic-acid-complex-internal-rotation-analysis-calculatio/) — experts.arizona.edu
267. [Romano, Patterson & Candès](https://arxiv.org/pdf/1905.03222) — arxiv.org
268. [Rotational spectra of van der Waals complexes, CSIC repository](https://digital.csic.es/bitstream/10261/230932/4/Rotational%20spectra%20of%20van%20der%20Waals%20complexes.pdf) — digital.csic.es
269. [Rowan](https://www.rowansci.com/blog/gpu4pyscf) — rowansci.com
270. [RPMD review, CJCP](https://cjcp.ustc.edu.cn/hxwlxb/cn/article/pdf/preview/10.1063/1674-0068/cjcp1808186.pdf) — cjcp.ustc.edu.cn
271. [running CFOUR in parallel](https://cfour.uni-mainz.de/cfour/index.php?n=Main.RunningCfourInParallel) — cfour.uni-mainz.de
272. [RWTH Aachen group's publication record](https://www.chemie.rwth-aachen.de/cms/chemie/forschung/uebersichtsseite-publikationen/publikationen/~mdfct/details/?file=64088&lidx=1) — chemie.rwth-aachen.de
273. [r²SCAN-3c](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60c752f6bb8c1a21633dbf6c/original/r2scan-3c-an-efficient-swiss-army-knife-composite-electronic-structure-method.pdf) — chemrxiv.org
274. [S66](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=98bccb93809e66f5eaf70568c6b9a6b9c80539d1) — citeseerx.ist.psu.edu
275. [SCM two-step geometry optimisation](https://www.scm.com/doc/AMS/Examples/2StepGO.html) — scm.com
276. [Semantic Scholar PDF](https://pdfs.semanticscholar.org/0be3/8225f41fd1700efaa6b69c19ef4e763d2784.pdf) — pdfs.semanticscholar.org
277. [Sigma2 install guide](https://documentation.sigma2.no/software/userinstallsw/ORCA.html) — documentation.sigma2.no
278. [Singh & Henkelman](https://theory.cm.utexas.edu/henkelman/pubs/singh24_10022.pdf) — theory.cm.utexas.edu
279. [sobereva Note 150](http://sobereva.com/150) — sobereva.com
280. [Speeding up MACE](https://arxiv.org/html/2510.23621v1) — arxiv.org
281. [SPIN course, First Steps with SPFIT/SPCAT](https://spin.astro.uni-koeln.de/chapter/SPFITSPCATUniverse/) — spin.astro.uni-koeln.de
282. [stability analysis](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/stabilityanalysis.html) — faccts.de
283. [Steinbach & Bannwarth, PCCP 26, 16567](https://pubmed.ncbi.nlm.nih.gov/38829649/) — pubmed.ncbi.nlm.nih.gov
284. [TACC Stampede3](https://docs.tacc.utexas.edu/hpc/stampede3/) — docs.tacc.utexas.edu
285. [TechPowerUp](https://www.techpowerup.com/cpu-specs/core-i7-13700k.c2850) — techpowerup.com
286. [TeraChem](https://arxiv.org/html/2406.14920v3) — arxiv.org
287. [Texas A&M HPRC](https://hprc.tamu.edu/kb/Software/ORCA/) — hprc.tamu.edu
288. [The Devil in the Details](https://s3-eu-west-1.amazonaws.com/itempdf74155353254prod/10187756/The_Devil_in_the_Details__What_Everybody_Should_Know_When_Running_DFT_Calculations_v1.pdf) — s3-eu-west-1.amazonaws.com
289. [Tom's Hardware](https://www.tomshardware.com/news/intel-reportedly-kills-avx-512-alder-lake-cpus) — tomshardware.com
290. [trends-in-science](http://trends-in-science.blogspot.com/2010/04/cfour-in-parallel.html) — trends-in-science.blogspot.com
291. [Tunnelling splittings in water clusters from PIMD, EPFL Infoscience](https://infoscience.epfl.ch/server/api/core/bitstreams/cc42831e-5aa1-461b-8e00-0d72817f4a4d/content) — infoscience.epfl.ch
292. [TURBOMOLE Educational](https://store.turbomole.org/product/turbomole-8-0-educational/) — store.turbomole.org
293. [UCLouvain, virtual computational-chemistry teaching laboratories](https://dial.uclouvain.be/pr/boreal/object/boreal:254784/datastream/PDF_01/view) — dial.uclouvain.be
294. [Uteva et al.](https://nottingham-repository.worktribe.com/OutputFile/1190028) — nottingham-repository.worktribe.com
295. [utilities](https://www.faccts.de/docs/orca/6.1/manual/contents/utilitiesvisualization/utilities.html) — faccts.de
296. [UWO, Spin statistics, chapter 8](https://physics.uwo.ca/~mhoude2/courses/astro9701/Spin_statistics.pdf) — physics.uwo.ca
297. [Valladolid microsolvation study](https://uvadoc.uva.es/bitstream/handle/10324/76017/Manuscript_corr_final.pdf?sequence=1&isAllowed=y) — uvadoc.uva.es
298. [vibrational analysis of methyl cation–rare gas complexes, arXiv:2009.05443](https://arxiv.org/pdf/2009.05443.pdf) — arxiv.org
299. [Vilarrasa-García](https://doublelayer.eu/vilab/2023/04/18/dft-geometry-optimizers/) — doublelayer.eu
300. [Vogt et al., Molecules 29, 5874](https://www.mdpi.com/1420-3049/29/24/5874) — mdpi.com
301. [Wales and co-workers, OSTI UCRL-JRNL-202191](https://www.osti.gov/servlets/purl/15013980) — osti.gov
302. [Wang, Yang, Carrington & Zhang, J. Chem. Phys. 163, 144308 (2025)](https://pubmed.ncbi.nlm.nih.gov/41070798/) — pubmed.ncbi.nlm.nih.gov
303. [water many-body review](https://pmc.ncbi.nlm.nih.gov/articles/PMC5450669/) — pmc.ncbi.nlm.nih.gov
304. [Western, PGOPHER](https://pgopher.chm.bris.ac.uk/Help/PGOPHERaccepted.pdf) — pgopher.chm.bris.ac.uk
305. [what are Codespaces](https://docs.github.com/en/codespaces/about-codespaces/what-are-codespaces) — docs.github.com
306. [When to Use MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html) — docs.nvidia.com
307. [Wikipedia release history](https://en.wikipedia.org/wiki/ORCA_(quantum_chemistry_program)) — en.wikipedia.org
308. [Wu et al.](https://arxiv.org/html/2404.09452v2) — arxiv.org
309. [xQC](https://arxiv.org/html/2507.09772v1) — arxiv.org
310. [xtb documentation, geometry optimization](https://xtb-docs.readthedocs.io/en/latest/optimization.html) — xtb-docs.readthedocs.io
311. [xtb Hessian](https://xtb-docs.readthedocs.io/en/latest/hessian.html) — xtb-docs.readthedocs.io
312. [xtb single point](https://xtb-docs.readthedocs.io/en/latest/sp.html) — xtb-docs.readthedocs.io
313. [Zhang, Wahib & Matsuoka](https://www.hpcs.cs.tsukuba.ac.jp/icpp2019/data/posters/Poster17-abst.pdf) — hpcs.cs.tsukuba.ac.jp
314. [Zhou et al., JCP 2019](https://research-repository.st-andrews.ac.uk/bitstream/handle/10023/16997/Zhou_2019_JCP_Synthesismicrowavespectra_AAM.pdf) — research-repository.st-andrews.ac.uk
315. [zymtrace](https://gozymtrace.com/blog/zymtrace-03-cuda-kernel-launch-latency) — gozymtrace.com
316. [Řezáč, Riley & Hobza, S66](https://pmc.ncbi.nlm.nih.gov/articles/PMC3152974/) — pmc.ncbi.nlm.nih.gov
317. [Δ-learning PES](https://arxiv.org/abs/2011.11601v1) — arxiv.org
318. [ωB97X-3c README](https://github.com/grimme-lab/wB97X-3c/blob/main/README.md) — github.com
319. [ωB97X-V paper](https://escholarship.org/content/qt7297t9vf/qt7297t9vf_noSplash_ae27d0ce06218f8fa9b5e5ef1289d1d8.pdf) — escholarship.org
