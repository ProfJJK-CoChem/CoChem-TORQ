# Authentic source acquisition follow-up — 9 October 2026

The inherited proxy and normal TLS-verifying HTTPS requests now returned HTTP200 for three specific sources at 00:33 UTC: NIST Chemistry WebBook water, NIST CCCBDB water and the live QCArchive `/api/v1/information` endpoint. The previous 8 October CONNECT403 responses remain preserved historical observations. Runtime configuration metadata still reported restricted networking with policy state unknown; actual response receipts establish these three successful requests, rather than publication or universal access.

The new [source-only packet](../../benchmarks/published-values/nist-water-source-candidates-20261009/) retains byte hashes, sizes, URLs, UTC times, actual HTTP/CONNECT statuses and extracted numerical facts. Original HTML/API response bytes and full curl receipts remain outside the checkout in the current acquisition environment. No original NIST page is redistributed: the WebBook expressly reserves SRD compilation rights and references the Standard Reference Data Act. Agents must resolve any further source redistribution or benchmark reuse requirements from authoritative policy.

| Source quantity | Original source values / cm⁻¹ | Interpretation retained |
|---|---|---|
| WebBook selected vibrational values, modes1/2/3 | 3657 / 1595 / 3756 | Experimental fundamentals, not harmonic Hessian eigenvalues; A rating says “0~1 cm⁻¹ uncertainty,” with no asserted standard uncertainty. |
| CCCBDB harmonic column, modes1/2/3 | 3832 / 1649 / 3943 | Source-tabulated harmonics; underlying extraction/derivation and uncertainty remain unconfirmed. |
| CCCBDB rotational A/B/C | 27.87700 / 14.51200 / 9.28500 | Equilibrium versus vibrational-ground-state identity is unconfirmed, so these do not replace either Bₑ or B₀ references. |

WebBook separately prints gas-phase IR values3656.65/1594.59/3755.79. They are distinct cells and remain separate from the selected rounded values. The original pages disagree in the label of the antisymmetric mode (`b1` versus `B2`); both source literals are retained while axis/symmetry conventions await reconciliation. Neither page established isotope numbers or all required benchmark identity fields.

The live QCArchive response identifies “The MolSSI OpenFF QCFractal Server,” version0.71 and API/client limits. This is genuine server metadata, **not a molecular calculation record**. No live molecule/energy record was obtained in this bounded handoff. The earlier archived NH₃ B2PLYP tutorial comparison remains a separately attributed historical source. QCSchema is a data format; QCArchive is the calculation repository.

The actual original-page table readback verified nine selected numerical candidates. An initial extractor attempt failed on an omitted HTML row-ending tag in the CCCBDB rotational table; the corrected readback explicitly handles that original markup and passed. This source parsing check did not execute a scientific engine. The packet has **zero accepted benchmark records**, `human_reviewed=false`, and `independent_curation_completed=false`. No acceptance gate or scientific qualification changes.

Remaining agent work: retrieve explicit primary measurement/method definitions, reconcile isotope/state/axis and uncertainty conventions, check actual reuse terms, acquire bounded genuine live QCArchive records, extend strict source-bound importer contracts for fundamentals without relabeling them harmonic, and perform genuine method-matched independent validation. These are research/coding tasks for agents. Only genuinely inaccessible restricted materials or nondelegated scientific/release acceptance require a person. Student deployment remains on hold.
