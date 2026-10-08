# Published reference inputs

These are retrieved scientific source bytes and attributed extraction records.
They are not invented test spectra, engine outputs, independently accepted
benchmark observations or a completed accuracy campaign.

| Collection | Original source | Included scope | Status |
|---|---|---|---|
| [QM9 samples](qm9-samples/) | Ramakrishnan et al., *Scientific Data* **1**, 140022 (2014), [10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22); publisher dataset [10.6084/m9.figshare.978904_D12](https://doi.org/10.6084/m9.figshare.978904_D12) | First 20 original extended XYZ members, original README and publisher API metadata | CC0; source-bound parser inputs |
| [Water comparison](qm9-water/) | Original QM9 member `dsgdb9nsd_000003.xyz` | Three equilibrium rotational constants and three harmonic frequency ranks, with byte pins and an explicitly automated extraction review | Descriptive comparison; original isotope/mass convention, explicit electronic state and physical mode assignments are unavailable |
| [Experimental B₀](experimental-b0-water-formaldehyde/) | Piccardo et al., [10.1021/jp511432m](https://doi.org/10.1021/jp511432m), original publisher supporting information | Six experimental ground-state constants reproduced for water and formaldehyde; original PDF/text, primary bibliography and printed uncertainty notation | CC BY-NC 4.0; secondary reproduction; Hamiltonian/isotope and uncertainty-confidence provenance remain incomplete |
| [Primary experimental B₀](chloroethanol-primary-ground-state/) | Bunn et al., [10.1021/acsearthspacechem.6c00221](https://doi.org/10.1021/acsearthspacechem.6c00221), original published scientific fit files | Six 35Cl/37Cl ground-state 2-chloroethanol constants, original .par/.res/.int bytes and publisher metadata | CC BY-NC 4.0; Watson A/I^r source explicit; remaining identity/calibrated uncertainty unqualified |
| [QCArchive component calculation](qcarchive-nh3/) | Official MolSSI QCArchive tutorial mirror | Historical fixed-geometry NH₃ B2PLYP component values | Computational component comparison; historical engine/version and original basis bytes are incomplete |

The water calculation uses TORQ's existing HF/STO-3G teaching recipe. QM9 uses
B3LYP/6-31G(2df,p). A residual between these protocols is a **cross-method
comparison**, not reproduction of the published protocol or an estimate of
experimental identification accuracy. Source rounding is not uncertainty.

Experimental B₀ and equilibrium Bₑ are different observables. The included
baseline water calculation has no B₀ result. Comparisons preserve that absence;
they do not relabel equilibrium constants. Scientific source files retain their
own declared licences and attribution, independently of the software licence.

The full QM9 compressed archive was streamed and hashed; only the 20 listed
small members were retained. The source index records their actual byte lengths
and SHA-256 digests. No licensed electronic-structure engine archive is included.

See [the runnable product guide](../../docs/development/published_reference_product.md)
for acquisition, actual calculations, strict comparison and remaining provenance
requirements. See [the researcher context](../../docs/development/agent_and_researcher_context.md)
for remaining agent work and narrowly defined owner/researcher prerequisites.
