# Retained genuine TOPOS/xTB handoff

The active `topos_native_xtb_reviewed_water.json` is an unchanged copy of the
handoff generated during the installed ecosystem acceptance on 2026-10-08.
That acceptance used ordinary installed wheels, the BASE execution backend, and
an actual xTB 6.7.1 water optimization. Its electronic energy was
**−5.070544054679 Eh**. These consumer tests import retained results; they do not
run xTB, perform a TORQ calculation, or establish full SRS scientific acceptance.

The original acceptance sources were:

| Package | Git revision |
| --- | --- |
| TOPOS | `ef750eaaf6a990a3b9e1d65eca240ac89b6632ce` |
| BASE | `397d48d806d2f5fed2605cd68ca4a21dac470e23` |
| TORQ | `79fbb111125e50627a1a2c129888a45496f368d4` |

TOPOS wheel SHA-256:
`a8e57f4ed91050f5e45be8121ed390de76524377ff7295286a03b9bf892a9f8f`.

The retained acceptance evidence folder is
`ef750ea-v2-evidence/installed-application-success`. Its `preservation-proof.json`
records the hashes of all 122 original retained files. The active fixture is
byte-identical to `torq-handoff.json` and the matching run's retained handoff;
no record, scientific value, review event, timestamp, or checksum was edited.

| Identity | Value |
| --- | --- |
| Fixture file SHA-256 | `d76512c2c41cc796842db7ae0d9458078f74f8bd158f01bf6216e1075f40795c` |
| Producer run | `run_719d0bbddef742aeba87068878d989cb` |
| Native attempt | `attempt_fa99b4b023f04e71959c149f86df0f9b` |
| Native attempt completion | `2026-10-08T05:34:28.167716+00:00` |
| Handoff content digest | `e242be884e7a1dd31b620a497a0701944e9229a1121bae1c7106d683f5cc36a5` |
| Ensemble digest | `2e16f11e6d5d5ea1b054feabdde10191397068ad1f389804d9173b7bcf638889` |
| Source record digest | `9042a3d138617dd7f3f2847799859f6dadcb18a2788d8bf27e7a5495548843a0` |
| Installed acceptance receipt file SHA-256 | `23c13b2b2d90ec41b40f08f896ba483f43d2a164c3109105d253f2693490ba80` |
| Preservation proof file SHA-256 | `a9fbc857c189fcd8fbd4f6d626dd54c030d48236660134c943784b75dd1c7fcd` |
| Original TORQ acknowledgment receipt digest | `9a188ccc5ec5b8cccd4e82bbb68f84b0aadad7648325479e8618bdb523bf8d9b` |

The handoff retains the inventory of native input, output and snapshot artifacts. The original
raw files remain unchanged in that acceptance evidence, including:

| Raw artifact | SHA-256 |
| --- | --- |
| `engine.stdout` | `65dcd547d9a2d73d950e5b6fab4807fdc91544224aa7bc121d2bd24ea58d2b53` |
| `input.xyz` (native attempt input) | `0659cd458a45467014db801da62e021dc4a6c152cc56cf9e7cdcb89d875485a8` |
| `xtbopt.xyz` | `06d2d5e2def88c873c0d2363011b4910129e37ca6c65f706b920f409b9e5e8cc` |
| `version.stdout` | `f47686023e4be88b6e2241c9a2660fffe132a193f8873d3890594b8f9ca0dd91` |

An inventory is not a substitute for those raw files when conducting independent
scientific validation. The original TORQ acknowledgment explicitly records
`computation_performed: false` and acknowledges durable import only.

## Preserved historical fixture

`topos_native_xtb_reviewed_water_legacy_pre_abcluster.json` preserves every byte
of the former active fixture, with file SHA-256
`6953ebda7161a359b6b7a9196948f3b77449105147798e913f63af54ee0f9075`.

Its request predates the explicit `abcluster_options` field. Current TOPOS
record validation adds `abcluster_options: null` when normalizing that request.
The original record digest is
`1d090cd588bcae5bb37745e1ba9483c9bb9aec11dbba08e310d35197cce06340`;
the normalized record digest becomes
`768309464a66b0414ae55398215600e8df8df15029abd2d304d9923340d59050`.
It therefore fails the current reviewed ensemble/record identity check. The
historical test records this incompatibility; it does not silently migrate the
record, recompute its authenticity checksums, or relax the production verifier.
