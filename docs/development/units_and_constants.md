# Versioned physical units and constants

`src/cochem_torq/units.py` implements the canonical units service required by
TORQ-SCI-002. Its profile is
`cochem.constants.codata-2022-torq-compatible/1`. The service reads SciPy's
documented CODATA 2022 table, records the actual SciPy provider version, and
exposes frozen constant definitions through a read-only mapping. Unsupported
CODATA releases fail explicitly rather than being labeled as the selected release.

`constants_provenance()` returns a fresh JSON-compatible record containing the
profile, data release, [NIST source](https://physics.nist.gov/cuu/Constants/),
value/SI-unit/standard-uncertainty records, provider version and definition digest.
Changing that returned record does not change the authoritative definitions.
Fundamental-constant uncertainty is distinct from molecular model, derivative,
spectroscopic or experimental uncertainty; this service does not calibrate those.

The migrated harmonic, rotational, energetics and ecosystem modules obtain their
conversion constants from this service. Original floating-point operation order
is retained for Bohr/Angstrom, atomic-mass/electron-mass and Hartree/wavenumber
factors. Existing harmonic and electronic numerical models and BASE-owned
identity payloads remain unchanged.

## Explicit unit and dimension boundaries

`convert(values, source, target)` supports explicitly named units for lengths,
energies, molar energies, cycle frequencies, angular frequencies, wavenumbers,
dipoles, masses, Kelvin temperatures, angles, Cartesian gradients/Hessians,
dimensionless coordinates and mass-weighted normal coordinates. Values are real
float64 scalars or rectangular arrays. Complex data, even with a zero imaginary
part, cannot lose its imaginary component silently. Strings, booleans, object
arrays, nonfinite data and conversion overflow are rejected. Signed frequencies
and genuine zeros are preserved.

Dimension changes require an explicit physical equivalence:

```python
from cochem_torq.units import convert

length_angstrom = convert(1.0, "bohr", "angstrom")
energy_kj_per_mol = convert(1.0, "hartree", "kJ/mol", equivalence="molar")
frequency_hz = convert(1.0, "cm^-1", "Hz", equivalence="spectroscopic")
angular_frequency = convert(1.0, "Hz", "rad/s", equivalence="spectroscopic")
```

Molar equivalence uses Avogadro's constant to distinguish energy per molecule
from energy per mole. Spectroscopic equivalence explicitly uses photon relations
and distinguishes cycles from radians. Degrees/radians are angles, not an
implicitly interchangeable dimensionless normal coordinate. A mass-weighted
normal coordinate in `bohr*sqrt(electron_mass)` cannot become Cartesian Bohr
coordinates through a scalar unit conversion; that operation needs the actual
mode transformation and isotope masses.

## Cartesian and normal-coordinate contracts

`CartesianFrame` requires an explicit frame ID, reference-frame ID, origin with
a declared length unit, and an orthonormal right-handed axis matrix. Its columns
are target axes expressed in the reference frame. Origins and axes are copied
into immutable byte-backed arrays. This declaration does not infer molecular
symmetry, a center of mass or a wavefunction origin.

`PhysicalQuantity` records its quantity kind, unit, values and constants profile.
Cartesian coordinates, gradients, forces, Hessians and dipoles require an
explicit frame. Atom-indexed Cartesian arrays additionally require a unique atom
row map; Hessians use atom-major XYZ order. Normal coordinates require the actual
mode-basis digest and preserve their declared mass-weighted or dimensionless
convention. No fabricated frame, row identity or normal-coordinate basis is
inserted to permit a conversion.

Unit conversion preserves the quantity kind. `gradient_to_force()` explicitly
implements force = minus the energy gradient; changing a gradient's unit does not
change its sign or silently relabel it as a force. Unit conversion also preserves
the declared origin and frame. Rotating vectors/tensors or shifting an ionic
dipole's origin requires the separate validated transformation, including the
total-charge origin correction.

## Preserved Debye convention

Existing TORQ calculations used `3.33564e-30 C m` per Debye. The compatibility
profile retains that rounded factor and labels its convention explicitly.
`debye_exact` is a separately named opt-in convention using `1e-21/c` and is
recorded as a reference in constants provenance. Its value is not substituted
into existing intensities or catalogs. Switching conventions is a versioned
numerical change requiring the applicable downstream replay and qualification.

## Verification scope

`tests/test_units.py` uses published SI defining constants, independent decimal
energy-to-molar calculations, frequency/wavenumber identities, signed roundtrips,
strict numerical rejection, immutable records, frame handedness, explicit
gradient/force sign and coordinate-dimension checks. Existing spectroscopy and
ecosystem serialization checks verify unchanged factors and identity behavior.
These are mathematical and software checks, with no substitute engine outputs
or experimental-accuracy claims. Applicable real-engine release gates remain
independently required.
