"""Codespaces notebook interface to the same application used by the CLI.

Opening the interface never starts an electronic-structure calculation. Molecule
previews use tabulated isotopic masses and label the supplied geometry as input.
Submission, validation and result handling are delegated to the application.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from scipy.constants import physical_constants

from Libraries.cochem_isotopes import isotope_mass

BOHR_TO_ANGSTROM = physical_constants["Bohr radius"][0] * 1e10


def _read_json_text(text: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for name, item in pairs:
            if name in value:
                raise ValueError(f"Duplicate JSON field: {name}")
            value[name] = item
        return value

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"Nonfinite JSON number: {value}")

    value = json.loads(text, object_pairs_hook=unique, parse_constant=reject_nonfinite)
    if not isinstance(value, dict):
        raise ValueError("A TORQ JSON input must be an object.")
    return value


def molecule_from_xyz(text: str, *, charge: int, multiplicity: int) -> dict[str, Any]:
    """Read a single Cartesian XYZ geometry (Å), preserving its electronic state."""
    lines = text.strip().splitlines()
    if len(lines) < 3:
        raise ValueError("XYZ input needs an atom count, a comment and atom rows.")
    try:
        count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("The first XYZ line must be an integer atom count.") from exc
    if count <= 0 or len(lines) != count + 2:
        raise ValueError(
            "Supply one XYZ structure with exactly the declared atom count."
        )
    if not isinstance(charge, int) or isinstance(charge, bool):
        raise ValueError("Molecular charge must be an integer.")
    if (
        not isinstance(multiplicity, int)
        or isinstance(multiplicity, bool)
        or multiplicity < 1
    ):
        raise ValueError("Spin multiplicity must be a positive integer.")
    symbols, geometry, isotopes = [], [], []
    for index, line in enumerate(lines[2:], 1):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(
                f"Atom {index} must contain a symbol and three Cartesian coordinates."
            )
        isotope_label = {"D": "2H", "T": "3H"}.get(fields[0], fields[0])
        isotope_mass(isotope_label)
        match = re.fullmatch(
            r"(?:(\d+)([A-Za-z]{1,2})|([A-Za-z]{1,2})(\d*)?)", isotope_label
        )
        assert match is not None  # isotope_mass already validated this identifier
        symbol = (match.group(2) or match.group(3)).capitalize()
        mass_number = match.group(1) or match.group(4)
        try:
            position = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"Atom {index} contains an invalid coordinate.") from exc
        if not np.isfinite(position).all():
            raise ValueError(f"Atom {index} contains a nonfinite coordinate.")
        symbols.append(symbol)
        isotopes.append(int(mass_number) if mass_number else None)
        geometry.append([value / BOHR_TO_ANGSTROM for value in position])
    return {
        "symbols": symbols,
        "geometry_bohr": geometry,
        "charge": charge,
        "multiplicity": multiplicity,
        "atom_ids": [f"atom-{index}" for index in range(1, count + 1)],
        "isotopes": isotopes,
    }


@dataclass(frozen=True)
class MassFrame:
    """Mass-centered principal-axis view of the input; no optimization claim."""

    symbols: tuple[str, ...]
    coordinates_angstrom: np.ndarray
    masses_u: np.ndarray
    center_of_mass_angstrom: np.ndarray
    moments_u_angstrom2: np.ndarray
    rotation: np.ndarray


def input_mass_frame(molecule: dict[str, Any]) -> MassFrame:
    """Construct a finite, right-handed display frame from actual supplied atoms."""
    symbols = tuple(molecule["symbols"])
    coordinates = np.asarray(molecule["geometry_bohr"], dtype=float) * BOHR_TO_ANGSTROM
    if coordinates.shape != (len(symbols), 3) or not len(symbols):
        raise ValueError("Geometry must contain one three-component position per atom.")
    if not np.isfinite(coordinates).all():
        raise ValueError("A molecular preview cannot display nonfinite coordinates.")
    isotopes = molecule.get("isotopes") or [None] * len(symbols)
    if len(isotopes) != len(symbols) or any(
        number is not None
        and (not isinstance(number, int) or isinstance(number, bool) or number < 1)
        for number in isotopes
    ):
        raise ValueError(
            "Isotopes must contain one positive mass number or null per atom."
        )
    mass_symbols = [
        f"{number}{symbol}" if number is not None else symbol
        for symbol, number in zip(symbols, isotopes)
    ]
    masses = np.asarray([isotope_mass(symbol) for symbol in mass_symbols], dtype=float)
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError(
            "A molecular preview requires a tabulated positive mass for every atom."
        )
    center = np.average(coordinates, axis=0, weights=masses)
    centered = coordinates - center
    inertia = np.eye(3) * np.sum(masses * np.sum(centered**2, axis=1))
    inertia -= np.einsum("i,ij,ik->jk", masses, centered, centered)
    moments, rotation = np.linalg.eigh(inertia)
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    return MassFrame(symbols, centered @ rotation, masses, center, moments, rotation)


def inspect_downloaded_results(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Show genuine scientific stage states after verifying each sealed shard."""
    from cochem_torq.artifacts import verify_shard
    from cochem_torq.domain import StageResult, read_json

    directories: set[Path] = set()
    for supplied in paths:
        path = Path(supplied)
        if path.is_dir():
            directories.update(
                manifest.parent
                for manifest in path.rglob("manifest.json")
                if read_json(manifest).get("schema_version") == "cochem.torq.shard/1"
            )
        elif path.name == "manifest.json":
            if read_json(path).get("schema_version") == "cochem.torq.shard/1":
                directories.add(path.parent)
        elif path.name == "result.json":
            directories.add(path.parent)
    summaries = []
    for directory in sorted(directories):
        verify_shard(directory)
        result = read_json(directory / "result.json")
        if result.get("schema_version") != "cochem.torq.result/1":
            raise ValueError("Downloaded result uses an unsupported scientific schema.")
        stages = {
            name: StageResult.model_validate(stage).model_dump(mode="json")
            for name, stage in result["stages"].items()
        }
        summaries.append(
            {
                "directory": str(directory),
                "request_id": result["request_id"],
                "scientific_status": result["status"],
                "recipe": result["recipe"]["label"],
                "experimental_accuracy_established": result[
                    "experimental_accuracy_established"
                ],
                "identification_ready": result["identification_ready"],
                "stages": stages,
                "errors": result["errors"],
            }
        )
    if not summaries:
        raise ValueError(
            "No verified TORQ result shard was present in the downloaded files."
        )
    return summaries


class StudentSession:
    """Stateful student workflow; scientific decisions belong to the application."""

    def __init__(self, results_directory: str | Path | None = None) -> None:
        self.request: dict[str, Any] | None = None
        self.submission: dict[str, Any] | None = None
        self.results_directory = Path(
            results_directory or Path.home() / "CoChem_Artifacts" / "downloads"
        )

    def load_request(self, source: str | Path | dict[str, Any]) -> dict[str, Any]:
        if isinstance(source, dict):
            request = json.loads(json.dumps(source, allow_nan=False))
        else:
            request = _read_json_text(Path(source).read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("A TORQ request must be a JSON object.")
        input_mass_frame(request["molecule"])
        self.request = request
        self.submission = None
        return request

    def prepare_xyz(
        self,
        text: str,
        *,
        charge: int,
        multiplicity: int,
        recipe: str,
        products: list[str],
        cores: int,
        memory_mb: int,
        wall_seconds: int,
    ) -> dict[str, Any]:
        return self.load_request(
            {
                "molecule": molecule_from_xyz(
                    text, charge=charge, multiplicity=multiplicity
                ),
                "recipe": recipe,
                "products": list(products),
                "resources": {
                    "cores": cores,
                    "memory_mb": memory_mb,
                    "wall_seconds": wall_seconds,
                },
                "source_provenance": {
                    "producer": "explicit_import",
                    "source_format": "xyz",
                    "input_text_sha256": sha256(text.encode("utf-8")).hexdigest(),
                    "geometry_status": "input",
                },
            }
        )

    def prepare_handoff(
        self,
        handoff: Any,
        *,
        recipe: str,
        products: list[str],
        cores: int,
        memory_mb: int,
        wall_seconds: int,
    ) -> dict[str, Any]:
        """Preserve the strict BASE/TOPOS handoff and its original method evidence."""
        return self.load_request(
            {
                "molecule": handoff.to_application_molecule(),
                "source_provenance": handoff.to_application_provenance(),
                "recipe": recipe,
                "products": list(products),
                "resources": {
                    "cores": cores,
                    "memory_mb": memory_mb,
                    "wall_seconds": wall_seconds,
                },
            }
        )

    def validate(self) -> dict[str, Any]:
        from cochem_torq.application import validate_request

        report = validate_request(self._request())
        self.request = report["request"]
        return report

    def submit(self) -> dict[str, Any]:
        from cochem_torq.application import submit_request

        # Retain the normalized UUID even when the user bypassed Check request.
        self.validate()
        self.submission = submit_request(self._request(), execution="github_actions")
        return self.submission

    def status(self, run_id: str | int | None = None) -> dict[str, Any]:
        from cochem_torq.application import get_run_status

        return get_run_status(self._run_id(run_id))

    def download(self, run_id: str | int | None = None) -> list[Path]:
        from cochem_torq.application import download_run_results

        return [
            Path(path)
            for path in download_run_results(
                self._run_id(run_id), self.results_directory
            )
        ]

    def export_to_base(
        self, run_directory: str | Path, destination: str | Path
    ) -> Path:
        from cochem_torq.ecosystem import export_base_calculation_result

        return export_base_calculation_result(run_directory, destination)

    def _request(self) -> dict[str, Any]:
        if self.request is None:
            raise ValueError(
                "Import a molecule or request before validating or submitting."
            )
        return self.request

    def _run_id(self, run_id: str | int | None) -> str:
        value = run_id or (self.submission or {}).get("run_id")
        if not value:
            raise ValueError(
                "Enter the GitHub Actions run ID or submit a request first."
            )
        return str(value)


def _display_frame(frame: MassFrame, azimuth: float, elevation: float) -> None:
    import matplotlib.pyplot as plt
    from IPython.display import display

    figure = plt.figure(figsize=(7, 5))
    axis = figure.add_subplot(111, projection="3d")
    positions = frame.coordinates_angstrom
    axis.scatter(*positions.T, s=90)
    for symbol, point in zip(frame.symbols, positions):
        axis.text(*point, symbol)
    radius = max(float(np.max(np.abs(positions))), 0.5) * 1.2
    axis.set(
        xlim=(-radius, radius),
        ylim=(-radius, radius),
        zlim=(-radius, radius),
        xlabel="a / Å",
        ylabel="b / Å",
        zlabel="c / Å",
        title="Supplied geometry · mass-centered principal axes",
    )
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=elevation, azim=azimuth)
    display(figure)
    plt.close(figure)


def launch_student_app(
    *,
    request_path: str | Path | None = None,
    results_directory: str | Path | None = None,
) -> Any:
    """Return an interactive Jupyter widget without submitting or calculating."""
    import ipywidgets as widgets
    from IPython.display import clear_output

    session = StudentSession(results_directory)
    source_kind = widgets.Dropdown(
        options=[
            ("XYZ (Å)", "xyz"),
            ("TORQ request JSON", "json"),
            ("BASE handoff manifest", "base"),
            ("TOPOS HDF5", "topos"),
        ],
        description="Input",
    )
    source = widgets.Textarea(
        placeholder="Paste your XYZ geometry or a TORQ request JSON object.",
        layout=widgets.Layout(width="100%", height="190px"),
    )
    handoff_metadata = widgets.Textarea(
        placeholder=(
            "BASE/TOPOS only: explicit import metadata JSON (see the student guide)."
        ),
        description="Handoff",
        layout=widgets.Layout(width="100%", height="110px"),
    )

    def explain_source(*_: Any) -> None:
        handoff_metadata.layout.display = (
            "" if source_kind.value in {"base", "topos"} else "none"
        )
        source.placeholder = (
            "Enter the local handoff file path."
            if source_kind.value in {"base", "topos"}
            else "Paste one XYZ geometry or the complete TORQ request JSON."
        )

    source_kind.observe(explain_source, names="value")
    explain_source()
    charge = widgets.IntText(value=0, description="Charge")
    multiplicity = widgets.BoundedIntText(
        value=1, min=1, max=100, description="Multiplicity"
    )
    from cochem_torq.registry import list_method_profiles

    profiles = list_method_profiles()
    recipe = widgets.Dropdown(
        options=[
            (f"{item['label']} ({item['availability']})", item["id"])
            for item in profiles
        ],
        value="hf-sto-3g-education",
        description="Recipe",
        layout=widgets.Layout(width="100%"),
    )
    recipe_note = widgets.HTML()

    def explain_recipe(*_: Any) -> None:
        import html

        selected = next(item for item in profiles if item["id"] == recipe.value)
        recipe_note.value = (
            "<p>" + html.escape(str(selected.get("reason", ""))) + "</p>"
        )

    recipe.observe(explain_recipe, names="value")
    explain_recipe()
    products = widgets.SelectMultiple(
        options=[
            ("Optimized geometry", "geometry"),
            ("Harmonic characterization", "harmonic"),
            ("Equilibrium rotational constants (Bₑ)", "equilibrium_constants"),
            ("Rigid-rotor teaching catalog", "rigid_rotor_catalog"),
            ("Anharmonic force field", "anharmonic_force_field"),
            ("VPT2", "vpt2"),
            ("Vibration-corrected constants (B₀)", "ground_state_constants"),
            ("Qualified identification catalog", "identification_catalog"),
        ],
        value=("geometry", "equilibrium_constants"),
        description="Products",
        rows=8,
    )
    cores = widgets.BoundedIntText(value=2, min=1, max=2, description="Cores")
    memory = widgets.BoundedIntText(
        value=2048, min=256, max=2048, description="Memory MB"
    )
    wall = widgets.BoundedIntText(value=600, min=1, max=600, description="Wall sec")

    def update_source_controls(*_: Any) -> None:
        for control in (recipe, recipe_note, products, cores, memory, wall):
            control.layout.display = "none" if source_kind.value == "json" else ""
        for control in (charge, multiplicity):
            control.layout.display = "" if source_kind.value == "xyz" else "none"

    source_kind.observe(update_source_controls, names="value")
    update_source_controls()
    load = widgets.Button(description="Import and preview", button_style="info")
    validate = widgets.Button(description="Check request")
    submit = widgets.Button(description="Submit to Actions", button_style="success")
    run = widgets.Text(description="Run ID", placeholder="Your GitHub Actions run ID")
    refresh = widgets.Button(description="Refresh status")
    download = widgets.Button(description="Download results")
    results = widgets.Text(
        value=str(session.results_directory),
        description="Save to",
        layout=widgets.Layout(width="100%"),
    )
    verified_run = widgets.Dropdown(
        options=[], description="BASE source", layout=widgets.Layout(width="100%")
    )
    base_destination = widgets.Text(
        value=str(session.results_directory / "base-result.json"),
        description="BASE file",
        layout=widgets.Layout(width="100%"),
    )
    export_base = widgets.Button(description="Export energy to BASE", disabled=True)
    azimuth = widgets.IntSlider(value=-60, min=-180, max=180, description="Rotate")
    elevation = widgets.IntSlider(value=25, min=-90, max=90, description="Tilt")
    output, preview = widgets.Output(), widgets.Output()

    def show(value: Any) -> None:
        with output:
            clear_output(wait=True)
            print(
                json.dumps(value, indent=2, default=str, allow_nan=False)
                if isinstance(value, (dict, list))
                else value
            )

    def action(callback: Any) -> None:
        try:
            show(callback())
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            show(f"Unable to complete this step: {exc}")

    def redraw(*_: Any) -> None:
        if session.request:
            with preview:
                clear_output(wait=True)
                _display_frame(
                    input_mass_frame(session.request["molecule"]),
                    azimuth.value,
                    elevation.value,
                )

    def load_input(_: Any) -> None:
        def perform() -> dict[str, Any]:
            if source_kind.value == "json":
                session.load_request(_read_json_text(source.value))
            elif source_kind.value == "xyz":
                session.prepare_xyz(
                    source.value,
                    charge=charge.value,
                    multiplicity=multiplicity.value,
                    recipe=recipe.value.strip(),
                    products=list(products.value),
                    cores=cores.value,
                    memory_mb=memory.value,
                    wall_seconds=wall.value,
                )
            else:
                from cochem_torq.ecosystem import (
                    MethodProvenance,
                    read_base_handoff,
                    read_topos_handoff,
                )

                metadata = _read_json_text(handoff_metadata.value)
                if not isinstance(metadata, dict):
                    raise ValueError("Handoff metadata must be a JSON object.")
                if source_kind.value == "base":
                    handoff = read_base_handoff(
                        Path(source.value.strip()).expanduser(), **metadata
                    )
                else:
                    metadata["method_provenance"] = MethodProvenance.model_validate(
                        metadata["method_provenance"]
                    )
                    handoff = read_topos_handoff(
                        Path(source.value.strip()).expanduser(), **metadata
                    )
                session.prepare_handoff(
                    handoff,
                    recipe=recipe.value,
                    products=list(products.value),
                    cores=cores.value,
                    memory_mb=memory.value,
                    wall_seconds=wall.value,
                )
            redraw()
            return {
                "input": "Imported; this geometry has not been optimized.",
                "request": session.request,
            }

        action(perform)

    def send(_: Any) -> None:
        def perform() -> dict[str, Any]:
            value = session.submit()
            if value.get("run_id"):
                run.value = str(value["run_id"])
            return value

        action(perform)

    load.on_click(load_input)
    validate.on_click(lambda _: action(session.validate))
    submit.on_click(send)
    refresh.on_click(
        lambda _: action(lambda: session.status(run.value.strip() or None))
    )

    def save(_: Any) -> None:
        session.results_directory = Path(results.value).expanduser()

        def perform() -> dict[str, Any]:
            paths = session.download(run.value.strip() or None)
            summaries = inspect_downloaded_results(paths)
            verified_run.options = [
                (f"{summary['request_id']} · {summary['recipe']}", summary["directory"])
                for summary in summaries
            ]
            export_base.disabled = False
            base_destination.value = str(
                session.results_directory
                / "base-exports"
                / f"{summaries[0]['request_id']}.json"
            )
            return {
                "downloaded": [str(path) for path in paths],
                "scientific_results": summaries,
            }

        action(perform)

    download.on_click(save)
    export_base.on_click(
        lambda _: action(
            lambda: session.export_to_base(
                verified_run.value, Path(base_destination.value).expanduser()
            )
        )
    )
    azimuth.observe(redraw, names="value")
    elevation.observe(redraw, names="value")
    if request_path is not None:
        session.load_request(request_path)
        source_kind.value = "json"
        source.value = json.dumps(session.request, indent=2)
        redraw()

    def invalidate_changed_input(_: Any) -> None:
        if session.request is None:
            return
        session.request = None
        session.submission = None
        with preview:
            clear_output(wait=True)
        show(
            "Inputs changed. Select Import and preview, then check the revised request."
        )

    for control in (
        source_kind,
        source,
        handoff_metadata,
        charge,
        multiplicity,
        recipe,
        products,
        cores,
        memory,
        wall,
    ):
        control.observe(invalidate_changed_input, names="value")

    instructions = widgets.HTML(
        "<h2>CoChem-TORQ</h2><p>Import a molecule, check its state and method, "
        "and submit a calculation to GitHub Actions. Codespaces provides this "
        "interface. A requested scientific product is available only when its "
        "method and backend are qualified.</p><p>Use your institution’s approved "
        "GitHub repository and Actions runner. Check request reports before "
        "submission. A completed workflow can contain partial or unavailable "
        "scientific stages; inspect the downloaded result manifest.</p>"
    )
    app = widgets.VBox(
        [
            instructions,
            source_kind,
            source,
            handoff_metadata,
            widgets.HBox([charge, multiplicity]),
            recipe,
            recipe_note,
            products,
            widgets.HBox([cores, memory, wall]),
            widgets.HBox([load, validate, submit]),
            widgets.HBox([azimuth, elevation]),
            preview,
            widgets.HBox([run, refresh]),
            results,
            download,
            verified_run,
            base_destination,
            export_base,
            output,
        ]
    )
    app.student_session = session
    return app
