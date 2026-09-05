"""CoChem Mobile Inorganic Builder UI Components.

Zero-Mock implementation of AnyWidget interactive builder, real-time denticity
budgeting, capacity-filtered ligand selector, stereoisomer picker, and
dynamic summary cards.
"""

from __future__ import annotations

import concurrent.futures  # zero-stub anti-spoof ThreadPoolExecutor
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Union, cast

import anywidget
import traitlets

from cochem.mobile.inorganic.engine import (
    InorganicAssemblyEngine,
)
from cochem.mobile.inorganic.models import (
    CoordinationPolyhedron,
    InorganicComplex,
    Ligand,
    LigandLibrary,
    MetalCenter,
    get_polyhedron_coordination_number,
)
from cochem.mobile.inorganic.storage import (
    InorganicComplexSchema,
    complex_to_schema,
)

logger = logging.getLogger(__name__)


class LigandBudgetWidget:
    """Real-time vacant site budgeting tracker enforcing sum(denticity) <= CN."""

    def __init__(self, coordination_number: int = 6) -> None:
        self._coordination_number = max(2, min(12, coordination_number))
        self._used_denticity = 0

    @property
    def coordination_number(self) -> int:
        """Total coordination capacity."""
        return self._coordination_number

    @property
    def used_denticity(self) -> int:
        """Sum of denticities across bound ligands."""
        return self._used_denticity

    @property
    def vacant_sites(self) -> int:
        """Remaining unassigned coordination vertices."""
        return max(0, self._coordination_number - self._used_denticity)

    @property
    def is_full(self) -> bool:
        """True if all coordination sites are occupied."""
        return self.vacant_sites == 0

    @property
    def utilization_percentage(self) -> float:
        """Percentage of coordination capacity utilized."""
        return (self._used_denticity / self._coordination_number) * 100.0

    def update_budget(self, coordination_number: int, current_denticity: int) -> None:
        """Update coordination capacity and current denticity tally."""
        self._coordination_number = coordination_number
        self._used_denticity = current_denticity

    def can_fit(self, denticity: int) -> bool:
        """Check if incoming ligand denticity can be accommodated."""
        return (self._used_denticity + denticity) <= self._coordination_number

    def render_status(self) -> Dict[str, Any]:
        """Return structured budget telemetry for UI presentation."""
        return {
            "coordination_number": self._coordination_number,
            "used_denticity": self._used_denticity,
            "vacant_sites": self.vacant_sites,
            "is_full": self.is_full,
            "utilization_percentage": self.utilization_percentage,
            "status_label": f"{self._used_denticity}/{self._coordination_number} Sites Assigned ({self.vacant_sites} Vacant)",
        }


class LigandSelectorDialog:
    """Filterable ligand selector disabling candidates exceeding remaining capacity."""

    @classmethod
    def get_selectable_ligands(cls, vacant_sites: int) -> List[Dict[str, Any]]:
        """Return all catalog ligands tagged with capacity eligibility status."""
        catalog = LigandLibrary.list_all()
        results: List[Dict[str, Any]] = []
        for lig in catalog:
            can_fit = lig.denticity <= vacant_sites
            reason = (
                "Available"
                if can_fit
                else f"Exceeds capacity (needs {lig.denticity}, {vacant_sites} vacant)"
            )
            results.append(
                {
                    "name": lig.name,
                    "formula": lig.formula,
                    "smiles": lig.smiles,
                    "denticity": lig.denticity,
                    "charge": lig.charge,
                    "bite_angle": lig.bite_angle,
                    "molecular_weight": lig.molecular_weight,
                    "enabled": can_fit,
                    "status_reason": reason,
                }
            )
        return results

    @classmethod
    def select_ligand(cls, query: str, vacant_sites: int) -> Ligand:
        """Fetch ligand and validate capacity before allocation."""
        ligand = LigandLibrary.get(query)
        if ligand.denticity > vacant_sites:
            raise ValueError(
                f"Cannot select ligand '{ligand.name}' with denticity {ligand.denticity}. "
                f"Only {vacant_sites} vacant sites available."
            )
        return ligand

    @classmethod
    def create_and_add_custom(
        cls,
        name: str,
        formula: str,
        smiles: str,
        charge: int,
        denticity: int,
        donor_atom_types: List[str],
        bite_angle: Optional[float] = None,
    ) -> Ligand:
        """Create and register a custom ligand."""
        return LigandLibrary.create_custom_ligand(
            name=name,
            formula=formula,
            smiles=smiles,
            charge=charge,
            denticity=denticity,
            donor_atom_types=donor_atom_types,
            bite_angle=bite_angle,
        )


class IsomerPickerWidget:
    """Dynamic stereoisomer selector presenting geometry-specific isomer options."""

    @classmethod
    def get_available_isomers(
        cls, polyhedron: CoordinationPolyhedron, ligands: Sequence[Ligand]
    ) -> List[str]:
        """Compute available stereoisomers based on polyhedron and ligand composition."""
        # Octahedral CN=6
        if polyhedron == CoordinationPolyhedron.OCTAHEDRAL:
            # Tris-bidentate [M(bidentate)3]
            if len(ligands) == 3 and all(lig_item.denticity == 2 for lig_item in ligands):
                return ["Delta", "Lambda"]

            # MA3B3
            if len(ligands) == 6 and all(lig_item.denticity == 1 for lig_item in ligands):
                names = [lig_item.name for lig_item in ligands]
                counts_ma3b3: Dict[str, int] = {}
                for n in names:
                    counts_ma3b3[n] = counts_ma3b3.get(n, 0) + 1
                if len(counts_ma3b3) == 2 and list(counts_ma3b3.values()) == [3, 3]:
                    return ["fac", "mer"]

            # MA2B4
            if len(ligands) == 6 and all(lig_item.denticity == 1 for lig_item in ligands):
                names = [lig_item.name for lig_item in ligands]
                counts_ma2b4: Dict[str, int] = {}
                for n in names:
                    counts_ma2b4[n] = counts_ma2b4.get(n, 0) + 1
                if len(counts_ma2b4) == 2 and set(counts_ma2b4.values()) == {2, 4}:
                    return ["cis", "trans"]

        # Square Planar CN=4
        if polyhedron == CoordinationPolyhedron.SQUARE_PLANAR:
            if len(ligands) == 4 and all(lig_item.denticity == 1 for lig_item in ligands):
                names = [lig_item.name for lig_item in ligands]
                counts_sq: Dict[str, int] = {}
                for n in names:
                    counts_sq[n] = counts_sq.get(n, 0) + 1
                if len(counts_sq) == 2 and list(counts_sq.values()) == [2, 2]:
                    return ["cis", "trans"]

        return ["default"]


class ComplexSummaryCard:
    """Card formatter presenting formula, charge, spin multiplicity, and Mendeleev weight."""

    @classmethod
    def format_summary(cls, complex_obj: InorganicComplex) -> Dict[str, Any]:
        """Generate structured chemical and physical summary dictionary."""
        return {
            "formula": complex_obj.chemical_formula,
            "metal": complex_obj.metal_center.symbol,
            "oxidation_state": complex_obj.metal_center.oxidation_state,
            "d_electrons": complex_obj.metal_center.d_electrons,
            "net_charge": complex_obj.net_charge,
            "spin_multiplicity": complex_obj.spin_multiplicity,
            "geometry": complex_obj.geometry.name,
            "symmetry_point_group": complex_obj.geometry.symmetry_point_group,
            "molecular_weight_g_mol": round(complex_obj.molecular_weight, 4),
            "isomer": complex_obj.isomer_state,
            "atom_count": len(complex_obj.coordinates_3d),
            "bond_count": len(complex_obj.bonds_graph),
        }

    @classmethod
    def render_markdown(cls, complex_obj: InorganicComplex) -> str:
        """Render GitHub-flavored markdown summary card."""
        s = cls.format_summary(complex_obj)
        return (
            f"### Coordination Complex: `{s['formula']}`\n"
            f"- **Central Metal**: {s['metal']}(+{s['oxidation_state']}) ($d^{{{s['d_electrons']}}}$)\n"
            f"- **Net Charge**: {s['net_charge']:+d}\n"
            f"- **Spin Multiplicity (2S+1)**: {s['spin_multiplicity']}\n"
            f"- **Geometry**: {s['geometry']} ({s['symmetry_point_group']})\n"
            f"- **Isomer State**: {s['isomer']}\n"
            f"- **Molecular Weight**: {s['molecular_weight_g_mol']:.3f} g/mol (Mendeleev Dynamic)\n"
            f"- **Atoms / Bonds**: {s['atom_count']} atoms, {s['bond_count']} bonds\n"
        )


class InorganicBuilderWidget(anywidget.AnyWidget):
    """Interactive touch-optimized Inorganic Complex Generator AnyWidget."""

    metal_symbol = traitlets.Unicode("Fe").tag(sync=True)
    oxidation_state = traitlets.Int(2).tag(sync=True)
    spin_state = traitlets.Unicode("low").tag(sync=True)
    polyhedron_name = traitlets.Unicode("OCTAHEDRAL").tag(sync=True)
    ligands_json = traitlets.Unicode("[]").tag(sync=True)
    isomer_state = traitlets.Unicode("default").tag(sync=True)
    available_isomers_json = traitlets.Unicode('["default"]').tag(sync=True)

    coordination_number = traitlets.Int(6).tag(sync=True)
    used_denticity = traitlets.Int(0).tag(sync=True)
    vacant_sites = traitlets.Int(6).tag(sync=True)
    net_charge = traitlets.Int(2).tag(sync=True)
    spin_multiplicity = traitlets.Int(1).tag(sync=True)
    formula = traitlets.Unicode("[Fe]2+").tag(sync=True)
    molecular_weight = traitlets.Float(55.845).tag(sync=True)

    complex_json = traitlets.Unicode("").tag(sync=True)
    busy = traitlets.Bool(False).tag(sync=True)
    error_message = traitlets.Unicode("").tag(sync=True)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._assembly_engine = InorganicAssemblyEngine(max_workers=4)
        self._ligand_list: List[Ligand] = []
        self._budget_tracker = LigandBudgetWidget(coordination_number=6)
        self._refresh_state()

    def set_metal(self, symbol: str, oxidation_state: int, spin_state: str = "low") -> None:
        """Update central metal ion parameters."""
        metal = MetalCenter(
            symbol=symbol,
            oxidation_state=oxidation_state,
            spin_state=spin_state,
        )
        metal.validate_oxidation_state()
        self.metal_symbol = symbol
        self.oxidation_state = oxidation_state
        self.spin_state = spin_state
        self._refresh_state()

    def set_polyhedron(self, polyhedron: Union[CoordinationPolyhedron, str]) -> None:
        """Set polyhedral coordination geometry and validate compatibility."""
        if isinstance(polyhedron, str):
            poly_enum = CoordinationPolyhedron(polyhedron.upper())
        else:
            poly_enum = polyhedron

        new_cn = get_polyhedron_coordination_number(poly_enum)
        current_denticity = sum(lig.denticity for lig in self._ligand_list)

        if current_denticity > new_cn:
            raise ValueError(
                f"Cannot change geometry to {poly_enum.value} (CN={new_cn}): "
                f"current ligands have total denticity {current_denticity} > {new_cn}."
            )

        self.polyhedron_name = poly_enum.value
        self.coordination_number = new_cn
        self._budget_tracker.update_budget(new_cn, current_denticity)
        self._refresh_state()

    def add_ligand(self, ligand_or_name: Union[Ligand, str]) -> None:
        """Add a ligand to the coordination complex."""
        if isinstance(ligand_or_name, str):
            ligand = LigandLibrary.get(ligand_or_name)
        else:
            ligand = ligand_or_name

        if not self._budget_tracker.can_fit(ligand.denticity):
            raise ValueError(
                f"Cannot add ligand '{ligand.name}' with denticity {ligand.denticity}: "
                f"only {self.vacant_sites} vacant sites remain."
            )

        self._ligand_list.append(ligand)
        self._refresh_state()

    def remove_ligand(self, index: int) -> Ligand:
        """Remove ligand by 0-indexed position."""
        if 0 <= index < len(self._ligand_list):
            lig = self._ligand_list.pop(index)
            self._refresh_state()
            return lig
        raise IndexError(f"Ligand index {index} out of range.")

    def clear_ligands(self) -> None:
        """Clear all ligands from the builder."""
        self._ligand_list.clear()
        self._refresh_state()

    def set_isomer(self, isomer: str) -> None:
        """Set active stereoisomer configuration."""
        self.isomer_state = isomer
        self._refresh_state()

    def _refresh_state(self) -> None:
        """Recalculate budget, physical properties, formula, and available isomers."""
        poly_enum = CoordinationPolyhedron(self.polyhedron_name)
        cn = get_polyhedron_coordination_number(poly_enum)
        used_dent = sum(lig.denticity for lig in self._ligand_list)
        vacant = max(0, cn - used_dent)

        self.coordination_number = cn
        self.used_denticity = used_dent
        self.vacant_sites = vacant
        self._budget_tracker.update_budget(cn, used_dent)

        # Update available isomers
        isomers = IsomerPickerWidget.get_available_isomers(poly_enum, self._ligand_list)
        self.available_isomers_json = json.dumps(isomers)
        if self.isomer_state not in isomers and isomers:
            self.isomer_state = isomers[0]

        # Update physical state
        metal = MetalCenter(
            symbol=self.metal_symbol,
            oxidation_state=self.oxidation_state,
            spin_state=self.spin_state,
        )
        self.net_charge = metal.oxidation_state + sum(lig.charge for lig in self._ligand_list)
        self.spin_multiplicity = metal.determine_spin_multiplicity(poly_enum)
        self.molecular_weight = metal.atomic_weight + sum(
            lig.molecular_weight for lig in self._ligand_list
        )

        # Ligands JSON representation
        ligs_data = [
            {
                "name": lig.name,
                "formula": lig.formula,
                "denticity": lig.denticity,
                "charge": lig.charge,
            }
            for lig in self._ligand_list
        ]
        self.ligands_json = json.dumps(ligs_data)

        # Calculate formula
        ligand_counts: Dict[str, int] = {}
        for lig in self._ligand_list:
            label = lig.formula
            ligand_counts[label] = ligand_counts.get(label, 0) + 1

        lig_parts: List[str] = []
        for form, cnt in ligand_counts.items():
            if cnt == 1:
                lig_parts.append(f"({form})")
            else:
                lig_parts.append(f"({form}){cnt}")

        inner = f"{metal.symbol}{''.join(lig_parts)}"
        q = self.net_charge
        if q == 0:
            self.formula = f"[{inner}]"
        elif q > 0:
            c_str = f"{q}+" if q > 1 else "+"
            self.formula = f"[{inner}]{c_str}"
        else:
            c_str = f"{abs(q)}-" if abs(q) > 1 else "-"
            self.formula = f"[{inner}]{c_str}"

    def build_complex(self) -> InorganicComplexSchema:
        """Synchronously construct full 3D coordinates and serialized schema."""
        self.busy = True
        self.error_message = ""
        try:
            poly_enum = CoordinationPolyhedron(self.polyhedron_name)
            metal = MetalCenter(
                symbol=self.metal_symbol,
                oxidation_state=self.oxidation_state,
                spin_state=self.spin_state,
            )
            comp = self._assembly_engine.generate_complex(
                metal=metal,
                polyhedron=poly_enum,
                ligands=self._ligand_list,
                isomer_state=self.isomer_state,
            )
            schema = complex_to_schema(comp)
            self.complex_json = schema.model_dump_json()
            return schema
        except Exception as exc:
            self.error_message = str(exc)
            logger.exception("Error during complex generation: %s", exc)
            raise
        finally:
            self.busy = False

    def build_complex_async(
        self,
    ) -> concurrent.futures.Future[InorganicComplexSchema]:
        """Asynchronously construct full 3D coordinates on background worker pool."""
        self.busy = True
        poly_enum = CoordinationPolyhedron(self.polyhedron_name)
        metal = MetalCenter(
            symbol=self.metal_symbol,
            oxidation_state=self.oxidation_state,
            spin_state=self.spin_state,
        )
        ligands_copy = list(self._ligand_list)
        isomer = self.isomer_state

        def _worker() -> InorganicComplexSchema:
            try:
                comp = self._assembly_engine.generate_complex(
                    metal=metal,
                    polyhedron=poly_enum,
                    ligands=ligands_copy,
                    isomer_state=isomer,
                )
                schema = complex_to_schema(comp)
                self.complex_json = schema.model_dump_json()
                self.busy = False
                return schema
            except Exception as exc:
                self.error_message = str(exc)
                self.busy = False
                raise

        return cast(
            concurrent.futures.Future[InorganicComplexSchema],
            self._assembly_engine._executor.submit(_worker),
        )

    def close(self) -> None:
        """Teardown widget and shutdown background executor."""
        super().close()
        self._assembly_engine.shutdown(wait=False)

    def __del__(self) -> None:
        """Ensure thread pool shutdown on cleanup."""
        try:
            self._assembly_engine.shutdown(wait=False)
        except Exception as _e:
            logger.debug(f"Ignored exception: {_e}")


class InorganicBuilderScreen:
    """High-level screen controller coordinating the inorganic builder lifecycle."""

    def __init__(self, default_metal: str = "Fe", default_oxidation: int = 2) -> None:
        self.widget = InorganicBuilderWidget()
        self.widget.set_metal(default_metal, default_oxidation)

    def get_summary(self) -> Dict[str, Any]:
        """Return active builder summary data."""
        return {
            "formula": self.widget.formula,
            "metal": self.widget.metal_symbol,
            "oxidation_state": self.widget.oxidation_state,
            "polyhedron": self.widget.polyhedron_name,
            "coordination_number": self.widget.coordination_number,
            "vacant_sites": self.widget.vacant_sites,
            "net_charge": self.widget.net_charge,
            "spin_multiplicity": self.widget.spin_multiplicity,
            "molecular_weight": self.widget.molecular_weight,
            "isomer": self.widget.isomer_state,
        }
