"""Chemical Exception & Crash Reporter Translator.

Module: cochem.mobile.crash_reporter
Authoritative Reference: SRS Chunk 02 BASE UI & Web (Part 2), Prompt 3.

Translates low-level numerical exceptions and electronic structure engine faults
(ORCA, CREST, PySCF, xTB) into sanitized, human-readable pedagogical diagnostics
for undergraduate students and research assistants:
1. SCF Non-Convergence (SCFConvergenceError): Remediation via level-shifting, DIIS damping, or UHF.
2. Singular Overlap Matrix (SingularBasisError): Linear dependency identification and basis pruning.
3. Geometry Gradient Explosion (GeometryGradientCrash): Coordinate displacement detection and xTB pre-opt.
4. Wavefunction Instability (WavefunctionInstabilityError): Unrestricted/broken-symmetry recommendations.
5. Path sanitization: Strips workstation usernames, filesystem paths, and environment tokens.
6. Structured Pydantic reporting via ChemicalDiagnosticReport.
"""

from __future__ import annotations

import logging
import re
import sys
import traceback
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class ChemicalFaultCategory(str, Enum):
    """Classification of quantum chemical and molecular modeling faults."""

    SCF_CONVERGENCE = "SCF_CONVERGENCE"
    SINGULAR_BASIS = "SINGULAR_BASIS"
    GRADIENT_EXPLOSION = "GRADIENT_EXPLOSION"
    WAVEFUNCTION_INSTABILITY = "WAVEFUNCTION_INSTABILITY"
    ZERO_DIVISION_OR_OVERFLOW = "ZERO_DIVISION_OR_OVERFLOW"
    UNKNOWN_CHEMICAL_FAULT = "UNKNOWN_CHEMICAL_FAULT"


class ChemicalEngineError(Exception):
    """Base exception for chemical modeling and electronic structure engine faults."""

    def __init__(self, message: str, engine: Optional[str] = None) -> None:
        super().__init__(message)
        self.engine = engine


class SCFConvergenceError(ChemicalEngineError):
    """Raised when Self-Consistent Field (SCF) iterations fail to converge within maximum cycles."""

    def __init__(
        self,
        message: str = "Self-Consistent Field (SCF) iterations did not converge.",
        engine: Optional[str] = None,
        cycles_completed: Optional[int] = None,
        energy_delta: Optional[float] = None,
    ) -> None:
        super().__init__(message, engine=engine)
        self.cycles_completed = cycles_completed
        self.energy_delta = energy_delta


class SingularBasisError(ChemicalEngineError):
    """Raised when atomic orbital basis set exhibits near-linear dependency (singular overlap matrix S)."""

    def __init__(
        self,
        message: str = "Overlap matrix S is singular due to linear dependencies in diffuse basis functions.",
        engine: Optional[str] = None,
        smallest_eigenvalue: Optional[float] = None,
    ) -> None:
        super().__init__(message, engine=engine)
        self.smallest_eigenvalue = smallest_eigenvalue


class GeometryGradientCrash(ChemicalEngineError):
    """Raised when geometry optimization step size explodes or nuclear gradients exceed physical limits."""

    def __init__(
        self,
        message: str = "Nuclear gradient explosion detected during geometry optimization step.",
        engine: Optional[str] = None,
        max_gradient_au: Optional[float] = None,
    ) -> None:
        super().__init__(message, engine=engine)
        self.max_gradient_au = max_gradient_au


class WavefunctionInstabilityError(ChemicalEngineError):
    """Raised when restricted wavefunction possesses negative Hessian eigenvalues indicating internal instability."""

    def __init__(
        self,
        message: str = "Wavefunction stability analysis detected internal RHF->UHF instability.",
        engine: Optional[str] = None,
    ) -> None:
        super().__init__(message, engine=engine)


class ChemicalDiagnosticReport(BaseModel):
    """Pedagogical diagnostic report formatted for student-facing Mobile UI."""

    model_config = ConfigDict(frozen=True)

    report_id: str
    timestamp_utc: str
    category: ChemicalFaultCategory
    engine: str
    headline: str
    pedagogical_explanation: str
    actionable_remediation: List[str]
    input_adjustment_suggestions: Dict[str, Any]
    sanitized_traceback: str
    original_exception: Optional[str] = None


def sanitize_traceback_text(raw_text: str) -> str:
    """Sanitize stack traces and log texts, removing local usernames, system directories, and sensitive tokens.

    Preserves function names, line numbers, and chemical tensor identifiers.
    """
    if not raw_text:
        return ""

    sanitized = raw_text

    # Strip Windows User profiles: C:\Users\<name>\... -> <WORKSPACE>\...
    sanitized = re.sub(
        r"[A-Za-z]:\\[Uu]sers\\[^\\]+\\",
        "<USER_HOME>/",
        sanitized,
    )
    # Strip Linux/macOS user paths: /home/<name>/... or /Users/<name>/... -> <WORKSPACE>/...
    sanitized = re.sub(
        r"/(home|Users)/[^/]+/",
        "<USER_HOME>/",
        sanitized,
    )
    # Strip ephemeral execution temp sandboxes: /tmp/cochem_exec_<uuid>/ -> <SCRATCH>/
    sanitized = re.sub(
        r"(/tmp|[A-Za-z]:[/\\][Tt]emp)[/\\]cochem_exec_[0-9a-fA-F-]+[/\\]?",
        "<SCRATCH>/",
        sanitized,
    )

    # Normalize backslashes in paths
    sanitized = sanitized.replace("\\", "/")

    return sanitized


class DiagnosticKnowledgeBase:
    """Authoritative mapping of quantum chemistry error signatures to pedagogical remediations."""

    @staticmethod
    def get_scf_convergence_diagnostic() -> Tuple[str, str, List[str], Dict[str, Any]]:
        headline = "Electronic SCF Iterations Did Not Converge"
        explanation = (
            "The electronic Self-Consistent Field (SCF) procedure iteratively solves the Roothaan-Hall equations "
            "to find the minimum electronic energy. When convergence fails, the electron density oscillates or "
            "sloshes between nearly degenerate molecular orbitals, common in transition metal complexes, radicals, "
            "or systems with small HOMO-LUMO gaps."
        )
        remediation = [
            "Switch from Restricted Hartree-Fock (RHF) to Unrestricted (UHF) if your molecule has unpaired electrons.",
            "Apply electronic level-shifting (e.g., LevelShift 0.25 au in ORCA) to artificially widen the virtual orbital gap during early cycles.",
            "Increase DIIS damping or activate Second-Order SCF (SOSCF) convergence acceleration.",
            "Verify molecular charge and spin multiplicity (2S+1) are chemically correct.",
        ]
        adjustments = {
            "scf_max_cycles": 150,
            "level_shift_au": 0.25,
            "diis_damping": 0.2,
            "convergence_scheme": "SOSCF",
            "check_spin_multiplicity": True,
        }
        return headline, explanation, remediation, adjustments

    @staticmethod
    def get_singular_basis_diagnostic(smallest_eval: Optional[float] = None) -> Tuple[str, str, List[str], Dict[str, Any]]:
        headline = "Linear Dependency Detected in Basis Set (Singular Overlap Matrix S)"
        eval_str = f" (Smallest eigenvalue: {smallest_eval:.2e})" if smallest_eval is not None else ""
        explanation = (
            f"The atomic orbital overlap matrix S has nearly linearly dependent rows{eval_str}. "
            "This occurs when using large basis sets with multiple diffuse functions (e.g., aug-cc-pVTZ or def2-QZVPPD) "
            "on closely packed atoms. When two diffuse Gaussians overlap almost identically, S cannot be inverted."
        )
        remediation = [
            "Prune diffuse augmentation functions from hydrogen atoms or non-anionic heavy atoms.",
            "Switch to a moderately sized basis set (e.g., def2-TZVP instead of aug-cc-pVQZ).",
            "Increase the linear dependency threshold in your quantum chemistry input (e.g., Thresh 1e-6 in ORCA).",
            "Inspect molecular 3D geometry for unnatural interatomic bond clashes (r_ij < 0.7 Å).",
        ]
        adjustments = {
            "basis_set_recommended": "def2-TZVP",
            "drop_diffuse_hydrogens": True,
            "s_threshold": 1e-6,
            "check_interatomic_clashes": True,
        }
        return headline, explanation, remediation, adjustments

    @staticmethod
    def get_gradient_explosion_diagnostic(grad_val: Optional[float] = None) -> Tuple[str, str, List[str], Dict[str, Any]]:
        headline = "Geometry Optimization Nuclear Force Explosion"
        grad_str = f" (Force norm: {grad_val:.2f} au)" if grad_val is not None else ""
        explanation = (
            f"Nuclear gradients during the geometry optimization step exceeded stable numerical thresholds{grad_str}. "
            "This typically happens when initial atoms are placed inside each other's repulsive cores, causing the nuclear "
            "repulsion gradient to spike, or when using Cartesian coordinates for floppy or ring systems instead of redundant internal coordinates."
        )
        remediation = [
            "Pre-optimize the molecular geometry with fast semiempirical xTB (GFN2-xTB) or forcefields (MMFF94) before DFT.",
            "Convert coordinate system to Redundant Internal Coordinates (RIC) to prevent torsional step explosions.",
            "Reduce the optimization trust radius (e.g., Trust 0.1 in ORCA %geom) to enforce smaller step sizes.",
            "Verify all bond lengths in your input file are within realistic chemical ranges (1.0 to 2.5 Å).",
        ]
        adjustments = {
            "pre_opt_engine": "GFN2-xTB",
            "coordinate_system": "RedundantInternals",
            "max_step_radius_bohr": 0.1,
            "enforce_bond_length_check": True,
        }
        return headline, explanation, remediation, adjustments

    @staticmethod
    def get_wavefunction_instability_diagnostic() -> Tuple[str, str, List[str], Dict[str, Any]]:
        headline = "Negative Wavefunction Stability (Internal Instability Detected)"
        explanation = (
            "A wavefunction stability analysis indicated that the current closed-shell solution is not a true local minimum "
            "on the electronic potential energy surface. There is a lower-energy electronic state with broken spatial or spin symmetry."
        )
        remediation = [
            "Switch calculation from Restricted (RHF/RKS) to Unrestricted (UHF/UKS) or Broken-Symmetry (BS).",
            "Mix the HOMO and LUMO orbitals to generate an initial guess that breaks spin symmetry.",
            "Confirm whether the system exhibits diradical or antiferromagnetically coupled spin centers.",
        ]
        adjustments = {
            "unrestricted": True,
            "broken_symmetry": True,
            "homo_lumo_mix": True,
        }
        return headline, explanation, remediation, adjustments

    @staticmethod
    def get_generic_diagnostic(error_name: str) -> Tuple[str, str, List[str], Dict[str, Any]]:
        headline = f"Quantum Chemistry Calculation Fault: {error_name}"
        explanation = (
            f"The calculation encountered an unclassified error ({error_name}). "
            "This can occur from numerical overflow, missing engine binaries, or unsupported input keywords."
        )
        remediation = [
            "Check that input coordinates and atomic symbols adhere strictly to IUPAC specifications.",
            "Ensure that total charge and spin multiplicity are chemically consistent.",
            "Consult the Method Matrix for recommended basis set and functional pairings.",
        ]
        adjustments = {
            "method_matrix_audit_required": True,
        }
        return headline, explanation, remediation, adjustments


class ChemicalCrashTranslator:
    """Translates Python exceptions and quantum chemistry solver faults into pedagogical reports."""

    def __init__(self) -> None:
        self._custom_patterns: List[
            Tuple[re.Pattern[str], ChemicalFaultCategory, str, str, List[str], Dict[str, Any]]
        ] = []
        self._original_excepthook: Optional[Any] = None
        self._register_default_patterns()

    def _register_default_patterns(self) -> None:
        """Register default regex patterns matching known electronic structure output logs."""
        # 1. SCF Convergence Failures (ORCA, PySCF, Q-Chem, xTB)
        scf_patterns = [
            r"SCF NOT CONVERGED",
            r"SCF did not converge",
            r"Iteration failed to reach convergence",
            r"ERROR: Max iterations reached in SCF",
            r"SCC did not converge",
            r"ORCA finished by error termination in SCF",
        ]
        head_scf, exp_scf, rem_scf, adj_scf = DiagnosticKnowledgeBase.get_scf_convergence_diagnostic()
        for pat in scf_patterns:
            self._custom_patterns.append(
                (re.compile(pat, re.IGNORECASE), ChemicalFaultCategory.SCF_CONVERGENCE, head_scf, exp_scf, rem_scf, adj_scf)
            )

        # 2. Singular Overlap Matrix / Linear Dependency
        s_patterns = [
            r"LINEAR DEPENDENC(Y|IES) IN BASIS SET",
            r"Overlap matrix S is singular",
            r"Smallest eigenvalue of overlap matrix",
            r"Linear dependency in basis set",
            r"Singular basis set detected",
        ]
        head_s, exp_s, rem_s, adj_s = DiagnosticKnowledgeBase.get_singular_basis_diagnostic()
        for pat in s_patterns:
            self._custom_patterns.append(
                (re.compile(pat, re.IGNORECASE), ChemicalFaultCategory.SINGULAR_BASIS, head_s, exp_s, rem_s, adj_s)
            )

        # 3. Gradient Explosion / Geometry Optimization Crash
        grad_patterns = [
            r"ORCA finished by error termination in GSTEP",
            r"Geometric step rejected: gradient too large",
            r"Nuclear gradient explosion",
            r"Coordinate explosion detected",
            r"Geometry optimization step failed",
            r"abnormal termination in xtb",
        ]
        head_g, exp_g, rem_g, adj_g = DiagnosticKnowledgeBase.get_gradient_explosion_diagnostic()
        for pat in grad_patterns:
            self._custom_patterns.append(
                (re.compile(pat, re.IGNORECASE), ChemicalFaultCategory.GRADIENT_EXPLOSION, head_g, exp_g, rem_g, adj_g)
            )

        # 4. Wavefunction Instability
        wf_patterns = [
            r"Wavefunction stability analysis indicates an instability",
            r"Negative eigenvalue in electronic Hessian",
            r"RHF->UHF instability found",
            r"Internal wavefunction instability",
        ]
        head_w, exp_w, rem_w, adj_w = DiagnosticKnowledgeBase.get_wavefunction_instability_diagnostic()
        for pat in wf_patterns:
            self._custom_patterns.append(
                (re.compile(pat, re.IGNORECASE), ChemicalFaultCategory.WAVEFUNCTION_INSTABILITY, head_w, exp_w, rem_w, adj_w)
            )

    def register_custom_pattern(
        self,
        regex_pattern: str,
        category: ChemicalFaultCategory,
        headline: str,
        explanation: str,
        remediation: List[str],
        adjustments: Dict[str, Any],
    ) -> None:
        """Register a domain-specific regex pattern for custom solver translation."""
        compiled = re.compile(regex_pattern, re.IGNORECASE)
        self._custom_patterns.append((compiled, category, headline, explanation, remediation, adjustments))

    def translate_exception(
        self,
        exc: BaseException,
        engine_hint: Optional[str] = None,
    ) -> ChemicalDiagnosticReport:
        """Translate a Python exception instance into a structured ChemicalDiagnosticReport."""
        now_iso = datetime.now(timezone.utc).isoformat()
        report_id = str(uuid.uuid4())
        raw_tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        sanitized_tb = sanitize_traceback_text(raw_tb)
        engine_name = engine_hint or getattr(exc, "engine", None) or "Python / Engine"

        # Explicit ChemicalEngineError subclass handling
        if isinstance(exc, SCFConvergenceError):
            head, exp, rem, adj = DiagnosticKnowledgeBase.get_scf_convergence_diagnostic()
            return ChemicalDiagnosticReport(
                report_id=report_id,
                timestamp_utc=now_iso,
                category=ChemicalFaultCategory.SCF_CONVERGENCE,
                engine=engine_name,
                headline=head,
                pedagogical_explanation=exp,
                actionable_remediation=rem,
                input_adjustment_suggestions=adj,
                sanitized_traceback=sanitized_tb,
                original_exception=f"{type(exc).__name__}: {str(exc)}",
            )

        if isinstance(exc, SingularBasisError):
            head, exp, rem, adj = DiagnosticKnowledgeBase.get_singular_basis_diagnostic(exc.smallest_eigenvalue)
            return ChemicalDiagnosticReport(
                report_id=report_id,
                timestamp_utc=now_iso,
                category=ChemicalFaultCategory.SINGULAR_BASIS,
                engine=engine_name,
                headline=head,
                pedagogical_explanation=exp,
                actionable_remediation=rem,
                input_adjustment_suggestions=adj,
                sanitized_traceback=sanitized_tb,
                original_exception=f"{type(exc).__name__}: {str(exc)}",
            )

        if isinstance(exc, GeometryGradientCrash):
            head, exp, rem, adj = DiagnosticKnowledgeBase.get_gradient_explosion_diagnostic(exc.max_gradient_au)
            return ChemicalDiagnosticReport(
                report_id=report_id,
                timestamp_utc=now_iso,
                category=ChemicalFaultCategory.GRADIENT_EXPLOSION,
                engine=engine_name,
                headline=head,
                pedagogical_explanation=exp,
                actionable_remediation=rem,
                input_adjustment_suggestions=adj,
                sanitized_traceback=sanitized_tb,
                original_exception=f"{type(exc).__name__}: {str(exc)}",
            )

        if isinstance(exc, WavefunctionInstabilityError):
            head, exp, rem, adj = DiagnosticKnowledgeBase.get_wavefunction_instability_diagnostic()
            return ChemicalDiagnosticReport(
                report_id=report_id,
                timestamp_utc=now_iso,
                category=ChemicalFaultCategory.WAVEFUNCTION_INSTABILITY,
                engine=engine_name,
                headline=head,
                pedagogical_explanation=exp,
                actionable_remediation=rem,
                input_adjustment_suggestions=adj,
                sanitized_traceback=sanitized_tb,
                original_exception=f"{type(exc).__name__}: {str(exc)}",
            )

        # Pattern scan across exception message and traceback
        combined_text = f"{str(exc)}\n{raw_tb}"
        for pat, cat, head, exp, rem, adj in self._custom_patterns:
            if pat.search(combined_text):
                return ChemicalDiagnosticReport(
                    report_id=report_id,
                    timestamp_utc=now_iso,
                    category=cat,
                    engine=engine_name,
                    headline=head,
                    pedagogical_explanation=exp,
                    actionable_remediation=rem,
                    input_adjustment_suggestions=adj,
                    sanitized_traceback=sanitized_tb,
                    original_exception=f"{type(exc).__name__}: {str(exc)}",
                )

        # Fallback generic report
        exc_name = type(exc).__name__
        head, exp, rem, adj = DiagnosticKnowledgeBase.get_generic_diagnostic(exc_name)
        return ChemicalDiagnosticReport(
            report_id=report_id,
            timestamp_utc=now_iso,
            category=ChemicalFaultCategory.UNKNOWN_CHEMICAL_FAULT,
            engine=engine_name,
            headline=head,
            pedagogical_explanation=exp,
            actionable_remediation=rem,
            input_adjustment_suggestions=adj,
            sanitized_traceback=sanitized_tb,
            original_exception=f"{exc_name}: {str(exc)}",
        )

    def translate_log_output(
        self,
        log_text: str,
        engine_hint: str = "ORCA",
    ) -> ChemicalDiagnosticReport:
        """Scan raw terminal log or engine output file content and map detected faults to a report."""
        now_iso = datetime.now(timezone.utc).isoformat()
        report_id = str(uuid.uuid4())
        sanitized_log = sanitize_traceback_text(log_text)

        for pat, cat, head, exp, rem, adj in self._custom_patterns:
            if pat.search(log_text):
                return ChemicalDiagnosticReport(
                    report_id=report_id,
                    timestamp_utc=now_iso,
                    category=cat,
                    engine=engine_hint,
                    headline=head,
                    pedagogical_explanation=exp,
                    actionable_remediation=rem,
                    input_adjustment_suggestions=adj,
                    sanitized_traceback=sanitized_log[-2000:],  # preserve tail of log
                    original_exception=f"Engine error matched pattern: {pat.pattern}",
                )

        head, exp, rem, adj = DiagnosticKnowledgeBase.get_generic_diagnostic("Unclassified Log Fault")
        return ChemicalDiagnosticReport(
            report_id=report_id,
            timestamp_utc=now_iso,
            category=ChemicalFaultCategory.UNKNOWN_CHEMICAL_FAULT,
            engine=engine_hint,
            headline=head,
            pedagogical_explanation=exp,
            actionable_remediation=rem,
            input_adjustment_suggestions=adj,
            sanitized_traceback=sanitized_log[-2000:],
            original_exception="No known quantum chemistry signature matched in log.",
        )

    def install_sys_excepthook(
        self, handler_callback: Optional[Callable[[ChemicalDiagnosticReport], None]] = None
    ) -> None:
        """Install global sys.excepthook to intercept uncaught exceptions and output reports."""
        if self._original_excepthook is None:
            self._original_excepthook = sys.excepthook

        def _custom_hook(
            exc_type: type[BaseException],
            exc_val: BaseException,
            exc_tb: Any,
        ) -> None:
            report = self.translate_exception(exc_val)
            if handler_callback:
                handler_callback(report)
            else:
                sys.stderr.write(f"\n[CHEMICAL DIAGNOSTIC REPORT: {report.category.value}]\n")
                sys.stderr.write(f"Headline:    {report.headline}\n")
                sys.stderr.write(f"Explanation: {report.pedagogical_explanation}\n")
                sys.stderr.write("Remediation Suggestions:\n")
                for item in report.actionable_remediation:
                    sys.stderr.write(f"  - {item}\n")
                sys.stderr.write("\n")

        sys.excepthook = _custom_hook

    def restore_sys_excepthook(self) -> None:
        """Restore previous sys.excepthook."""
        if self._original_excepthook is not None:
            sys.excepthook = self._original_excepthook
            self._original_excepthook = None
