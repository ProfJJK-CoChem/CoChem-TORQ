"""PyTorch 2.0+ TorchDynamo ONNX Edge-Export Pipeline with Dynamic Axes for CoChem-TORQ.

Method Matrix v4 Provenance Tags:
- [M] Mandated: Numerical parity verification between PyTorch eager graph and ONNX Runtime (tol = 1e-5).
- [D] Derived: Dynamic axes specification covering 3D tensors, edge_index, and analytical outputs.
- [E] Empirical: Minimum ONNX opset version >= 17 (recommended 18).

Strict Zero-Mock Mandate v3: Absolutely no stubs, empty pass blocks, or mock data.
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from Libraries.cochem_torq_inference_errors import (
    NumericalParityError,
    OpsetUnsupportedError,
)
from Libraries.cochem_torq_inference_schemas import ONNXExportSpec

# Canonical 3D Dynamic Axes Specification [D]/[E]
DEFAULT_DYNAMIC_AXES: Dict[str, Dict[int, str]] = {
    "coordinates": {0: "batch_size", 1: "num_atoms", 2: "spatial_dim"},
    "atomic_numbers": {0: "batch_size", 1: "num_atoms"},
    "edge_index": {0: "edge_direction", 1: "num_edges"},
    "energy": {0: "batch_size"},
    "homo_lumo_gap": {0: "batch_size"},
    "forces": {0: "batch_size", 1: "num_atoms", 2: "spatial_dim"},
}


def validate_onnx_spec(spec: ONNXExportSpec) -> None:
    """Validate that ONNXExportSpec conforms to PyTorch 2.0+ requirements [M].

    Parameters
    ----------
    spec : ONNXExportSpec
        Export specification contract.

    Raises
    ------
    OpsetUnsupportedError
        If opset_version < 17 or unsupported export mechanism.
    """
    if spec.opset_version < 17:
        raise OpsetUnsupportedError(
            f"Target ONNX opset {spec.opset_version} is unsupported. CoChem-TORQ requires opset >= 17.",
            error_code="TORQ_OPSET_UNSUPPORTED",
            component="onnx_export",
            diagnostics={"opset_version": spec.opset_version},
        )

    supported_mechanisms = {"dynamo_export_aot_autograd", "direct_analytical_force_head"}
    if spec.export_mechanism not in supported_mechanisms:
        raise OpsetUnsupportedError(
            f"Unsupported ONNX export mechanism: '{spec.export_mechanism}'. "
            f"Supported mechanisms: {sorted(supported_mechanisms)}.",
            error_code="TORQ_EXPORT_MECHANISM_UNSUPPORTED",
            component="onnx_export",
            diagnostics={"export_mechanism": spec.export_mechanism},
        )


def export_to_onnx(
    model: nn.Module,
    sample_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    export_path: Union[str, Path],
    spec: ONNXExportSpec,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
) -> Path:
    """Compile and export PyTorch model to an ONNX binary artifact [M]/[D].

    Parameters
    ----------
    model : nn.Module
        PyTorch model to export.
    sample_inputs : Union[torch.Tensor, Tuple[torch.Tensor, ...]]
        Representative inputs for model tracing or shape analysis.
    export_path : Union[str, Path]
        Target file path for the .onnx binary.
    spec : ONNXExportSpec
        Configuration specification governing opset version and dynamic axes.
    input_names : Optional[List[str]]
        Names for input nodes in the exported ONNX graph.
    output_names : Optional[List[str]]
        Names for output nodes in the exported ONNX graph.

    Returns
    -------
    Path
        Absolute path to the validated ONNX file.

    Raises
    ------
    OpsetUnsupportedError
        If spec.opset_version < 17 or invalid export mechanism.
    """
    validate_onnx_spec(spec)

    out_path = Path(export_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(sample_inputs, tuple):
        sample_inputs = (sample_inputs,)

    if input_names is None:
        input_names = [f"input_{i}" for i in range(len(sample_inputs))]
    if output_names is None:
        output_names = ["output_0"]

    # Filter dynamic axes to match only the specified input/output names and tensor dimensions
    filtered_dynamic_axes: Dict[str, Dict[int, str]] = {}
    for name, tensor in zip(input_names, sample_inputs):
        if name in spec.dynamic_axes:
            filtered_dynamic_axes[name] = {
                ax: ax_name
                for ax, ax_name in spec.dynamic_axes[name].items()
                if ax < tensor.ndim
            }
    for name in output_names:
        if name in spec.dynamic_axes:
            filtered_dynamic_axes[name] = spec.dynamic_axes[name]

    eval_model = model.eval()

    # Route based on spec.export_mechanism
    if spec.export_mechanism == "dynamo_export_aot_autograd":
        try:
            buffer = io.StringIO()
            err_buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(err_buffer):
                if hasattr(torch.onnx, "dynamo_export"):
                    onnx_program = torch.onnx.dynamo_export(eval_model, *sample_inputs)
                    onnx_program.save(str(out_path))
                else:
                    torch.onnx.export(
                        eval_model,
                        sample_inputs,
                        str(out_path),
                        export_params=True,
                        opset_version=spec.opset_version,
                        do_constant_folding=True,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=filtered_dynamic_axes if filtered_dynamic_axes else None,
                    )
        except Exception:
            buffer = io.StringIO()
            err_buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(err_buffer):
                torch.onnx.export(
                    eval_model,
                    sample_inputs,
                    str(out_path),
                    export_params=True,
                    opset_version=spec.opset_version,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=filtered_dynamic_axes if filtered_dynamic_axes else None,
                    dynamo=False,
                )
    elif spec.export_mechanism == "direct_analytical_force_head":
        buffer = io.StringIO()
        err_buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(err_buffer):
            torch.onnx.export(
                eval_model,
                sample_inputs,
                str(out_path),
                export_params=True,
                opset_version=spec.opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=filtered_dynamic_axes if filtered_dynamic_axes else None,
                dynamo=False,
            )
    else:
        raise OpsetUnsupportedError(
            f"Unsupported ONNX export mechanism: '{spec.export_mechanism}'. "
            f"Supported mechanisms: 'dynamo_export_aot_autograd', 'direct_analytical_force_head'.",
            error_code="TORQ_EXPORT_MECHANISM_UNSUPPORTED",
            component="onnx_export",
            diagnostics={"export_mechanism": spec.export_mechanism},
        )

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to generate ONNX artifact at {out_path}.")

    return out_path


def verify_onnx_parity(
    torch_model: nn.Module,
    onnx_path: Union[str, Path],
    sample_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    tolerance: float = 1e-5,
) -> float:
    """Validate numerical parity between PyTorch eager execution and ONNX Runtime session [M].

    Parameters
    ----------
    torch_model : nn.Module
        Eager PyTorch model.
    onnx_path : Union[str, Path]
        Path to compiled ONNX binary artifact.
    sample_inputs : Union[torch.Tensor, Tuple[torch.Tensor, ...]]
        Input tensors for parity execution.
    tolerance : float
        Maximum allowable absolute elementwise discrepancy (default 1e-5) [M].

    Returns
    -------
    float
        Maximum absolute discrepancy across all graph outputs.

    Raises
    ------
    NumericalParityError
        If discrepancy exceeds tolerance or evaluation fails.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("onnxruntime is required for verify_onnx_parity.") from exc

    if not isinstance(sample_inputs, tuple):
        sample_inputs = (sample_inputs,)

    model_eval = torch_model.eval()
    with torch.no_grad():
        torch_outputs = model_eval(*sample_inputs)

    if isinstance(torch_outputs, torch.Tensor):
        torch_outputs_list = [torch_outputs]
    elif isinstance(torch_outputs, (tuple, list)):
        torch_outputs_list = list(torch_outputs)
    else:
        raise TypeError(f"Unsupported model output type: {type(torch_outputs)}.")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_input_names = [inp.name for inp in session.get_inputs()]

    ort_feed: Dict[str, np.ndarray] = {}
    for name, tensor in zip(ort_input_names, sample_inputs):
        ort_feed[name] = tensor.detach().cpu().numpy()

    ort_outputs = session.run(None, ort_feed)

    max_absolute_error = 0.0
    for t_out, o_out in zip(torch_outputs_list, ort_outputs):
        t_np = t_out.detach().cpu().numpy()
        abs_diff = float(np.max(np.abs(t_np - o_out)))
        if abs_diff > max_absolute_error:
            max_absolute_error = abs_diff

    if max_absolute_error > tolerance:
        raise NumericalParityError(
            f"ONNX parity verification failed: maximum absolute error {max_absolute_error:.6e} "
            f"exceeds tolerance {tolerance:.6e}.",
            error_code="TORQ_ONNX_PARITY_DIVERGENCE",
            component="onnx_verifier",
            diagnostics={
                "max_absolute_error": max_absolute_error,
                "tolerance": float(tolerance),
            },
        )

    return max_absolute_error
