"""Custom py3Dmol Jupyter Visualizer Widget for CoChem-TOPOS.

Provides safe headless environment detection, static HTML fallback representation,
interactive py3Dmol widget creation, and standalone HTML document export.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cochem.topos.graph import TopologyGraph

logger = logging.getLogger("cochem.topos.visualization")


def build_minimal_sdf(graph: TopologyGraph, coords: np.ndarray | None = None) -> str:
    """Serializes TopologyGraph and optional Cartesian coordinates into V2000 SDF string."""
    sorted_nodes = sorted(graph.nodes())
    node_to_idx = {n: i + 1 for i, n in enumerate(sorted_nodes)}
    num_nodes = len(sorted_nodes)
    num_edges = graph.number_of_edges()

    lines: list[str] = [
        "CoChem-TOPOS Molecule",
        "  ToposEngine 3.0",
        "",
        f"{num_nodes:3d}{num_edges:3d}  0  0  0  0  0  0  0  0999 V2000",
    ]

    for i, n in enumerate(sorted_nodes):
        sym = str(graph.nodes[n].get("symbol", "C"))
        if coords is not None and i < len(coords):
            x, y, z = coords[i]
        else:
            x, y, z = 0.0, 0.0, 0.0
        charge = int(graph.nodes[n].get("formal_charge", 0))
        lines.append(f"{x:10.4f}{y:10.4f}{z:10.4f} {sym:<3s} 0  {charge:3d}  0  0  0  0  0  0  0  0")

    for u, v, data in graph.edges(data=True):
        idx_u = node_to_idx[u]
        idx_v = node_to_idx[v]
        bo = int(round(float(data.get("bond_order", 1.0))))
        if bo < 1:
            bo = 1
        lines.append(f"{idx_u:3d}{idx_v:3d}  {bo:2d}  0  0  0  0")

    lines.append("M  END")
    lines.append("$$$$")
    return "\n".join(lines)


class TOPOSpy3DmolWidget:
    """Visualizer widget providing headless-safe rendering and standalone HTML export."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height

    @staticmethod
    def is_headless() -> bool:
        """Deterministically detects whether running in a headless CI/terminal environment."""
        if os.environ.get("COCHEM_HEADLESS") == "1":
            return True

        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is None:
                return True
            shell_name = ip.__class__.__name__
            if shell_name in ("TerminalInteractiveShell", "NoneType"):
                return True
            # In non-interactive batch mode or without an active kernel
            if "ipykernel" not in sys.modules and not hasattr(ip, "kernel"):
                return True
            return False
        except Exception:
            return True

    def render(
        self,
        graph: TopologyGraph,
        coords: np.ndarray | None = None,
        style: str = "stick",
    ) -> Any:
        """Renders topology. In headless mode, returns HTML string without GUI errors."""
        sdf_data = build_minimal_sdf(graph, coords)

        if self.is_headless():
            return (
                f"<div class='cochem-topos-headless' data-nodes='{graph.number_of_nodes()}' "
                f"data-edges='{graph.number_of_edges()}' style='width:{self.width}px; height:{self.height}px; "
                f"border:1px solid #ccc; padding:10px; font-family:sans-serif;'>"
                f"<h4>CoChem-TOPOS Molecular Topology (Headless Mode)</h4>"
                f"<p><strong>Nodes:</strong> {graph.number_of_nodes()} | "
                f"<strong>Edges:</strong> {graph.number_of_edges()}</p>"
                f"<pre style='font-size:10px; max-height:200px; overflow:auto;'>{sdf_data[:400]}</pre>"
                f"</div>"
            )

        try:
            import py3Dmol
            view = py3Dmol.view(width=self.width, height=self.height)
            view.addModel(sdf_data, "sdf")
            if style == "sphere":
                view.setStyle({"sphere": {}})
            elif style == "cartoon":
                view.setStyle({"cartoon": {}})
            else:
                view.setStyle({"stick": {}})
            view.zoomTo()
            return view
        except Exception as exc:
            logger.warning("py3Dmol interactive viewer construction failed (%s); falling back to HTML", exc)
            return (
                f"<div class='cochem-topos-fallback'>"
                f"<p>TopologyGraph Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}</p>"
                f"</div>"
            )

    def export_html(
        self,
        graph: TopologyGraph,
        coords: np.ndarray | None = None,
        output_path: Path | str | None = None,
    ) -> str:
        """Exports a self-contained HTML document with embedded py3Dmol viewer."""
        sdf_data = build_minimal_sdf(graph, coords)
        clean_sdf = sdf_data.replace("`", "\\`").replace("\\", "\\\\").replace("$", "\\$")

        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CoChem-TOPOS py3Dmol Visualizer</title>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <style>
    body {{ margin: 0; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    #viewport {{ width: {self.width}px; height: {self.height}px; position: relative; border: 1px solid #ddd; border-radius: 4px; }}
  </style>
</head>
<body>
  <h3>CoChem-TOPOS Molecular Topology</h3>
  <div id="viewport"></div>
  <script>
    document.addEventListener("DOMContentLoaded", function() {{
      let element = document.getElementById("viewport");
      let config = {{ backgroundColor: "white" }};
      let viewer = $3Dmol.createViewer(element, config);
      let sdfData = `{clean_sdf}`;
      viewer.addModel(sdfData, "sdf");
      viewer.setStyle({{}}, {{ stick: {{}} }});
      viewer.zoomTo();
      viewer.render();
    }});
  </script>
</body>
</html>
"""
        if output_path is not None:
            out_file = Path(output_path).resolve()
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(html_doc, encoding="utf-8")

        return html_doc
