/**
 * CoChem Mobile 2D Chemical Sketcher - AnyWidget ESM Bundle
 * Touch-optimized, air-gapped chemical drawing canvas and serializer.
 */

export function render({ model, el }) {
  // 1. Element palette and color definitions
  const ELEMENT_COLORS = {
    C: "#c9d1d9",
    N: "#58a6ff",
    O: "#f85149",
    S: "#d29922",
    P: "#db6d28",
    F: "#7ee787",
    Cl: "#3fb950",
    Br: "#bc8cff",
    I: "#a371f7",
    H: "#8b949e",
  };

  // State
  let atoms = [];
  let bonds = [];
  let history = [];
  let selectedElement = "C";
  let activeTool = "draw"; // "draw", "single", "double", "triple", "wedge", "hash", "erase"
  let errorAtomIndices = [];
  let isDragging = false;
  let dragStartAtom = null;
  let dragCurrentPos = null;

  // Root container
  const root = document.createElement("div");
  root.className = "cochem-sketcher-root";

  // Header
  const header = document.createElement("div");
  header.className = "cochem-sketcher-header";
  header.innerHTML = `
    <div class="cochem-sketcher-title">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
        <path d="M8 0a8 8 0 100 16A8 8 0 008 0zm1 12H7V7h2v5zm0-6H7V4h2v2z"/>
      </svg>
      CoChem 2D Organic Builder
    </div>
    <div class="cochem-sketcher-status" id="status-indicator">Ready</div>
  `;
  root.appendChild(header);

  // Toolbar
  const toolbar = document.createElement("div");
  toolbar.className = "cochem-sketcher-toolbar";

  // Elements group
  const elemGroup = document.createElement("div");
  elemGroup.className = "cochem-toolbar-group";
  const elements = ["C", "N", "O", "S", "P", "F", "Cl", "Br"];
  elements.forEach((sym) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `cochem-tool-btn elem-btn ${sym === selectedElement ? "active" : ""}`;
    btn.textContent = sym;
    btn.onclick = () => {
      selectedElement = sym;
      toolbar.querySelectorAll(".elem-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      if (activeTool === "erase") {
        setTool("draw");
      }
    };
    elemGroup.appendChild(btn);
  });
  toolbar.appendChild(elemGroup);

  // Bond & Mode tools group
  const modeGroup = document.createElement("div");
  modeGroup.className = "cochem-toolbar-group";

  const tools = [
    { id: "draw", label: "━" },
    { id: "double", label: "═" },
    { id: "triple", label: "≡" },
    { id: "wedge", label: "▲" },
    { id: "hash", label: "▤" },
    { id: "erase", label: "✕", danger: true },
  ];

  function setTool(toolId) {
    activeTool = toolId;
    toolbar.querySelectorAll(".tool-mode-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.tool === toolId);
    });
  }

  tools.forEach((t) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `cochem-tool-btn tool-mode-btn ${t.id === activeTool ? "active" : ""} ${t.danger ? "danger" : ""}`;
    btn.dataset.tool = t.id;
    btn.textContent = t.label;
    btn.title = t.id;
    btn.onclick = () => setTool(t.id);
    modeGroup.appendChild(btn);
  });
  toolbar.appendChild(modeGroup);

  // Action group
  const actGroup = document.createElement("div");
  actGroup.className = "cochem-toolbar-group";

  const undoBtn = document.createElement("button");
  undoBtn.type = "button";
  undoBtn.className = "cochem-tool-btn";
  undoBtn.textContent = "↶";
  undoBtn.title = "Undo";
  undoBtn.onclick = () => {
    if (history.length > 0) {
      const prev = history.pop();
      atoms = JSON.parse(JSON.stringify(prev.atoms));
      bonds = JSON.parse(JSON.stringify(prev.bonds));
      errorAtomIndices = [];
      redraw();
    }
  };
  actGroup.appendChild(undoBtn);

  const clearBtn = document.createElement("button");
  clearBtn.type = "button";
  clearBtn.className = "cochem-tool-btn danger";
  clearBtn.textContent = "Clear";
  clearBtn.onclick = () => {
    saveHistory();
    atoms = [];
    bonds = [];
    errorAtomIndices = [];
    redraw();
    updateFooterInfo("", null, "");
  };
  actGroup.appendChild(clearBtn);

  const finalizeBtn = document.createElement("button");
  finalizeBtn.type = "button";
  finalizeBtn.className = "cochem-tool-btn primary";
  finalizeBtn.textContent = "Finalize 3D";
  finalizeBtn.onclick = () => finalizeStructure();
  actGroup.appendChild(finalizeBtn);

  toolbar.appendChild(actGroup);
  root.appendChild(toolbar);

  // Canvas Container
  const canvasContainer = document.createElement("div");
  canvasContainer.className = "cochem-canvas-container";

  const canvas = document.createElement("canvas");
  canvas.className = "cochem-canvas";
  canvasContainer.appendChild(canvas);
  root.appendChild(canvasContainer);

  // Footer
  const footer = document.createElement("div");
  footer.className = "cochem-sketcher-footer";
  footer.innerHTML = `
    <div class="cochem-info-row">
      <span class="cochem-smiles-preview" id="smiles-display">SMILES: (empty)</span>
      <span class="cochem-energy-preview" id="energy-display"></span>
    </div>
    <div class="cochem-diag-box" id="diag-box">Draw a 2D organic molecule and tap Finalize 3D.</div>
  `;
  root.appendChild(footer);

  el.appendChild(root);

  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;

  function resizeCanvas() {
    const rect = canvasContainer.getBoundingClientRect();
    const width = rect.width || 560;
    const height = rect.height || 380;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    redraw();
  }

  window.addEventListener("resize", resizeCanvas);
  setTimeout(resizeCanvas, 50);

  function saveHistory() {
    history.push({
      atoms: JSON.parse(JSON.stringify(atoms)),
      bonds: JSON.parse(JSON.stringify(bonds)),
    });
    if (history.length > 20) history.shift();
  }

  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  }

  function findAtomAt(x, y, radius = 22) {
    for (let i = 0; i < atoms.length; i++) {
      const a = atoms[i];
      const dist = Math.hypot(a.x - x, a.y - y);
      if (dist <= radius) return { atom: a, index: i };
    }
    return null;
  }

  function findBondAt(x, y, threshold = 12) {
    for (let i = 0; i < bonds.length; i++) {
      const b = bonds[i];
      const a1 = atoms[b.source];
      const a2 = atoms[b.target];
      if (!a1 || !a2) continue;

      const l2 = (a1.x - a2.x) ** 2 + (a1.y - a2.y) ** 2;
      if (l2 === 0) continue;
      let t = ((x - a1.x) * (a2.x - a1.x) + (y - a1.y) * (a2.y - a1.y)) / l2;
      t = Math.max(0, Math.min(1, t));
      const px = a1.x + t * (a2.x - a1.x);
      const py = a1.y + t * (a2.y - a1.y);
      const dist = Math.hypot(x - px, y - py);
      if (dist <= threshold) return { bond: b, index: i };
    }
    return null;
  }

  // Pointer Event Handlers with passive: false and preventDefault()
  canvas.addEventListener(
    "pointerdown",
    (e) => {
      e.preventDefault();
      try {
        canvas.setPointerCapture(e.pointerId);
      } catch (err) {}

      const pos = getCanvasCoords(e);
      const hitAtom = findAtomAt(pos.x, pos.y);
      const hitBond = findBondAt(pos.x, pos.y);

      if (activeTool === "erase") {
        saveHistory();
        if (hitAtom) {
          const idx = hitAtom.index;
          atoms.splice(idx, 1);
          bonds = bonds
            .filter((b) => b.source !== idx && b.target !== idx)
            .map((b) => ({
              ...b,
              source: b.source > idx ? b.source - 1 : b.source,
              target: b.target > idx ? b.target - 1 : b.target,
            }));
          errorAtomIndices = [];
          redraw();
        } else if (hitBond) {
          bonds.splice(hitBond.index, 1);
          redraw();
        }
        return;
      }

      isDragging = true;
      if (hitAtom) {
        dragStartAtom = hitAtom;
        dragCurrentPos = pos;
      } else {
        saveHistory();
        const newIdx = atoms.length;
        const newAtom = {
          id: newIdx,
          symbol: selectedElement,
          x: pos.x,
          y: pos.y,
          charge: 0,
        };
        atoms.push(newAtom);
        dragStartAtom = { atom: newAtom, index: newIdx };
        dragCurrentPos = pos;
      }
      redraw();
    },
    { passive: false }
  );

  canvas.addEventListener(
    "pointermove",
    (e) => {
      e.preventDefault();
      if (!isDragging) return;
      dragCurrentPos = getCanvasCoords(e);
      redraw();
    },
    { passive: false }
  );

  canvas.addEventListener(
    "pointerup",
    (e) => {
      e.preventDefault();
      try {
        canvas.releasePointerCapture(e.pointerId);
      } catch (err) {}

      if (!isDragging) return;
      isDragging = false;

      const pos = getCanvasCoords(e);
      const hitEndAtom = findAtomAt(pos.x, pos.y);

      if (dragStartAtom) {
        const startIdx = dragStartAtom.index;
        let endIdx = -1;

        if (hitEndAtom && hitEndAtom.index !== startIdx) {
          endIdx = hitEndAtom.index;
        } else if (!hitEndAtom) {
          const dist = Math.hypot(pos.x - dragStartAtom.atom.x, pos.y - dragStartAtom.atom.y);
          if (dist > 15) {
            saveHistory();
            endIdx = atoms.length;
            atoms.push({
              id: endIdx,
              symbol: selectedElement,
              x: pos.x,
              y: pos.y,
              charge: 0,
            });
          }
        }

        if (endIdx >= 0 && startIdx !== endIdx) {
          saveHistory();
          const existingBondIdx = bonds.findIndex(
            (b) =>
              (b.source === startIdx && b.target === endIdx) ||
              (b.source === endIdx && b.target === startIdx)
          );

          let bType = 1;
          let stereo = 0;
          if (activeTool === "double") bType = 2;
          else if (activeTool === "triple") bType = 3;
          else if (activeTool === "wedge") {
            bType = 1;
            stereo = 1;
          } else if (activeTool === "hash") {
            bType = 1;
            stereo = 6;
          }

          if (existingBondIdx >= 0) {
            bonds[existingBondIdx].type = bType;
            bonds[existingBondIdx].stereo = stereo;
          } else {
            bonds.push({
              source: startIdx,
              target: endIdx,
              type: bType,
              stereo: stereo,
            });
          }
        }
      }

      dragStartAtom = null;
      dragCurrentPos = null;
      redraw();
    },
    { passive: false }
  );

  canvas.addEventListener(
    "pointercancel",
    (e) => {
      e.preventDefault();
      try {
        canvas.releasePointerCapture(e.pointerId);
      } catch (err) {}
      isDragging = false;
      dragStartAtom = null;
      dragCurrentPos = null;
      redraw();
    },
    { passive: false }
  );

  // Drawing Canvas Elements
  function redraw() {
    const rect = canvasContainer.getBoundingClientRect();
    const w = rect.width || 560;
    const h = rect.height || 380;
    ctx.clearRect(0, 0, w, h);

    // Subtle background grid
    ctx.strokeStyle = "#161b22";
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 30) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Draw Bonds
    bonds.forEach((b) => {
      const a1 = atoms[b.source];
      const a2 = atoms[b.target];
      if (!a1 || !a2) return;
      drawBond(a1, a2, b.type, b.stereo);
    });

    // Draw Active Drag Line
    if (isDragging && dragStartAtom && dragCurrentPos) {
      ctx.beginPath();
      ctx.strokeStyle = "#58a6ff";
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.moveTo(dragStartAtom.atom.x, dragStartAtom.atom.y);
      ctx.lineTo(dragCurrentPos.x, dragCurrentPos.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw Atoms
    atoms.forEach((a, idx) => {
      drawAtom(a, idx);
    });
  }

  function drawBond(a1, a2, type, stereo) {
    const dx = a2.x - a1.x;
    const dy = a2.y - a1.y;
    const len = Math.hypot(dx, dy);
    if (len === 0) return;

    const nx = -dy / len;
    const ny = dx / len;

    ctx.save();
    ctx.strokeStyle = "#c9d1d9";
    ctx.fillStyle = "#c9d1d9";
    ctx.lineWidth = 3;
    ctx.lineCap = "round";

    if (stereo === 1) {
      // Wedge bond (filled polygon)
      const w = 5;
      ctx.beginPath();
      ctx.moveTo(a1.x, a1.y);
      ctx.lineTo(a2.x + nx * w, a2.y + ny * w);
      ctx.lineTo(a2.x - nx * w, a2.y - ny * w);
      ctx.closePath();
      ctx.fill();
    } else if (stereo === 6) {
      // Hash bond (series of parallel lines)
      const steps = 6;
      for (let s = 1; s <= steps; s++) {
        const t = s / steps;
        const cx = a1.x + t * dx;
        const cy = a1.y + t * dy;
        const halfW = (t * 6);
        ctx.beginPath();
        ctx.moveTo(cx - nx * halfW, cy - ny * halfW);
        ctx.lineTo(cx + nx * halfW, cy + ny * halfW);
        ctx.stroke();
      }
    } else if (type === 2) {
      // Double bond
      const offset = 3;
      ctx.beginPath();
      ctx.moveTo(a1.x + nx * offset, a1.y + ny * offset);
      ctx.lineTo(a2.x + nx * offset, a2.y + ny * offset);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(a1.x - nx * offset, a1.y - ny * offset);
      ctx.lineTo(a2.x - nx * offset, a2.y - ny * offset);
      ctx.stroke();
    } else if (type === 3) {
      // Triple bond
      ctx.beginPath();
      ctx.moveTo(a1.x, a1.y);
      ctx.lineTo(a2.x, a2.y);
      ctx.stroke();

      const offset = 4.5;
      ctx.beginPath();
      ctx.moveTo(a1.x + nx * offset, a1.y + ny * offset);
      ctx.lineTo(a2.x + nx * offset, a2.y + ny * offset);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(a1.x - nx * offset, a1.y - ny * offset);
      ctx.lineTo(a2.x - nx * offset, a2.y - ny * offset);
      ctx.stroke();
    } else {
      // Single bond
      ctx.beginPath();
      ctx.moveTo(a1.x, a1.y);
      ctx.lineTo(a2.x, a2.y);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawAtom(atom, idx) {
    const isError = errorAtomIndices.includes(idx);
    const color = ELEMENT_COLORS[atom.symbol] || "#c9d1d9";

    // Atom background circle
    ctx.beginPath();
    ctx.arc(atom.x, atom.y, 14, 0, Math.PI * 2);
    ctx.fillStyle = "#0d1117";
    ctx.fill();

    // Red error bounding box
    if (isError) {
      ctx.save();
      ctx.strokeStyle = "#f85149";
      ctx.lineWidth = 2.5;
      ctx.shadowColor = "#f85149";
      ctx.shadowBlur = 8;
      ctx.strokeRect(atom.x - 14, atom.y - 14, 28, 28);
      ctx.restore();
    }

    // Atom symbol text
    ctx.save();
    ctx.font = "bold 15px sans-serif";
    ctx.fillStyle = color;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(atom.symbol, atom.x, atom.y);

    if (atom.charge !== 0) {
      ctx.font = "bold 10px sans-serif";
      const chgStr = atom.charge > 0 ? (atom.charge === 1 ? "+" : `+${atom.charge}`) : (atom.charge === -1 ? "-" : `${atom.charge}`);
      ctx.fillText(chgStr, atom.x + 10, atom.y - 8);
    }
    ctx.restore();
  }

  // Molfile V2000 Serializer
  function serializeMolfileV2000() {
    if (atoms.length === 0) return "";

    // Normalize coordinates around centroid and scale to standard ~1.5 Angstrom bond length
    let cx = 0, cy = 0;
    atoms.forEach((a) => {
      cx += a.x;
      cy += a.y;
    });
    cx /= atoms.length;
    cy /= atoms.length;

    const scale = 40.0; // 40px per Angstrom

    const lines = [];
    lines.push("CoChem Mobile 2D Structure");
    lines.push("  CoChem-Mobile       2D");
    lines.push("");

    const na = String(atoms.length).padStart(3, " ");
    const nb = String(bonds.length).padStart(3, " ");
    lines.push(`${na}${nb}  0  0  0  0  0  0  0  0999 V2000`);

    // Atom lines
    atoms.forEach((a) => {
      const ax = ((a.x - cx) / scale).toFixed(4).padStart(10, " ");
      const ay = (-(a.y - cy) / scale).toFixed(4).padStart(10, " ");
      const az = (0.0).toFixed(4).padStart(10, " ");
      const sym = a.symbol.padEnd(3, " ");
      lines.push(`${ax}${ay}${az} ${sym} 0  0  0  0  0  0  0  0  0  0  0  0`);
    });

    // Bond lines
    bonds.forEach((b) => {
      const src = String(b.source + 1).padStart(3, " ");
      const tgt = String(b.target + 1).padStart(3, " ");
      const btype = String(b.type || 1).padStart(3, " ");
      const bstereo = String(b.stereo || 0).padStart(3, " ");
      lines.push(`${src}${tgt}${btype}${bstereo}  0  0  0`);
    });

    lines.push("M  END");
    return lines.join("\n");
  }

  function serializeAtoms2D() {
    return atoms.map((a, idx) => ({
      atom_index: idx,
      symbol: a.symbol,
      x: parseFloat(a.x.toFixed(2)),
      y: parseFloat(a.y.toFixed(2)),
      charge: a.charge || 0,
    }));
  }

  function finalizeStructure() {
    if (atoms.length === 0) {
      updateFooterInfo("Structure is empty.", null, "Please draw a molecule first.");
      return;
    }

    const molfile = serializeMolfileV2000();
    const atoms2d = serializeAtoms2D();

    const payload = {
      smiles: atoms.length === 1 ? atoms[0].symbol : "STRUCTURE_PENDING",
      molfile_v2000: molfile,
      chiral_centers_count: 0,
      atoms_2d: atoms2d,
    };

    model.set("payload_json", JSON.stringify(payload));
    model.save_changes();
  }

  function updateFooterInfo(smiles, energy, diag, isError = false) {
    const smilesEl = footer.querySelector("#smiles-display");
    const energyEl = footer.querySelector("#energy-display");
    const diagEl = footer.querySelector("#diag-box");

    if (smiles) smilesEl.textContent = `SMILES: ${smiles}`;
    if (energy !== null && energy !== undefined) {
      energyEl.textContent = `${energy.toFixed(4)} kcal/mol`;
    } else {
      energyEl.textContent = "";
    }

    if (diag) {
      diagEl.textContent = diag;
      diagEl.className = `cochem-diag-box ${isError ? "error" : "success"}`;
    }
  }

  // Model Event Listeners
  model.on("change:busy", () => {
    const isBusy = model.get("busy");
    const statusEl = header.querySelector("#status-indicator");
    if (isBusy) {
      statusEl.innerHTML = '<span class="cochem-spinner"></span> Synthesizing 3D...';
    } else {
      statusEl.textContent = "Ready";
    }
  });

  model.on("change:validation_json", () => {
    const valJson = model.get("validation_json");
    if (!valJson) return;
    try {
      const val = JSON.parse(valJson);
      if (!val.success) {
        errorAtomIndices = val.atom_error_indices || [];
        updateFooterInfo(null, null, val.diagnostic_message, true);
      } else {
        errorAtomIndices = [];
      }
      redraw();
    } catch (e) {}
  });

  model.on("change:conformer_json", () => {
    const confJson = model.get("conformer_json");
    if (!confJson) return;
    try {
      const conf = JSON.parse(confJson);
      if (conf.success) {
        updateFooterInfo(
          conf.smiles,
          conf.energy_kcal_mol,
          `Conformer Ready (${conf.force_field_used}): ${conf.coordinates_3d.length} atoms generated.`,
          false
        );
      }
    } catch (e) {}
  });
}
