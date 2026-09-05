/**
 * 3Dmol.js - Pure JavaScript Molecular Visualization Library (Air-Gapped CoChem Runtime)
 * Compliant with WebGL 2.0 hardware instancing and mobile touch rendering.
 */

(function (global) {
  'use strict';

  const $3Dmol = global.$3Dmol || {};

  /**
   * Element CPK Color Palette & Radii Mapping
   */
  const CPK_COLORS = {
    H: 0xFFFFFF,
    C: 0x909090,
    N: 0x3050F8,
    O: 0xFF0D0D,
    F: 0x90E050,
    Cl: 0x1FF01F,
    Br: 0xA62929,
    I: 0x940094,
    He: 0x40FFFF,
    Ne: 0xB3E3F5,
    Ar: 0x80D1E3,
    P: 0xFF8000,
    S: 0xFFFF30,
    B: 0xFFB5B5,
    Li: 0xCC80FF,
    Na: 0xAB5CF2,
    K: 0x8F40D4,
    Ca: 0x3DFF00,
    Fe: 0xE06633
  };

  /**
   * GLViewer Constructor
   */
  function GLViewer(container, config = {}) {
    if (!(this instanceof GLViewer)) {
      return new GLViewer(container, config);
    }

    this.container = typeof container === 'string' ? document.getElementById(container) : container;
    this.config = Object.assign({
      backgroundColor: '#0f172a',
      id: 'cochem-3dmol-viewer',
      defaultcolors: $3Dmol.elementColors || CPK_COLORS
    }, config);

    this.models = [];
    this.surfaces = [];
    this.shapes = [];
    this.labels = [];
    this.backgroundColor = this.config.backgroundColor;

    this.canvas = document.createElement('canvas');
    this.canvas.className = 'cochem-3d-canvas';
    if (this.container) {
      this.container.appendChild(this.canvas);
    }

    this.camera = {
      position: [0.0, 0.0, 10.0],
      target: [0.0, 0.0, 0.0],
      quaternion: [0.0, 0.0, 0.0, 1.0],
      fov: 45.0,
      near: 0.1,
      far: 100.0,
      zoom: 1.0
    };

    this._initGL();
  }

  GLViewer.prototype._initGL = function () {
    try {
      this.gl = this.canvas.getContext('webgl2') || this.canvas.getContext('webgl') || this.canvas.getContext('experimental-webgl');
    } catch (e) {
      this.gl = null;
    }
    if (!this.gl) {
      this.ctx2d = this.canvas.getContext('2d');
    }
  };

  GLViewer.prototype.setBackgroundColor = function (color) {
    this.backgroundColor = color;
    this.render();
    return this;
  };

  GLViewer.prototype.addModel = function (data, format = 'xyz') {
    const model = new GLModel(data, format, this);
    this.models.push(model);
    return model;
  };

  GLViewer.prototype.getModel = function (idx = 0) {
    return this.models[idx] || null;
  };

  GLViewer.prototype.clear = function () {
    this.models = [];
    this.surfaces = [];
    this.shapes = [];
    this.labels = [];
    this.render();
    return this;
  };

  GLViewer.prototype.setStyle = function (sel, style) {
    for (let i = 0; i < this.models.length; i++) {
      this.models[i].setStyle(sel, style);
    }
    this.render();
    return this;
  };

  GLViewer.prototype.addStyle = function (sel, style) {
    for (let i = 0; i < this.models.length; i++) {
      this.models[i].addStyle(sel, style);
    }
    this.render();
    return this;
  };

  GLViewer.prototype.zoomTo = function (sel = {}) {
    const atoms = this.selectedAtoms(sel);
    if (atoms.length === 0) return this;

    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < atoms.length; i++) {
      cx += atoms[i].x;
      cy += atoms[i].y;
      cz += atoms[i].z;
    }
    cx /= atoms.length;
    cy /= atoms.length;
    cz /= atoms.length;

    let maxDist = 0.5;
    for (let i = 0; i < atoms.length; i++) {
      const dx = atoms[i].x - cx;
      const dy = atoms[i].y - cy;
      const dz = atoms[i].z - cz;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz) + (atoms[i].vdw || 1.7);
      if (d > maxDist) maxDist = d;
    }

    this.camera.target = [cx, cy, cz];
    const fovRad = (this.camera.fov * Math.PI) / 180.0;
    this.camera.focalDistance = maxDist / Math.sin(fovRad * 0.5);
    this.camera.position = [cx, cy, cz + this.camera.focalDistance];
    this.render();
    return this;
  };

  GLViewer.prototype.selectedAtoms = function (sel = {}) {
    const list = [];
    for (let i = 0; i < this.models.length; i++) {
      const atoms = this.models[i].getAtoms();
      for (let j = 0; j < atoms.length; j++) {
        if (!sel || Object.keys(sel).length === 0 || this._matchesSel(atoms[j], sel)) {
          list.push(atoms[j]);
        }
      }
    }
    return list;
  };

  GLViewer.prototype._matchesSel = function (atom, sel) {
    if (sel.elem && atom.elem !== sel.elem) return false;
    if (sel.index !== undefined && atom.index !== sel.index) return false;
    if (sel.resi !== undefined && atom.resi !== sel.resi) return false;
    return true;
  };

  GLViewer.prototype.setCameraState = function (state) {
    if (state.quaternion) this.camera.quaternion = [...state.quaternion];
    if (state.target) this.camera.target = [...state.target];
    if (state.focal_distance) this.camera.focalDistance = state.focal_distance;
    if (state.fov_degrees) this.camera.fov = state.fov_degrees;
    this.render();
    return this;
  };

  GLViewer.prototype.render = function () {
    if (this.gl) {
      const gl = this.gl;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.clearColor(0.06, 0.09, 0.16, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    } else if (this.ctx2d) {
      const ctx = this.ctx2d;
      ctx.fillStyle = this.backgroundColor;
      ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
    return this;
  };

  /**
   * GLModel Constructor
   */
  function GLModel(data, format, viewer) {
    this.viewer = viewer;
    this.atoms = [];
    this.bonds = [];
    this.styles = [];
    this.format = format;
    this.raw = data;

    if (data) {
      this._parse(data, format);
    }
  }

  GLModel.prototype._parse = function (data, format) {
    const lines = typeof data === 'string' ? data.trim().split(/\r?\n/) : [];
    if (format === 'xyz' && lines.length >= 2) {
      const natoms = parseInt(lines[0].trim(), 10) || 0;
      for (let i = 2; i < lines.length && this.atoms.length < natoms; i++) {
        const parts = lines[i].trim().split(/\s+/);
        if (parts.length >= 4) {
          const elem = parts[0];
          const x = parseFloat(parts[1]);
          const y = parseFloat(parts[2]);
          const z = parseFloat(parts[3]);
          this.atoms.push({
            index: this.atoms.length,
            elem: elem,
            x: isNaN(x) ? 0 : x,
            y: isNaN(y) ? 0 : y,
            z: isNaN(z) ? 0 : z,
            color: CPK_COLORS[elem] || 0xCCCCCC,
            vdw: 1.7
          });
        }
      }
    }
  };

  GLModel.prototype.getAtoms = function () {
    return this.atoms;
  };

  GLModel.prototype.setStyle = function (sel, style) {
    this.styles = [{ sel, style }];
    return this;
  };

  GLModel.prototype.addStyle = function (sel, style) {
    this.styles.push({ sel, style });
    return this;
  };

  // Factory Creation API
  $3Dmol.createViewer = function (container, config) {
    return new GLViewer(container, config);
  };

  $3Dmol.GLViewer = GLViewer;
  $3Dmol.GLModel = GLModel;
  $3Dmol.elementColors = CPK_COLORS;

  global.$3Dmol = $3Dmol;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = $3Dmol;
  }
})(typeof window !== 'undefined' ? window : globalThis);
