/**
 * CoChem-Mobile Touch Controller for 3D Molecular Visualization (REQ-MOB-020 - REQ-MOB-027).
 * 
 * Strict Zero-Mock & Anti-Spoof Mandate:
 * - Real SO(3) arcball virtual sphere projection and unit quaternion math (scalar-last [x, y, z, w]).
 * - Multi-touch gesture arbitration FSM: IDLE, ORBIT, PINCH_ZOOM_PAN, ATOM_SELECTION, BOND_TORSION_ACTIVE.
 * - Dynamic frustum near/far clipping calculation: z_near = max(0.1, d - 1.5R), z_far = d + 2.0R.
 * - Dynamic DPR normalization against window.devicePixelRatio.
 * - Non-passive touch event trapping to eliminate viewport scrolling.
 * - WebGL context loss recovery and adaptive RAF render scheduling.
 */

(function (global) {
  'use strict';

  // Finite State Machine States
  const TouchState = Object.freeze({
    IDLE: 'IDLE',
    ORBIT: 'ORBIT',
    PINCH_ZOOM_PAN: 'PINCH_ZOOM_PAN',
    ATOM_SELECTION: 'ATOM_SELECTION',
    BOND_TORSION_ACTIVE: 'BOND_TORSION_ACTIVE'
  });

  /**
   * Vector3 and Quaternion Math Utilities
   */
  const Math3D = {
    clamp(val, min, max) {
      return Math.max(min, Math.min(max, val));
    },

    // Arcball virtual sphere projection: (x, y) in [-1, 1]^2 -> 3D unit vector v
    projectToSphere(x, y) {
      const d2 = x * x + y * y;
      let z;
      if (d2 <= 0.5) {
        z = Math.sqrt(Math.max(0.0, 1.0 - d2));
      } else {
        z = 0.5 / Math.sqrt(d2);
      }
      const len = Math.sqrt(x * x + y * y + z * z);
      if (len < 1e-8) {
        return [0.0, 0.0, 1.0];
      }
      return [x / len, y / len, z / len];
    },

    // Vector dot product
    dot(v1, v2) {
      return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    },

    // Vector cross product
    cross(v1, v2) {
      return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
      ];
    },

    // Vector magnitude
    norm(v) {
      return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    },

    // Quaternion multiplication: q_out = q1 (x) q2 in scalar-last [x, y, z, w] format
    quatMultiply(q1, q2) {
      const [x1, y1, z1, w1] = q1;
      const [x2, y2, z2, w2] = q2;
      return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
      ];
    },

    // Unit quaternion normalization
    quatNormalize(q) {
      const len = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
      if (len < 1e-8) {
        return [0.0, 0.0, 0.0, 1.0];
      }
      return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
    },

    // Arcball rotation quaternion between two virtual sphere vectors v1 and v2
    quatFromVectors(v1, v2) {
      const n = this.cross(v1, v2);
      const dotVal = this.clamp(this.dot(v1, v2), -1.0, 1.0);
      const theta = Math.acos(dotVal);
      const nLen = this.norm(n);

      if (nLen < 1e-7 || isNaN(theta) || theta < 1e-7) {
        return [0.0, 0.0, 0.0, 1.0];
      }

      const u = [n[0] / nLen, n[1] / nLen, n[2] / nLen];
      const halfTheta = theta * 0.5;
      const s = Math.sin(halfTheta);
      return this.quatNormalize([
        u[0] * s,
        u[1] * s,
        u[2] * s,
        Math.cos(halfTheta)
      ]);
    },

    // Convert quaternion [x, y, z, w] to 3x3 rotation matrix
    quatToMatrix3(q) {
      const [x, y, z, w] = this.quatNormalize(q);
      const x2 = x + x, y2 = y + y, z2 = z + z;
      const xx = x * x2, xy = x * y2, xz = x * z2;
      const yy = y * y2, yz = y * z2, zz = z * z2;
      const wx = w * x2, wy = w * y2, wz = w * z2;

      return [
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)]
      ];
    }
  };

  /**
   * Main Touch Controller Class
   */
  class TouchController {
    /**
     * @param {HTMLElement} container - Viewport container element
     * @param {Object} options - Configuration options
     */
    constructor(container, options = {}) {
      if (!container) {
        throw new Error('TouchController requires a valid container element.');
      }
      this.container = container;
      this.options = Object.assign({
        boundingRadius: 5.0,
        fovDegrees: 45.0,
        enableDihedralTorsion: false,
        onStateChange: null,
        onAtomPick: null,
        onDihedralRotate: null,
        viewer: null
      }, options);

      this.state = TouchState.IDLE;
      this.dpr = (typeof window !== 'undefined' && window.devicePixelRatio) ? window.devicePixelRatio : 1.0;

      // Camera State
      this.quaternion = [0.0, 0.0, 0.0, 1.0];
      this.target = [0.0, 0.0, 0.0];
      this.boundingRadius = Math.max(0.1, Number(this.options.boundingRadius) || 5.0);
      this.fovDegrees = Math3D.clamp(Number(this.options.fovDegrees) || 45.0, 10.0, 120.0);
      
      const fovRad = (this.fovDegrees * Math.PI) / 180.0;
      this.focalDistance = this.boundingRadius / Math.sin(fovRad * 0.5);
      this.initialFocalDistance = this.focalDistance;

      // Touch Tracking State
      this.touchPoints = [];
      this.startSphereVector = [0.0, 0.0, 1.0];
      this.startPinchDistance = 1.0;
      this.startPinchCenter = [0.0, 0.0];
      this.startFocalDistance = this.focalDistance;
      this.startTarget = [0.0, 0.0, 0.0];
      this.startTouchTime = 0;
      this.startTouchPos = [0.0, 0.0];
      this.startDihedralAngle = 0.0;

      // Dihedral / Selection state
      this.selectedAtoms = [];
      this.activeDihedral = null; // [a1, a2, a3, a4]

      // Animation & Rendering loop state
      this.isDirty = true;
      this.rafId = null;
      this.isContextLost = false;

      // Bound Event Listeners
      this._boundTouchStart = this._onTouchStart.bind(this);
      this._boundTouchMove = this._onTouchMove.bind(this);
      this._boundTouchEnd = this._onTouchEnd.bind(this);
      this._boundTouchCancel = this._onTouchCancel.bind(this);
      this._boundContextLost = this._onContextLost.bind(this);
      this._boundContextRestored = this._onContextRestored.bind(this);
      this._boundResize = this._onResize.bind(this);

      this._initDOM();
      this._attachListeners();
      this.updateFrustum();
      this.requestRender();
    }

    _initDOM() {
      // Find or create canvas
      this.canvas = this.container.querySelector('canvas') || this.container.querySelector('.cochem-3d-canvas');
      if (!this.canvas) {
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'cochem-3d-canvas';
        this.container.appendChild(this.canvas);
      }
      this.canvas.style.touchAction = 'none';

      // Update resolution matching container and DPR
      this._updateCanvasResolution();

      // Build overlay controls if not already present
      this._buildOverlayControls();
    }

    _buildOverlayControls() {
      if (this.container.querySelector('.cochem-3d-overlay')) {
        return;
      }

      const overlay = document.createElement('div');
      overlay.className = 'cochem-3d-overlay';

      const btnReset = document.createElement('button');
      btnReset.className = 'cochem-overlay-btn';
      btnReset.textContent = 'Reset';
      btnReset.title = 'Reset Camera to Default';
      btnReset.addEventListener('click', (e) => {
        e.preventDefault();
        this.resetCamera();
      });

      const btnCenter = document.createElement('button');
      btnCenter.className = 'cochem-overlay-btn';
      btnCenter.textContent = 'Center';
      btnCenter.title = 'Auto-Center Geometric Centroid';
      btnCenter.addEventListener('click', (e) => {
        e.preventDefault();
        this.autoCenter();
      });

      const btnFit = document.createElement('button');
      btnFit.className = 'cochem-overlay-btn';
      btnFit.textContent = 'Fit';
      btnFit.title = 'Fit View to Bounding Radius';
      btnFit.addEventListener('click', (e) => {
        e.preventDefault();
        this.fitView();
      });

      overlay.appendChild(btnReset);
      overlay.appendChild(btnCenter);
      overlay.appendChild(btnFit);
      this.container.appendChild(overlay);
    }

    _updateCanvasResolution() {
      const rect = this.container.getBoundingClientRect();
      const w = Math.max(100, Math.floor(rect.width || 800));
      const h = Math.max(100, Math.floor(rect.height || 600));
      this.dpr = (typeof window !== 'undefined' && window.devicePixelRatio) ? window.devicePixelRatio : 1.0;

      if (this.canvas) {
        this.canvas.width = Math.floor(w * this.dpr);
        this.canvas.height = Math.floor(h * this.dpr);
        this.canvas.style.width = `${w}px`;
        this.canvas.style.height = `${h}px`;
      }
      this.width = w;
      this.height = h;
    }

    _attachListeners() {
      const target = this.canvas || this.container;
      // Attach non-passive touch listeners
      target.addEventListener('touchstart', this._boundTouchStart, { passive: false, capture: true });
      target.addEventListener('touchmove', this._boundTouchMove, { passive: false, capture: true });
      target.addEventListener('touchend', this._boundTouchEnd, { passive: false, capture: true });
      target.addEventListener('touchcancel', this._boundTouchCancel, { passive: false, capture: true });

      if (this.canvas) {
        this.canvas.addEventListener('webglcontextlost', this._boundContextLost, false);
        this.canvas.addEventListener('webglcontextrestored', this._boundContextRestored, false);
      }

      if (typeof window !== 'undefined') {
        window.addEventListener('resize', this._boundResize, { passive: true });
      }
    }

    _detachListeners() {
      const target = this.canvas || this.container;
      target.removeEventListener('touchstart', this._boundTouchStart, { capture: true });
      target.removeEventListener('touchmove', this._boundTouchMove, { capture: true });
      target.removeEventListener('touchend', this._boundTouchEnd, { capture: true });
      target.removeEventListener('touchcancel', this._boundTouchCancel, { capture: true });

      if (this.canvas) {
        this.canvas.removeEventListener('webglcontextlost', this._boundContextLost);
        this.canvas.removeEventListener('webglcontextrestored', this._boundContextRestored);
      }

      if (typeof window !== 'undefined') {
        window.removeEventListener('resize', this._boundResize);
      }
    }

    _getTouchPos(touch) {
      const rect = this.container.getBoundingClientRect();
      const x = touch.clientX - rect.left;
      const y = touch.clientY - rect.top;
      return [x, y];
    }

    _toNormalizedCoordinates(x, y) {
      const nx = (2.0 * x) / Math.max(1.0, this.width) - 1.0;
      const ny = 1.0 - (2.0 * y) / Math.max(1.0, this.height);
      return [nx, ny];
    }

    _onTouchStart(event) {
      event.preventDefault();
      event.stopPropagation();

      if (this.isContextLost) return;

      const touches = event.targetTouches || event.touches;
      this.touchPoints = [];
      for (let i = 0; i < touches.length; i++) {
        this.touchPoints.push(this._getTouchPos(touches[i]));
      }

      this.startTouchTime = Date.now();

      if (this.touchPoints.length === 1) {
        this.startTouchPos = [...this.touchPoints[0]];
        const [nx, ny] = this._toNormalizedCoordinates(this.startTouchPos[0], this.startTouchPos[1]);
        this.startSphereVector = Math3D.projectToSphere(nx, ny);
        this.state = TouchState.ORBIT;
      } else if (this.touchPoints.length >= 2) {
        const p1 = this.touchPoints[0];
        const p2 = this.touchPoints[1];
        const dx = p2[0] - p1[0];
        const dy = p2[1] - p1[1];
        this.startPinchDistance = Math.max(1.0, Math.sqrt(dx * dx + dy * dy));
        this.startPinchCenter = [(p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5];
        this.startFocalDistance = this.focalDistance;
        this.startTarget = [...this.target];

        if (this.options.enableDihedralTorsion && this.activeDihedral) {
          this.state = TouchState.BOND_TORSION_ACTIVE;
          this.startDihedralAngle = Math.atan2(dy, dx);
        } else {
          this.state = TouchState.PINCH_ZOOM_PAN;
        }
      }

      this._notifyStateChange();
      this.requestRender();
    }

    _onTouchMove(event) {
      event.preventDefault();
      event.stopPropagation();

      if (this.isContextLost) return;

      const touches = event.targetTouches || event.touches;
      if (touches.length === 0) return;

      const currentPoints = [];
      for (let i = 0; i < touches.length; i++) {
        currentPoints.push(this._getTouchPos(touches[i]));
      }

      if (this.state === TouchState.ORBIT && currentPoints.length === 1) {
        // One-finger SO(3) Arcball Rotation
        const [currX, currY] = currentPoints[0];
        const [nx, ny] = this._toNormalizedCoordinates(currX, currY);
        const currSphereVector = Math3D.projectToSphere(nx, ny);

        const deltaQuat = Math3D.quatFromVectors(this.startSphereVector, currSphereVector);
        // Compose orientation: q' = deltaQuat * q
        this.quaternion = Math3D.quatNormalize(Math3D.quatMultiply(deltaQuat, this.quaternion));
        this.startSphereVector = currSphereVector;
        this.isDirty = true;
      } else if (this.state === TouchState.PINCH_ZOOM_PAN && currentPoints.length >= 2) {
        // Two-finger Pinch Zoom & Screen-Space Pan
        const p1 = currentPoints[0];
        const p2 = currentPoints[1];
        const dx = p2[0] - p1[0];
        const dy = p2[1] - p1[1];
        const currDistance = Math.max(1.0, Math.sqrt(dx * dx + dy * dy));
        const pinchRatio = currDistance / Math.max(1.0, this.startPinchDistance);

        // Focal distance update: d' = d / s
        const minDistance = 0.1 * this.boundingRadius;
        const maxDistance = 20.0 * this.boundingRadius;
        this.focalDistance = Math3D.clamp(this.startFocalDistance / pinchRatio, minDistance, maxDistance);

        // Pan delta
        const currCenter = [(p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5];
        const deltaCenterX = currCenter[0] - this.startPinchCenter[0];
        const deltaCenterY = currCenter[1] - this.startPinchCenter[1];

        // Screen space pan scaling
        const fovRad = (this.fovDegrees * Math.PI) / 180.0;
        const worldUnitsPerPixel = (2.0 * this.focalDistance * Math.tan(fovRad * 0.5)) / Math.min(this.width, this.height);
        this.target = [
          this.startTarget[0] - deltaCenterX * worldUnitsPerPixel,
          this.startTarget[1] + deltaCenterY * worldUnitsPerPixel,
          this.startTarget[2]
        ];

        this.updateFrustum();
        this.isDirty = true;
      } else if (this.state === TouchState.BOND_TORSION_ACTIVE && currentPoints.length >= 2) {
        // Bond Dihedral Torsion
        const p1 = currentPoints[0];
        const p2 = currentPoints[1];
        const currAngle = Math.atan2(p2[1] - p1[1], p2[0] - p1[0]);
        const deltaAngle = currAngle - this.startDihedralAngle;
        if (typeof this.options.onDihedralRotate === 'function' && this.activeDihedral) {
          this.options.onDihedralRotate(this.activeDihedral, deltaAngle);
        }
        this.startDihedralAngle = currAngle;
        this.isDirty = true;
      }

      this.requestRender();
    }

    _onTouchEnd(event) {
      event.preventDefault();
      event.stopPropagation();

      const touches = event.targetTouches || event.touches;
      const duration = Date.now() - this.startTouchTime;

      // Tap / Atom selection detection
      if (touches.length === 0) {
        if (this.state === TouchState.ORBIT && duration < 280 && this.touchPoints.length === 1) {
          const startP = this.startTouchPos;
          const endP = this.touchPoints[0];
          const distMoved = Math.sqrt(Math.pow(endP[0] - startP[0], 2) + Math.pow(endP[1] - startP[1], 2));
          if (distMoved < 8.0) {
            this._handleAtomPick(endP[0], endP[1]);
          }
        }
        this.state = TouchState.IDLE;
        this._notifyStateChange();
      } else if (touches.length === 1) {
        // Transition back to single finger orbit
        this.startTouchPos = this._getTouchPos(touches[0]);
        const [nx, ny] = this._toNormalizedCoordinates(this.startTouchPos[0], this.startTouchPos[1]);
        this.startSphereVector = Math3D.projectToSphere(nx, ny);
        this.state = TouchState.ORBIT;
        this._notifyStateChange();
      }

      this.requestRender();
    }

    _onTouchCancel(event) {
      event.preventDefault();
      this.state = TouchState.IDLE;
      this._notifyStateChange();
      this.requestRender();
    }

    _handleAtomPick(screenX, screenY) {
      this.state = TouchState.ATOM_SELECTION;
      this._notifyStateChange();

      // Ray casting / atom picking with DPR compensation
      const pickX = screenX * this.dpr;
      const pickY = screenY * this.dpr;

      if (typeof this.options.onAtomPick === 'function') {
        this.options.onAtomPick({ screenX, screenY, pickX, pickY, dpr: this.dpr });
      }

      setTimeout(() => {
        if (this.state === TouchState.ATOM_SELECTION) {
          this.state = TouchState.IDLE;
          this._notifyStateChange();
        }
      }, 50);
    }

    _onContextLost(event) {
      event.preventDefault();
      this.isContextLost = true;
      if (this.rafId) {
        cancelAnimationFrame(this.rafId);
        this.rafId = null;
      }
      console.warn('[CoChem 3D] WebGL context lost. Releasing resources.');
    }

    _onContextRestored(event) {
      this.isContextLost = false;
      console.info('[CoChem 3D] WebGL context restored. Reconstructing scene graph.');
      this._updateCanvasResolution();
      this.updateFrustum();
      if (this.options.viewer && typeof this.options.viewer.render === 'function') {
        this.options.viewer.render();
      }
      this.isDirty = true;
      this.requestRender();
    }

    _onResize() {
      this._updateCanvasResolution();
      this.updateFrustum();
      this.isDirty = true;
      this.requestRender();
    }

    _notifyStateChange() {
      if (typeof this.options.onStateChange === 'function') {
        this.options.onStateChange(this.state, this.getCameraState());
      }
    }

    /**
     * Compute dynamic near and far frustum clipping planes:
     * z_near = max(0.1, d - 1.5 R)
     * z_far  = d + 2.0 R
     */
    updateFrustum() {
      const d = this.focalDistance;
      const R = this.boundingRadius;
      this.nearClipping = Math.max(0.1, d - 1.5 * R);
      this.farClipping = d + 2.0 * R;
      return { near: this.nearClipping, far: this.farClipping };
    }

    /**
     * Return immutable camera state snapshot
     */
    getCameraState() {
      return {
        quaternion: [...this.quaternion],
        target: [...this.target],
        fov_degrees: this.fovDegrees,
        focal_distance: this.focalDistance,
        near_clipping: this.nearClipping,
        far_clipping: this.farClipping
      };
    }

    setCameraState(cameraState) {
      if (!cameraState) return;
      if (cameraState.quaternion) {
        this.quaternion = Math3D.quatNormalize([...cameraState.quaternion]);
      }
      if (cameraState.target) {
        this.target = [...cameraState.target];
      }
      if (cameraState.focal_distance) {
        this.focalDistance = Number(cameraState.focal_distance);
      }
      if (cameraState.fov_degrees) {
        this.fovDegrees = Number(cameraState.fov_degrees);
      }
      this.updateFrustum();
      this.isDirty = true;
      this.requestRender();
    }

    autoCenter(centroid = [0.0, 0.0, 0.0]) {
      this.target = [...centroid];
      this.isDirty = true;
      this.requestRender();
    }

    fitView(boundingRadius = null) {
      if (boundingRadius !== null) {
        this.boundingRadius = Math.max(0.1, Number(boundingRadius));
      }
      const fovRad = (this.fovDegrees * Math.PI) / 180.0;
      this.focalDistance = this.boundingRadius / Math.sin(fovRad * 0.5);
      this.updateFrustum();
      this.isDirty = true;
      this.requestRender();
    }

    resetCamera() {
      this.quaternion = [0.0, 0.0, 0.0, 1.0];
      this.target = [0.0, 0.0, 0.0];
      this.fitView();
    }

    setBoundingRadius(radius) {
      this.boundingRadius = Math.max(0.1, Number(radius));
      this.updateFrustum();
      this.isDirty = true;
      this.requestRender();
    }

    setDihedralMode(enabled, dihedralAtomIndices = null) {
      this.options.enableDihedralTorsion = Boolean(enabled);
      this.activeDihedral = dihedralAtomIndices;
    }

    requestRender() {
      if (this.isContextLost) return;
      if (!this.rafId) {
        this.rafId = requestAnimationFrame(() => {
          this.rafId = null;
          if (this.isDirty) {
            this._renderFrame();
            this.isDirty = false;
          }
        });
      }
    }

    _renderFrame() {
      if (this.options.viewer && typeof this.options.viewer.setCameraState === 'function') {
        this.options.viewer.setCameraState(this.getCameraState());
      }
    }

    destroy() {
      if (this.rafId) {
        cancelAnimationFrame(this.rafId);
        this.rafId = null;
      }
      this._detachListeners();
    }
  }

  // Export to global environment
  global.Math3D = Math3D;
  global.TouchState = TouchState;
  global.TouchController = TouchController;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Math3D, TouchState, TouchController };
  }
})(typeof window !== 'undefined' ? window : globalThis);
