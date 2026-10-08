import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { GLTFLoader } from './vendor/GLTFLoader.js';
import { ZOOM_DIST, SHOTS, a0Shot, a1Shot, END_FRAME, BEE_WAIT, BEE_FLIGHT, BEE_FLAP_PRE, BEE_FLAP_POST } from './js/timeline.js';
import { S as liveS, displayState } from './js/live.js';
import { ctx as plotCtx, plotsEl, plotItems, plotState, scrollState, slowScroll, updatePlots, drawPlots } from './js/plots.js';

const D = window.PBIT_DATA;
const { alpha, lam, bias, J, dt, stride } = D.params;
const traj = D.traj;
const N = traj.length;
const dtFrame = dtFrameOf(stride);

import { C_BALL, C_GOAL, C_MAGENTA, C_ARROW_OFF, C_ARROW_ON, SQ_SHADOW, SQ_LIT, LETTER_SHADOW, LETTER_LIT,
         C_S, C_O, C_A, PANELS, SPACING, COL_SPACING, RANGE, RES, HSCALE, EC, SHOW_WALLS, SIM_RATE, dtFrameOf } from './js/config.js';
const C_RED = 0xff2a2a;

// Energy V(x) as in pbits/model.py
function energy(x) {
  let e = 0;
  const n = x.length;
  for (let i = 0; i < n; i++) e += (alpha / 4) * x[i] ** 4 - (lam / 2) * x[i] ** 2 - bias[i] * x[i];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) e -= 0.5 * x[i] * J[i][j] * x[j];
  return e;
}
// Self-test against JAX
{
  let maxErr = 0;
  D.selftest.x.forEach((x, k) => { maxErr = Math.max(maxErr, Math.abs(energy(x) - D.selftest.V[k])); });
  console.log('Selbsttest max |V_js - V_jax| =', maxErr);
  document.getElementById('status').textContent = maxErr < 1e-4 ? '' : 'WARNUNG: Energie JS ≠ JAX';
  window.__selftestErr = maxErr;
}

// Energy of a panel as a function of its two bits (a, b); the remaining bits come from x
// Local energy of a panel: double wells + bias of its two bits + coupling to all other bits (without the constant of the remaining bits)
const dw = (v) => (alpha / 4) * v ** 4 - (lam / 2) * v ** 2;
function panelEnergy(p, a, b) {
  const [i, j] = p.bits;
  return dw(a) + dw(b) - bias[i] * a - bias[j] * b - J[i][j] * a * b - a * p.fa - b * p.fb;
}
// Field of the neighbouring bits on the two bits of a panel (once per frame)
function setFields(p, x) {
  const [i, j] = p.bits; p.fa = 0; p.fb = 0;
  for (const k of p.nbrs) { p.fa += J[i][k] * x[k]; p.fb += J[j][k] * x[k]; }
}

const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1116);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera.position.set(-8.3, 9.04, 17.59);   // frontal view of the wall, far enough back
const controls = new OrbitControls(camera, canvas);
controls.enabled = false;   // no mouse interaction (rotate, zoom, pan)
controls.target.set(-0.8, 0, 0);
controls.minAzimuthAngle = -0.7; controls.maxAzimuthAngle = 0.7;   // limit rotation so the wall stays readable
controls.minPolarAngle = 0.9; controls.maxPolarAngle = 2.0;
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dl = new THREE.DirectionalLight(0xffffff, 0.8); dl.position.set(3, 8, 5); scene.add(dl);

const rowsN = Math.max(...PANELS.map(p => p.row)) + 1;
const colMid = (Math.min(...PANELS.map(p => p.col)) + Math.max(...PANELS.map(p => p.col))) / 2;

function colorRamp(t) {
  t = Math.min(1, Math.max(0, t));
  const c = new THREE.Color();
  c.setHSL(0.72 - 0.6 * t, 0.75, 0.25 + 0.3 * t);
  return c;
}

const labelsDiv = document.getElementById('labels');
for (const p of PANELS) {
  const g = new THREE.Group();
  // time (col) runs to the right, row 0 (actions/s) on top, last row (o) at the bottom
  g.position.set((p.col - colMid) * COL_SPACING, -(p.row - (rowsN - 1) / 2) * SPACING, 0);
  scene.add(g);
  p.group = g;

  const geo = new THREE.PlaneGeometry(2 * RANGE, 2 * RANGE, RES - 1, RES - 1);
  geo.rotateX(-Math.PI / 2);
  geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(RES * RES * 3), 3));
  p.geo = geo;
  g.add(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 0.6 })));
  g.add(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.06 })));

  p.ball = new THREE.Mesh(new THREE.SphereGeometry(0.13, 24, 16), new THREE.MeshStandardMaterial({ color: C_MAGENTA, emissive: C_MAGENTA, emissiveIntensity: 0.6 }));
  g.add(p.ball);

  p.trailN = 120;
  p.trailGeo = new THREE.BufferGeometry();
  p.trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(p.trailN * 3), 3));
  p.trail = new THREE.Line(p.trailGeo, new THREE.LineBasicMaterial({ color: C_MAGENTA }));
  p.trail.frustumCulled = false;
  g.add(p.trail);

  const el = document.createElement('div'); el.innerHTML = p.name.replace(/[₀-₉]/g, c => '<span style="font-size:0.78em;position:relative;top:0.28em;margin-left:1px;font-variant-numeric:lining-nums">' + '₀₁₂₃₄₅₆₇₈₉'.indexOf(c) + '</span>'); el.style.color = '#cbd5e1';
  {   // note below the label (centered), appears only from the clamp time on
    const cv = (D.params.clamped || {})[p.bits[0]], cw = (D.params.clamped || {})[p.bits[1]];
    if (p.name.startsWith('OBSERVATION') && cv !== undefined && cw !== undefined && (D.params.clamp_starts || {})[p.bits[0]] !== undefined) {
      const f = v => (v > 0 ? '+' : '-') + '1';
      const note = document.createElement('div');
      const isGoal = p.name === 'OBSERVATION o₂', isBlue = isGoal || p.name === 'OBSERVATION o₀' || p.name === 'OBSERVATION o₁';
      note.textContent = isGoal ? 'Goal: C' : p.name === 'OBSERVATION o₀' ? 'Observed: B' : p.name === 'OBSERVATION o₁' ? 'Observed: A' : 'Observed: (' + f(cv) + ', ' + f(cw) + ')';
      note.style.cssText = 'position:static;color:' + (isBlue ? '#ff8c1a' : '#ff2a2a') + ';' + (isBlue ? 'font-size:1.6em;font-weight:700;' : '') + 'text-align:center;margin-top:2px;visibility:hidden';
      el.appendChild(note); p.note = note;
    }
  }
  labelsDiv.appendChild(el); p.label = el;
  p.swap = p.name.toLowerCase().startsWith('action');   // actions: axis assignment swapped (front = bit 0, depth = bit 1)
  if (p.swap) g.rotation.y = -3 * Math.PI / 4;   // actions: rotated 135° clockwise (seen from above) around the vertical axis
  p.nbrs = [];
  for (let k = 0; k < J.length; k++) if (k !== p.bits[0] && k !== p.bits[1] && (J[p.bits[0]][k] !== 0 || J[p.bits[1]][k] !== 0)) p.nbrs.push(k);
}

const hOf = e => HSCALE * EC * Math.log1p(Math.max(e, -0.99 * EC) / EC);   // energy -> height

// ---- grid (floor + two walls facing away from the camera) and axis labels ----
const E_MIN = -0.8, E_MAX = 7;       // energy range of the wall height (display only)
const Y0 = hOf(E_MIN), Y1 = hOf(E_MAX);
const GRID_STEP = 0.5;
const FLOOR = 1.7;
const THICK = 0.12;   // thickness of the chessboard squares and the arrows (display only)

function ticksOf(lo, hi, step) {
  const out = [lo];
  for (let v = Math.ceil((lo + 1e-9) / step) * step; v < hi - 1e-9; v += step) out.push(+v.toFixed(6));
  out.push(hi);
  return out;
}
const isMajor = u => Math.abs(u) < 1e-6 || Math.abs(Math.abs(u) - 1) < 1e-6;

// map(u, v) -> [x, y, z]; lines in u direction (at positions vs) and v direction (at positions us)
function makeGrid(map, us, vs, majorU, majorV) {
  const minor = [], major = [];
  const push = (arr, a, b) => arr.push(...a, ...b);
  const u0 = us[0], u1 = us[us.length - 1], v0 = vs[0], v1 = vs[vs.length - 1];
  for (const u of us) push(majorU(u) ? major : minor, map(u, v0), map(u, v1));
  for (const v of vs) push(majorV(v) ? major : minor, map(u0, v), map(u1, v));
  const g = new THREE.Group();
  for (const [arr, op] of [[minor, 0.12], [major, 0.3]]) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(arr, 3));
    g.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0xaab4c8, transparent: true, opacity: op })));
  }
  return g;
}

function textSprite(text, color = '#9aa5b8', size = 0.32) {
  const c = document.createElement('canvas'); c.width = 128; c.height = 64;
  const ctx = c.getContext('2d');
  ctx.fillStyle = color; ctx.font = '600 38px system-ui, sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(text, 64, 32);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
  s.scale.set(size * 2, size, 1);
  return s;
}
const sub = i => 'x' + String(i).split('').map(c => '₀₁₂₃₄₅₆₇₈₉'[+c]).join('');

const ticksH = ticksOf(-RANGE, RANGE, GRID_STEP);
const ticksY = ticksOf(Y0, Y1, GRID_STEP);
const no = () => false;
for (const p of PANELS) {
  // s/o: floor (±RANGE) as a chessboard (four squares = four quadrants/wells); actions unchanged (±RANGE)
  const R = FLOOR;   // FLOOR: floor size for s/o, slightly larger than the surface
  const ticksF = ticksOf(-R, R, GRID_STEP);
  p.floor = makeGrid((u, v) => [u, Y0, v], ticksF, ticksF, isMajor, isMajor);
  if (!p.swap) {
    p.squares = [];
    for (const [cx, cz] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
      const white = (cx * cz) > 0;
      const sq = new THREE.Mesh(new THREE.BoxGeometry(R, THICK, R),
        new THREE.MeshBasicMaterial({ color: white ? 0xe6e9ef : 0x3a4150 }));
      sq.position.set(cx * R / 2, Y0 - 0.01 - THICK / 2, cz * R / 2);   // top edge just below the grid lines
      sq.userData = { cx, cz, white, shadow: new THREE.Color(SQ_SHADOW[white ? 0 : 1]), lit: new THREE.Color(SQ_LIT[white ? 0 : 1]) };
      // letter A–D lying flat on the square (white on dark, black on white squares)
      const letter = 'ABCD'[(cz > 0 ? 2 : 0) + (cx > 0 ? 1 : 0)];
      const lc = document.createElement('canvas'); lc.width = lc.height = 256;
      const lx = lc.getContext('2d');
      lx.fillStyle = white ? '#000000' : '#ffffff';
      lx.font = 'bold 256px sans-serif'; lx.textAlign = 'center'; lx.textBaseline = 'middle';
      lx.fillText(letter, 128, 146);
      const lm = new THREE.Mesh(new THREE.PlaneGeometry(R * 0.95, R * 0.95),
        new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(lc), transparent: true, depthWrite: false }));
      lm.rotation.x = -Math.PI / 2;
      lm.position.y = THICK / 2 + 0.004;
      sq.add(lm);
      sq.userData.letter = lm;
      p.squares.push(sq);
      p.group.add(sq);
    }
  }
  // walls at x = ±R (axes: z, y) and z = ±R (axes: x, y)
  p.floor.visible = false;   // floor grid hidden
  p.group.add(p.floor);
  if (SHOW_WALLS) {
    p.wallX = [-R, R].map(X => makeGrid((u, v) => [X, v, u], ticksH, ticksY, isMajor, no));
    p.wallZ = [-R, R].map(Z => makeGrid((u, v) => [u, v, Z], ticksH, ticksY, isMajor, no));
    for (const w of [...p.wallX, ...p.wallZ]) p.group.add(w);
  }

  const sg = p.swap ? -1 : 1;   // actions (rotated by 135°): labels on the opposite sides so they are readable from the front
  // ticks (−1, 0, 1) and axis names at the floor edge of the front side (initial view)
  for (const t of []) {   // tick numbers hidden
    const a = textSprite(String(t)); a.position.set(t, Y0, sg * (R + 0.25)); p.group.add(a);
    const b = textSprite(String(-t)); b.position.set(-sg * (R + 0.25), Y0, t); p.group.add(b);   // depth axis (bit 0) inverted: back = +1
  }
  if (false) {   // axis names hidden
    const ax = textSprite(sub(p.bits[p.swap ? 0 : 1]), '#d8dde6', 0.4); ax.position.set(0, Y0, sg * (R + 0.6)); p.group.add(ax);
    const az = textSprite(sub(p.bits[p.swap ? 1 : 0]), '#d8dde6', 0.4); az.position.set(-sg * (R + 0.6), Y0, 0); p.group.add(az);
  }
  if (p.swap) {
    // direction arrows at the four quadrants (bit0/bit1): up −1/−1, right −1/+1, down +1/+1, left +1/−1
    // local position: x = bit 0, z = −bit 1
    const dirs = [['↑', -1, -1], ['→', -1, 1], ['↓', 1, 1], ['←', 1, -1]];
    // arrows lie flat on the floor in world orientation (counter-rotation to the 135° group rotation):
    // ↑ points to the back (away from the camera), ↓ to the front, → to the right, ← to the left
    p.arrows = [];
    const ag = new THREE.Group();
    ag.rotation.y = 3 * Math.PI / 4;
    p.group.add(ag);
    for (const [ch, b0, b1] of dirs) {
      // extruded arrow (thickness = THICK like the chessboard squares), points to +Y of the shape in its plane
      const sh = new THREE.Shape();
      sh.moveTo(0, 0.5); sh.lineTo(0.34, 0.05); sh.lineTo(0.12, 0.05); sh.lineTo(0.12, -0.45);
      sh.lineTo(-0.12, -0.45); sh.lineTo(-0.12, 0.05); sh.lineTo(-0.34, 0.05); sh.closePath();
      const geo = new THREE.ExtrudeGeometry(sh, { depth: THICK, bevelEnabled: false });
      geo.scale(1.6, 1.6, 1);   // larger arrow, same thickness
      geo.rotateZ({ '↑': 0, '→': -Math.PI / 2, '↓': Math.PI, '←': Math.PI / 2 }[ch]);
      geo.rotateX(-Math.PI / 2);   // lay flat: shape direction +Y -> to the back (-z), thickness upwards
      const d = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ color: C_ARROW_OFF }));
      d.position.set(b0 * R * 0.7, Y0, -b1 * R * 0.7)
        .applyAxisAngle(new THREE.Vector3(0, 1, 0), -3 * Math.PI / 4);
      d.userData = { b0, b1 };
      p.arrows.push(d);
      ag.add(d);
    }
  }
}

// Per frame: show only the two walls facing away from the camera
function updateWalls(p) {
  if (!SHOW_WALLS) return;
  const cx = camera.position.x - p.group.position.x;
  const cz = camera.position.z - p.group.position.z;
  p.wallX[0].visible = cx > 0;  p.wallX[1].visible = cx <= 0;
  p.wallZ[0].visible = cz > 0;  p.wallZ[1].visible = cz <= 0;
}

function updateSurface(p, x) {
  const pos = p.geo.attributes.position, col = p.geo.attributes.color;
  setFields(p, x);
  const E = p.Ebuf || (p.Ebuf = new Float32Array(RES * RES));
  let emin = Infinity;
  for (let k = 0; k < RES * RES; k++) { E[k] = (p.swap ? panelEnergy(p, pos.getX(k), -pos.getZ(k)) : panelEnergy(p, -pos.getZ(k), pos.getX(k))); if (E[k] < emin) emin = E[k]; }
  p.emin = emin;
  for (let k = 0; k < RES * RES; k++) {
    const e = E[k] - emin;
    pos.setY(k, hOf(e));
    const c = colorRamp(1 - e / 9);
    col.setXYZ(k, c.r, c.g, c.b);
  }
  pos.needsUpdate = true; col.needsUpdate = true;
  p.geo.computeVertexNormals();
}
PANELS.forEach(p => p.group.children.forEach(c => { c.frustumCulled = false; }));

// Edges from J: line between two panels if J != 0 between their bits (thickness ~ |J|, dark = negative J)
{
  const owner = {};
  PANELS.forEach(p => p.bits.forEach(b => { owner[b] = p; }));
  const edges = new Map();
  for (let i = 0; i < J.length; i++) for (let j = i + 1; j < J.length; j++) {
    const w = J[i][j];
    if (w === 0 || !owner[i] || !owner[j] || owner[i] === owner[j]) continue;   // auxiliary p-bits (without a panel) are not drawn
    const key = PANELS.indexOf(owner[i]) + '-' + PANELS.indexOf(owner[j]);
    if (!edges.has(key)) edges.set(key, { a: owner[i], b: owner[j], list: [] });
    edges.get(key).list.push({ w, ia: owner[i].bits.indexOf(i), ib: owner[j].bits.indexOf(j) });   // one edge per J entry
  }
  const wMax = Math.max(1e-9, ...[...edges.values()].flatMap(e => e.list.map(l => Math.abs(l.w))));
  const yMid = (Y0 + Y1) / 2, hx = RANGE + 0.15, hy = (Y1 - Y0) / 2 + 0.1;
  const edgeMat = w => new THREE.MeshBasicMaterial({ color: w < 0 ? 0x6b7280 : 0xd8dde6, transparent: true, opacity: 0.7 });
  const addSeg = (p, q, r, w) => {   // cylinder between p and q (+ sphere at the end as a joint without a gap)
    const dir = q.clone().sub(p), len = dir.length();
    if (len < 1e-4) return;
    const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 12), edgeMat(w));
    m.position.copy(p).add(q).multiplyScalar(0.5);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
    scene.add(m);
    const s = new THREE.Mesh(new THREE.SphereGeometry(r, 12, 8), edgeMat(w));
    s.position.copy(q); scene.add(s);
  };
  for (const { a, b, list } of edges.values()) for (const { w, ia, ib } of list) {
    const act = a.name.toLowerCase().startsWith('action') ? a : (b.name.toLowerCase().startsWith('action') ? b : null);
    const r = 0.016 + 0.03 * (Math.abs(w) / wMax);
    if (act) {
      // inverted T; second state bit = inner angled edge in the crook
      const st = act === a ? b : a;
      const ab = act === a ? ia : ib, sb = act === a ? ib : ia;
      const side = st.group.position.x < act.group.position.x ? -1 : 1;
      const d2 = 2 * r;   // center distance = diameter: wires lie directly next to each other
      const z = st.group.position.z + (ab - 0.5) * d2;
      const yH = st.group.position.y + yMid + d2 + sb * d2;
      const xm = act.group.position.x + side * (d2 / 2 + sb * d2);
      const p0 = new THREE.Vector3(st.group.position.x - side * hx, yH, z);
      const p1 = new THREE.Vector3(xm, yH, z);
      const p2 = new THREE.Vector3(xm, act.group.position.y + Y0, z);
      addSeg(p0, p1, r, w);
      addSeg(p1, p2, r, w);
      continue;
    }
    const oz = new THREE.Vector3(0, 0, (ia - 0.5) * 2 * r);
    const ca = a.group.position.clone().add(new THREE.Vector3(0, yMid, 0)).add(oz);
    const cb = b.group.position.clone().add(new THREE.Vector3(0, yMid, 0)).add(oz);
    const d = cb.clone().sub(ca);
    const t = Math.min(hx / Math.max(1e-9, Math.abs(d.x)), hy / Math.max(1e-9, Math.abs(d.y)));
    const pa = ca.clone().addScaledVector(d, t), pb = cb.clone().addScaledVector(d, -t);
    const dir = pb.clone().sub(pa), len = dir.length();
    if (len < 0.05) continue;
    const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 12), edgeMat(w));
    m.position.copy(pa).add(pb).multiplyScalar(0.5);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
    scene.add(m);
  }
}

function stateAt(f) {
  const i = Math.min(N - 2, Math.max(0, Math.floor(f)));
  const w = f - i;
  return traj[i].map((v, k) => v * (1 - w) + traj[i + 1][k] * w);
}

const trails = PANELS.map(() => []);
let frame = 0, playing = true, speed = 0.1, lastT = performance.now();
liveS.frame = () => frame; liveS.playing = () => playing; liveS.speed = () => speed; liveS.stateAt = stateAt;
const tSlider = document.getElementById('t'), playBtn = document.getElementById('play');
const ICON_PLAY = '<svg width="28" height="28" viewBox="0 0 20 20"><path d="M5 2.5 L17 10 L5 17.5 Z" fill="currentColor"/></svg>';
const ICON_PAUSE = '<svg width="28" height="28" viewBox="0 0 20 20"><rect x="4" y="2.5" width="4.2" height="15" fill="currentColor"/><rect x="11.8" y="2.5" width="4.2" height="15" fill="currentColor"/></svg>';
playBtn.innerHTML = playing ? ICON_PAUSE : ICON_PLAY;
playBtn.onclick = () => { playing = !playing; playBtn.innerHTML = playing ? ICON_PAUSE : ICON_PLAY; };
document.getElementById('plotsToggle').onclick = () => {
  const open = document.body.classList.toggle('plots-open');
  const it = open && plotState.lastShot >= 0 && plotItems[plotState.lastShot];
  if (it) { scrollState.id++; plotsEl.scrollTop = it.box.offsetTop; }   // jump to the active plot immediately when opening
};   // slide the 2D plot window in/out
tSlider.onclick = (e) => {   // beads sit on top of the thumb: snap to a bead when the click lands on it
  const hit = BEADS.find(q => Math.abs(q.d.getBoundingClientRect().left + 12 - e.clientX) < 18);
  if (hit) { frame = hit.s; tSlider.value = frame / END_FRAME; trails.forEach(q => { q.length = 0; }); }
};
tSlider.oninput = () => { frame = parseFloat(tSlider.value) * END_FRAME; trails.forEach(t => { t.length = 0; }); };

const canvas2 = document.getElementById('c2');
const renderer2 = new THREE.WebGLRenderer({ canvas: canvas2, antialias: true });
const scene2 = new THREE.Scene();
scene2.background = new THREE.Color(0x1a212a);
const camera2 = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
camera2.position.set(-2.17, 5.66, 10.87);
const controls2 = new OrbitControls(camera2, canvas2);
controls2.enabled = false;   // no mouse interaction (rotate, zoom, pan)
controls2.target.set(0, 1.1, 0);
controls2.minDistance = 8; controls2.maxDistance = 20;
let wings = null;
const WING_AMP = 0.45, WING_HZ = 4;   // wing beat: amplitude (rad), frequency (Hz)
let beeObj = null, beeBase = null;   // bee and its start pose (square B), for the flight to square A
const BEE_HEIGHT = 0.9;   // arc height of the flight
const BEE_TURN = 0.35;   // fraction of the flight (from the start) in which the bee turns to the left
let hcTopY = null;   // top edge of the honeycomb on square C
const srgbTex = (cv) => { const t = new THREE.CanvasTexture(cv); t.colorSpace = THREE.SRGBColorSpace; return t; };
const glows = [];   // glow of the letters on the board with the bee: { mesh, panelName }
const GLOW_HOLD = 30, GLOW_SAMPLES = 20;   // 3 s constant, then 2 s fade-out (at speed 0.1x)
const BIG = 1.7;   // edge length of one square of the large board
for (const [cx, cz, white] of [[-1, -1, true], [1, -1, false], [-1, 1, false], [1, 1, true]]) {
  const sq = new THREE.Mesh(new THREE.BoxGeometry(BIG, THICK * 2, BIG),
    new THREE.MeshBasicMaterial({ color: white ? 0xe6e9ef : 0x050506 }));
  sq.position.set(cx * BIG / 2, 0, cz * BIG / 2);
  scene2.add(sq);
  // letter A–D, all in the same gray
  const lc = document.createElement('canvas'); lc.width = lc.height = 256;
  const lx = lc.getContext('2d');
  lx.fillStyle = white ? '#000000' : '#ffffff';
  lx.font = 'bold 256px sans-serif'; lx.textAlign = 'center'; lx.textBaseline = 'middle';
  lx.fillText('ABCD'[(cz > 0 ? 2 : 0) + (cx > 0 ? 1 : 0)], 128, 146);
  const lm = new THREE.Mesh(new THREE.PlaneGeometry(BIG * 0.95, BIG * 0.95),
    new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(lc), transparent: true, depthWrite: false }));
  lm.rotation.x = -Math.PI / 2;
  lm.position.set(cx * BIG / 2, THICK + 0.004, cz * BIG / 2);
  scene2.add(lm);
  if (cz === -1) {   // square B (o₀) and square A (o₁): orange glow over the gray letter
    const gLetter = cx === 1 ? 'B' : 'A';
    const gc = document.createElement('canvas'); gc.width = gc.height = 256;
    const gx = gc.getContext('2d');
    gx.font = 'bold 256px sans-serif'; gx.textAlign = 'center'; gx.textBaseline = 'middle';
    gx.fillStyle = '#ff8c1a';   // sharp and opaque orange, no halo
    gx.fillText(gLetter, 128, 146);
    const glowB = new THREE.Mesh(new THREE.PlaneGeometry(BIG * 0.95, BIG * 0.95),
      new THREE.MeshBasicMaterial({ map: srgbTex(gc), transparent: true, opacity: 0, depthWrite: false }));
    glowB.rotation.x = -Math.PI / 2;
    glowB.position.set(cx * BIG / 2, THICK + 0.008, cz * BIG / 2);
    scene2.add(glowB);
    glows.push({ mesh: glowB, panelName: cx === 1 ? 'OBSERVATION o₀' : 'OBSERVATION o₁' });
    if (cx === -1) {   // square A (o₁): not the letter, but the white background glows orange
      glowB.visible = false;
      const fieldGlow = new THREE.Mesh(new THREE.PlaneGeometry(BIG, BIG),
        new THREE.MeshBasicMaterial({ color: 0xff8c1a, transparent: true, opacity: 0, depthWrite: false }));
      fieldGlow.rotation.x = -Math.PI / 2;
      fieldGlow.position.set(cx * BIG / 2, THICK + 0.002, cz * BIG / 2);   // above the square, below the black letter
      scene2.add(fieldGlow);
      glows[glows.length - 1].mesh = fieldGlow;
    }
  }
}

{   // square C (o2, goal): orange glow of the letter on clamping
  const cx = -1, cz = 1;
  const gc = document.createElement('canvas'); gc.width = gc.height = 256;
  const gx = gc.getContext('2d');
  gx.font = 'bold 256px sans-serif'; gx.textAlign = 'center'; gx.textBaseline = 'middle';
  gx.fillStyle = '#ff8c1a';   // sharp and opaque orange, no halo
  gx.fillText('C', 128, 146);
  const glowC = new THREE.Mesh(new THREE.PlaneGeometry(BIG * 0.95, BIG * 0.95),
    new THREE.MeshBasicMaterial({ map: srgbTex(gc), transparent: true, opacity: 0, depthWrite: false }));
  glowC.rotation.x = -Math.PI / 2;
  glowC.position.set(cx * BIG / 2, THICK + 0.008, cz * BIG / 2);
  scene2.add(glowC);
  glows.push({ mesh: glowC, panelName: 'OBSERVATION o₂' });
}
scene2.add(new THREE.AmbientLight(0xffffff, 1.6));
const dl2 = new THREE.DirectionalLight(0xffffff, 1.5); dl2.position.set(3, 6, 5); scene2.add(dl2);
new GLTFLoader().load('./assets/bee.glb', (g) => {
  const bee = g.scene;
  const box = new THREE.Box3().setFromObject(bee);
  const size = box.getSize(new THREE.Vector3());
  const k = 0.6 * BIG / Math.max(size.x, size.z);
  bee.scale.setScalar(k);
  const b2 = new THREE.Box3().setFromObject(bee);
  const c = b2.getCenter(new THREE.Vector3());
  bee.position.set(BIG / 2 - c.x, THICK - b2.min.y, -BIG / 2 - c.z);
  bee.rotation.y = -Math.PI / 4;   // facing front left
  const wl = bee.getObjectByName('wing-left'), wr = bee.getObjectByName('wing-right');
  if (wl && wr) wings = { wl, wr, zl: wl.rotation.z, zr: wr.rotation.z };
  scene2.add(bee);
  beeObj = bee; beeBase = { pos: bee.position.clone(), rotY: bee.rotation.y };
}, undefined, (e) => console.error('bee.glb', e));
new GLTFLoader().load('./assets/honeycomb.glb', (g) => {
  const hc = g.scene;
  hc.rotation.y = -Math.PI / 9;   // 20 degrees clockwise (from above)
  const box = new THREE.Box3().setFromObject(hc);
  const size = box.getSize(new THREE.Vector3());
  hc.scale.setScalar(1.0 * BIG / Math.max(size.x, size.z));
  const b2 = new THREE.Box3().setFromObject(hc);
  const c = b2.getCenter(new THREE.Vector3());
  hc.position.set(-BIG / 2 - c.x, THICK - b2.min.y, BIG / 2 - c.z);
  scene2.add(hc);
  hcTopY = new THREE.Box3().setFromObject(hc).max.y;   // top edge of the honeycomb (landing height of the bee)
}, undefined, (e) => console.error('honeycomb.glb', e));

// honeycomb on square C in the o₂ plot (goal): appears as soon as o₂ is clamped
let hcGoal = null;
const HC_GLOW = 0.9;   // self-illumination of the goal honeycomb (0 = normally lit, 1 = fully bright)
new GLTFLoader().load('./assets/honeycomb.glb', (g) => {
  const po = PANELS.find(q => q.name === 'OBSERVATION o₂');
  const hc = g.scene;
  hc.rotation.y = -Math.PI / 9;
  const bx = new THREE.Box3().setFromObject(hc), sz = bx.getSize(new THREE.Vector3());
  hc.scale.setScalar(1.0 * FLOOR / Math.max(sz.x, sz.z));
  const b2 = new THREE.Box3().setFromObject(hc), c2 = b2.getCenter(new THREE.Vector3());
  hc.position.set(-FLOOR / 2 - c2.x, (Y0 - 0.01) - b2.min.y, FLOOR / 2 - c2.z);
  hc.traverse(o => {   // not in shadow: self-illumination in the texture/base color, slight shading is kept
    if (!o.isMesh) return;
    (Array.isArray(o.material) ? o.material : [o.material]).forEach(m => {
      if (m.emissive) { m.emissive.copy(m.color); if (m.map) m.emissiveMap = m.map; m.emissiveIntensity = HC_GLOW; m.needsUpdate = true; }
    });
  });
  hc.visible = false;
  po.group.add(hc);
  hcGoal = { obj: hc, panel: po };
}, undefined, (e) => console.error('honeycomb.glb (o2)', e));
let vw = 1, vh = 1, vx = 0;
function resize() {
  const r = canvas.getBoundingClientRect(), r2 = canvas2.getBoundingClientRect();
  vw = r.width; vh = r.height; vx = r.left;
  const pr = Math.min(2, window.devicePixelRatio);
  renderer.setPixelRatio(pr); renderer.setSize(r.width, r.height, false);
  {
    const Hm = Math.max(1, window.innerHeight - (window.innerWidth <= 700 || window.innerHeight <= 520 ? 60 : 76));
    const opR = Math.min(1, Math.max(0, 1 - r.height / Hm) * 2);   // 0 = window closed, 1 = open
    const small = window.innerWidth <= 700 || window.innerHeight <= 520;
    camera.fov = 45 * (1 + (small ? 0.2 : 0.05) * opR);   // small screens: zoom out so POMDP and labels are not cropped
    camera.aspect = r.width / r.height;
    camera.setViewOffset(r.width, r.height, 0, (small ? 0.05 + 0.04 * opR : 0.06) * r.height, r.width, r.height);   // shift upwards
    camera.updateProjectionMatrix();
  }
  renderer2.setPixelRatio(pr); renderer2.setSize(r2.width, r2.height, false);
  // bee board: pixel scale stays as with the closed window, the board only moves upwards
  const H0 = Math.max(1, window.innerHeight - (window.innerWidth <= 700 || window.innerHeight <= 520 ? 60 : 76));
  const op = Math.min(1, Math.max(0, 1 - r2.height / H0) * 2);   // 0 = closed, 1 = fully open
  const small2 = window.innerWidth <= 700 || window.innerHeight <= 520;
  const k = (1 + 0.2 * op) * (small2 ? 0.75 : 1);   // small screens: bee board larger
  camera2.fov = 2 * Math.atan(Math.tan(20 * Math.PI / 180) * k * r2.height / H0) * 180 / Math.PI;
  camera2.aspect = r2.width / r2.height;
  camera2.setViewOffset(r2.width, r2.height, (small2 ? 0.04 : 0) * r2.width, (0.12 * op + (small2 ? 0.06 : 0)) * r2.height, r2.width, r2.height);   // shift the image upwards
  camera2.updateProjectionMatrix();
}
window.addEventListener('resize', resize); resize();

const tmp = new THREE.Vector3();
// Camera flight to OBSERVATION o₂ (like a film, without mouse interaction)
// Bound to the playback position (frame). Values in samples relative to the clamp time CS of o₂
// (10 samples = 1 s real time at speed 0.1x)
// Sequence: o₂ (fly in 50 samples, wait 10, clamp, wait 40, fly out 50), then o₀ the same way
plotCtx.END_FRAME = END_FRAME; plotCtx.getFrame = () => frame; drawPlots();
// Stations below the bar: Setting Goal (o₂), Observation t=0 (o₀), Action t=0 (a₀), Observation t=1 (o₁), Action t=1 (a₁)
const SEG_LABELS = ['Setting Goal', 'Observation at t=0', 'Action at t=0', 'Observation at t=1', 'Action at t=1'];
const BEADS = [];   // clickable beads on the bar, each at the start of the station's fly-in
if (SHOTS.length === 5) {
  const segDiv = document.getElementById('segs');
  SHOTS.forEach((sh, i) => {
    const d = document.createElement('div'); d.className = 'bead';
    d.style.left = (sh.inS / END_FRAME * 100) + '%';
    const t = document.createElement('span'); t.textContent = SEG_LABELS[i]; d.appendChild(t);
    segDiv.appendChild(d); BEADS.push({ s: sh.inS, d });
  });
}
function updateSegs() {   // active bead = last station whose start has been reached
  let act = -1; BEADS.forEach((q, i) => { if (frame >= q.s) act = i; });
  BEADS.forEach((q, i) => q.d.classList.toggle('on', i === act));
}
let zoomBase = null;
function zoomStep() {
  if (!SHOTS.length) return;
  if (frame < SHOTS[0].inS) {   // before the flight (also after rewinding): initial view
    if (zoomBase) { camera.position.copy(zoomBase.pos); controls.target.copy(zoomBase.tgt); zoomBase = null; }
    controls.enabled = false;   // no mouse interaction (also before the flight)
    return;
  }
  if (!zoomBase) zoomBase = { pos: camera.position.clone(), tgt: controls.target.clone() };
  controls.enabled = false;   // film: no mouse interaction during and after the flight
  let e = 0, sh = null;
  for (const q of SHOTS) {
    if (frame < q.inS) break;
    sh = q;
    const u = frame < q.outS ? Math.min(1, (frame - q.inS) / (q.inE - q.inS))
                             : 1 - Math.min(1, (frame - q.outS) / (q.outE - q.outS));
    e = u * u * (3 - 2 * u);   // smoothstep
    if (frame < q.outE) break;
  }
  if (!sh || e === 0) { camera.position.copy(zoomBase.pos); controls.target.copy(zoomBase.tgt); return; }
  const tgt = sh.panel.group.position.clone().add(new THREE.Vector3(0, -0.4, 0));
  const dir = zoomBase.pos.clone().sub(zoomBase.tgt).normalize();
  camera.position.lerpVectors(zoomBase.pos, tgt.clone().addScaledVector(dir, ZOOM_DIST), e);
  controls.target.lerpVectors(zoomBase.tgt, tgt, e);
}
function loop(now) {
  if (Math.abs(canvas.getBoundingClientRect().height - vh) > 0.5) resize();   // window height changes when sliding out/in (also during the animation)
  zoomStep(); updateSegs();
  controls.update(); camera.updateMatrixWorld();   // camera first, so labels and canvas stay in sync
  const real = Math.min(0.1, (now - lastT) / 1000); lastT = now;
  if (playing) {
    frame += speed * SIM_RATE * real / dtFrame;
    if (frame >= END_FRAME) { frame = 0; trails.forEach(t => { t.length = 0; }); }
    tSlider.value = frame / END_FRAME;
  }
  {   // plot window: scroll once to the matching plot at the start of each station (up to a₁), freely scrollable in between
    let si = -1;
    SHOTS.forEach((sh, i) => { if (frame >= sh.inS) si = i; });
    if (si !== plotState.lastShot) {
      plotState.lastShot = si;
      plotItems.forEach((q, i) => q.box.classList.toggle('active', i === si));
      const it = si >= 0 && plotItems[si];
      if (it && plotsEl.clientHeight > 0) slowScroll(it.box.offsetTop);
    }
  }
  updatePlots();
  const x = displayState(real);
  PANELS.forEach((p, k) => {
    tmp.copy(p.group.position).add(new THREE.Vector3(0, Y0, (p.swap ? FLOOR * Math.SQRT2 : FLOOR) + 1.15)).project(camera);
    p.label.style.transform = `translate3d(${vx + (tmp.x + 1) / 2 * vw}px, ${(1 - tmp.y) / 2 * vh}px, 0) translateX(-50%)`;
    updateSurface(p, x);
    updateWalls(p);
    const a = x[p.bits[0]], b = x[p.bits[1]];
    const yb = hOf(panelEnergy(p, a, b) - p.emin) + 0.13;
    const px = p.swap ? a : b, pz = p.swap ? -b : -a;
    const st = (D.params.clamp_starts || {})[p.bits[0]];
    const red = st !== undefined && frame >= st;   // observation from the clamp time on
    const cc = red ? (p.name === 'OBSERVATION o₂' ? C_GOAL : (p.name === 'OBSERVATION o₀' || p.name === 'OBSERVATION o₁') ? C_BALL : C_RED) : C_MAGENTA;
    p.ball.material.color.setHex(cc); p.ball.material.emissive.setHex(cc);
    p.trail.material.color.setHex(cc);   // trail has the color of the ball
    if (p.note) p.note.style.visibility = red ? 'visible' : 'hidden';
    if (p.arrows) for (const d of p.arrows) {   // arrow in the ball's quadrant lights up
      const on = Math.sign(a) === d.userData.b0 && Math.sign(b) === d.userData.b1;
      d.material.color.setHex(on ? C_ARROW_ON : C_ARROW_OFF);
      d.material.emissive.setHex(on ? 0xd8d8d8 : 0x000000);   // self-illumination compensates for the weak scene lighting so the active arrow looks white
    }
    if (p.squares) for (const sq of p.squares) {   // square under the ball bright, the other three in shadow
      const on = Math.sign(b) === sq.userData.cx && -Math.sign(a) === sq.userData.cz;
      sq.material.color.copy(on ? sq.userData.lit : sq.userData.shadow);
      sq.userData.letter.material.color.setHex(on ? LETTER_LIT : LETTER_SHADOW);
      if (on && red && p.name.startsWith('OBSERVATION')) {   // clamped o: the white element (background of the white square or letter on a dark square) turns orange
        if (sq.userData.white) sq.material.color.setHex(C_BALL); else sq.userData.letter.material.color.setHex(C_BALL);
      }
    }
    p.ball.position.set(px, yb, pz);   // x axis (front) = bit 1, depth (inverted) = bit 0
    const T = trails[k];
    T.push([px, yb, pz]);
    if (T.length > p.trailN) T.shift();
    const arr = p.trailGeo.attributes.position.array;
    T.forEach((q, i) => arr.set(q, i * 3));
    p.trailGeo.attributes.position.needsUpdate = true;
    p.trailGeo.setDrawRange(0, T.length);
  });
  renderer.render(scene, camera);
  if (wings) {   // flapping only from 1 s before take-off to 1 s after landing (bound to frame), otherwise still
    let on = 0;
    for (const sh of [a0Shot, a1Shot]) {
      if (!sh) continue;
      const t0 = sh.inE + BEE_WAIT, a = t0 - BEE_FLAP_PRE, b = t0 + BEE_FLIGHT + BEE_FLAP_POST;
      on = Math.max(on, Math.min(1, (frame - a) / 3, (b - frame) / 3));   // short fade-in/out of the amplitude
    }
    on = Math.max(0, on);
    const w = on * WING_AMP * Math.sin(2 * Math.PI * WING_HZ * performance.now() / 1000);
    wings.wl.rotation.z = wings.zl + w;
    wings.wr.rotation.z = wings.zr - w;
  }
  if (beeObj && beeBase && a0Shot) {   // flight from square B to square A, bound to the playback position
    const t0 = a0Shot.inE + BEE_WAIT;   // after zooming in on a₀ and waiting 1 s
    const f = Math.min(1, Math.max(0, (frame - t0) / BEE_FLIGHT));
    const e = f * f * (3 - 2 * f);   // smooth take-off and landing
    let bx = beeBase.pos.x - BIG * e, by = beeBase.pos.y + BEE_HEIGHT * Math.sin(Math.PI * e), bz = beeBase.pos.z;
    const r = Math.min(1, f / BEE_TURN), re = r * r * (3 - 2 * r);   // rotation during take-off
    let rot = beeBase.rotY + (-Math.PI / 2 - beeBase.rotY) * re;   // facing left (−x)
    if (a1Shot) {   // second flight: from square A onto the honeycomb on square C (to the front, +z), same timing as the first
      const f2 = Math.min(1, Math.max(0, (frame - (a1Shot.inE + BEE_WAIT)) / BEE_FLIGHT));
      if (f2 > 0) {
        const e2 = f2 * f2 * (3 - 2 * f2), lift = hcTopY === null ? 0 : hcTopY - THICK;
        bx = beeBase.pos.x - BIG; bz = beeBase.pos.z + BIG * e2;
        by = beeBase.pos.y + lift * e2 + BEE_HEIGHT * Math.sin(Math.PI * e2);
        const r2 = Math.min(1, f2 / BEE_TURN), re2 = r2 * r2 * (3 - 2 * r2);
        rot = -Math.PI / 2 + (Math.PI / 2) * re2;   // rotation in flight direction (to the front)
      }
    }
    beeObj.position.set(bx, by, bz);
    beeObj.rotation.y = rot;
  }
  for (const gl of glows) {   // bound to the playback position: glow on clamping (3 s constant, then 2 s fade-out)
    const po = PANELS.find(q => q.name === gl.panelName), cs0 = po && (D.params.clamp_starts || {})[String(po.bits[0])];
    gl.mesh.material.opacity = cs0 === undefined ? 0 : (frame < cs0 ? 0 : Math.max(0, Math.min(1, 1 - (frame - cs0 - GLOW_HOLD) / GLOW_SAMPLES)));
  }
  if (hcGoal) {
    const csG = (D.params.clamp_starts || {})[String(hcGoal.panel.bits[0])];
    hcGoal.obj.visible = csG !== undefined && frame >= csG;
  }
  controls2.update(); renderer2.render(scene2, camera2);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
