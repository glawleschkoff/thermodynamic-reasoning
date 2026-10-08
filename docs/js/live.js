// Live Langevin while paused: the browser keeps computing the same dynamics as pbits/simulate.py from the state at the pause
// and blends smoothly back to the stored trajectory on resuming.
import { SIM_RATE, dtFrameOf } from './config.js';
const D = window.PBIT_DATA;
const { alpha, lam, bias, J, dt, stride } = D.params;
const dtFrame = dtFrameOf(stride);
// Set by main.js: getters for the playback state
export const S = { frame: () => 0, playing: () => true, speed: () => 1, stateAt: () => [] };

const BETA = D.params.beta, SIGMA = Math.sqrt(2 * dt / BETA), N_BITS = J.length;
const BLEND_S = 0.5;   // blend time (seconds) from the live state back to the stored trajectory on resuming
let live = null, livePrev = null, liveFrame = -1, liveAcc = 0, prevPlaying = true;
function gauss() { let u = 0, v = 0; while (u === 0) u = Math.random(); v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function liveStep(x, f) {   // one Euler–Maruyama step like langevin_step in pbits/simulate.py (clamping according to plan, depending on frame f)
  const nx = new Array(N_BITS);
  for (let i = 0; i < N_BITS; i++) {
    let jx = 0; const Ji = J[i];
    for (let j = 0; j < N_BITS; j++) jx += Ji[j] * x[j];
    const gradV = alpha * x[i] ** 3 - lam * x[i] - bias[i] - jx;
    nx[i] = x[i] - dt * gradV + SIGMA * gauss();
  }
  const cl = D.params.clamped || {}, cs = D.params.clamp_starts || {};
  for (const k in cl) { const st = cs[k] === undefined ? 0 : cs[k]; if (f >= st) nx[+k] = cl[k]; }
  return nx;
}
let blendOff = null, lastShown = null;
export function displayState(realDt) {   // returns the state to display: stored while playing, live while paused, without a speed kink when switching
  if (!S.playing()) {
    if (prevPlaying || live === null || S.frame() !== liveFrame) {   // start/restart of the live computation at the paused state
      livePrev = S.stateAt(S.frame()); live = livePrev; for (let q = 0; q < stride; q++) live = liveStep(live, S.frame());   // advance one frame immediately so the motion continues right away
      liveFrame = S.frame(); liveAcc = 0;
    }
    liveAcc += S.speed() * SIM_RATE * realDt / dtFrame;   // in frames (`stride` steps each), exactly like the film
    while (liveAcc >= 1) { livePrev = live; for (let q = 0; q < stride; q++) live = liveStep(live, S.frame()); liveAcc -= 1; }
    prevPlaying = false; blendOff = null;
    lastShown = livePrev.map((v, k) => v * (1 - liveAcc) + live[k] * liveAcc);   // linear interpolation between frames as in stateAt
    return lastShown;
  }
  const st = S.stateAt(S.frame());
  if (!prevPlaying) {   // play was just pressed: remember the offset to the film and let it decay (the film continues immediately at full speed)
    prevPlaying = true;
    blendOff = lastShown ? lastShown.map((v, k) => v - st[k]) : null;
  }
  if (blendOff) {
    const f = Math.exp(-realDt / (BLEND_S / 2));
    let m = 0;
    for (let k = 0; k < blendOff.length; k++) { blendOff[k] *= f; m = Math.max(m, Math.abs(blendOff[k])); }
    if (m < 0.003) { blendOff = null; return st; }
    return st.map((v, k) => v + blendOff[k]);
  }
  return st;
}
