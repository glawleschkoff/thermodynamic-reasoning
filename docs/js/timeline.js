// Choreography timeline (all in samples; 10 samples = 1 s at speed 0.1x): camera stations, bee flight, end of the video.
import { PANELS } from './config.js';

const D = window.PBIT_DATA;
const N = D.traj.length;
export const BEE_WAIT = 10, BEE_FLIGHT = 40;      // wait after arrival, flight duration
export const BEE_FLAP_PRE = 10, BEE_FLAP_POST = 10;   // flapping 1 s before take-off and 1 s after landing

export const ZOOM_DIST = 8.5;   // distance to the panel at the end of the fly-in (smaller = closer)
export const SHOTS = ['OBSERVATION o₂', 'OBSERVATION o₀'].map(name => {
  const panel = PANELS.find(q => q.name === name);
  const cs = (D.params.clamp_starts || {})[String(panel.bits[0])];
  return cs === undefined ? null : { panel, inS: cs - 60, inE: cs - 10, outS: cs + 40, outE: cs + 90 };
}).filter(Boolean);
export let a0Shot = null, a1Shot = null;
{   // Station 3: after the fly-out from o₀ wait 20 samples (2 s), then zoom in on ACTION a₀ and stay there
  const prev = SHOTS[SHOTS.length - 1], panel = PANELS.find(q => q.name === 'ACTION a₀');
  if (prev && panel) {
    const inS = prev.outE + 20;
    const inE = inS + 50;
    const outS = inE + BEE_WAIT + BEE_FLIGHT + BEE_FLAP_POST;   // zoom out again right after the flapping ends
    a0Shot = { panel, inS, inE, outS, outE: outS + 50 };
    SHOTS.push(a0Shot);
    // Station 4: after zooming out from a₀, zoom in on o₁ (clamping to A), same sequence as for o₀
    const p1 = PANELS.find(q => q.name === 'OBSERVATION o₁');
    const cs1 = (D.params.clamp_starts || {})[String(p1.bits[0])];
    if (cs1 !== undefined) {
      const s1 = a0Shot.outE + 10;   // fly-in starts here (1 s wide shot after a₀); clamping exactly 60 samples later
      const o1Shot = { panel: p1, inS: s1, inE: s1 + 50, outS: s1 + 100, outE: s1 + 150, clampAt: s1 + 60 };
      SHOTS.push(o1Shot);
      // Station 5: 20 samples (2 s) after zooming out from o₁, zoom in on ACTION a₁, same sequence as for a₀
      const pa1 = PANELS.find(q => q.name === 'ACTION a₁');
      if (pa1) {
        const i1 = o1Shot.outE + 20, i1e = i1 + 50, o1s = i1e + BEE_WAIT + BEE_FLIGHT + BEE_FLAP_POST;
        a1Shot = { panel: pa1, inS: i1, inE: i1e, outS: o1s, outE: o1s + 50 };
        SHOTS.push(a1Shot);
      }
    }
  }
}
// End of the video: 2 s (20 samples) after zooming out from a₁; the play bar shows only this section
export const END_FRAME = SHOTS.length ? Math.min(N - 2, SHOTS[SHOTS.length - 1].outE + 20) : N - 2;
