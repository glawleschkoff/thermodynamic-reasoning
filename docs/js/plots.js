// 2D plots in the sliding window: probabilities (ensemble) + exact Bayes posteriors over time.
import { PANELS } from './config.js';

const D = window.PBIT_DATA;
// Set by main.js: getFrame() returns the playback position, END_FRAME the end of the video.
export const ctx = { getFrame: () => 0, END_FRAME: 1 };

// ---- 2D plots in the sliding window: one plot per panel ----
const THUMB_HALF = 12;   // half width of the slider thumb (px); the thumb center runs from tr.left+12 to tr.right-12
export const plotsEl = document.getElementById('plots');
export const plotState = { lastShot: -2 };
const PLOT_SCROLL_MS = 2500;   // duration of the automatic scrolling (ms)
export const scrollState = { id: 0 };
let held = false;   // finger/mouse on the plot window: no automatic scrolling
for (const ev of ['touchstart', 'pointerdown']) plotsEl.addEventListener(ev, () => { held = true; scrollState.id++; }, { passive: true });
for (const ev of ['touchend', 'touchcancel', 'pointerup']) window.addEventListener(ev, () => { held = false; }, { passive: true });
let wheelUntil = 0;   // mouse wheel on the plot window: pause automatic scrolling briefly
plotsEl.addEventListener('wheel', () => { wheelUntil = performance.now() + 2500; scrollState.id++; }, { passive: true });
export function slowScroll(target) {
  if (held || performance.now() < wheelUntil) return;
  const from = plotsEl.scrollTop, t0 = performance.now(), id = ++scrollState.id;
  const step = (now) => {
    if (id !== scrollState.id) return;
    const u = Math.min(1, (now - t0) / PLOT_SCROLL_MS);
    const e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
    plotsEl.scrollTop = from + (target - from) * e;
    if (u < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
const PLOT_ORDER = ['OBSERVATION o₂', 'OBSERVATION o₀', 'ACTION a₀', 'OBSERVATION o₁', 'ACTION a₁', 'HIDDEN STATE s₀', 'HIDDEN STATE s₁', 'HIDDEN STATE s₂'];
export const plotItems = PLOT_ORDER.map(n => PANELS.find(q => q.name === n)).map(p => {
  const box = document.createElement('div'); box.className = 'plot';
  const cv = document.createElement('canvas');
  const dv = document.createElement('canvas');   // data lines (redrawn every frame)
  const cur = document.createElement('div'); cur.className = 'cur';
  const ttl = document.createElement('div'); ttl.className = 'ttl'; ttl.textContent = p.name;
  box.append(cv, dv, cur, ttl); plotsEl.appendChild(box);
  const ens = (D.ens || {})[p.bits[0] + ',' + p.bits[1]] || null;   // ensemble probabilities [frame][4] from pbits/simulate.py
  return { p, box, cv, dv, cur, ens };
});
const PL_COLS = ['#ff4fd8', '#5ec8ff', '#ffd23f', '#6ee7a0'];
function drawData(it) {   // distribution p(t) from the ensemble, continuous up to the current frame
  const G = it.G; if (!G || !it.ens) return;
  const g = it.dv.getContext('2d');
  g.setTransform(G.pr, 0, 0, G.pr, 0, 0); g.clearRect(0, 0, G.w, G.h);
  const f = Math.min(Math.floor(ctx.getFrame()), ctx.END_FRAME, it.ens.length - 1), step = Math.max(1, Math.floor(ctx.END_FRAME / (G.x1 - G.x0)));
  for (let k = 0; k < 4; k++) {
    g.strokeStyle = PL_COLS[k]; g.lineWidth = 1.8; g.beginPath();
    for (let i = 0; i <= f; i += step) {
      const X = G.x0 + i / ctx.END_FRAME * (G.x1 - G.x0), Y = G.y0 - it.ens[i][k] * (G.y0 - G.y1);
      i === 0 ? g.moveTo(X, Y) : g.lineTo(X, Y);
    }
    g.stroke();
  }
}
export function drawPlots() {
  const pr = Math.min(2, window.devicePixelRatio);
  plotItems.forEach((it) => {
    const { p, box, cv } = it;
    const w = box.clientWidth, h = box.clientHeight;
    if (!w || !h) return;
    cv.width = Math.round(w * pr); cv.height = Math.round(h * pr);
    it.dv.width = cv.width; it.dv.height = cv.height;
    const g = cv.getContext('2d'); g.scale(pr, pr);
    g.clearRect(0, 0, w, h);
    const tr = document.getElementById("t").getBoundingClientRect(), bl = box.getBoundingClientRect().left;
    const x0 = tr.left - bl + THUMB_HALF, x1 = tr.right - bl - THUMB_HALF, y0 = h - 40, y1 = 38;
    it.G = { pr, w, h, x0, x1, y0, y1 };
    g.font = '11px -apple-system, sans-serif';
    g.fillStyle = '#9aa4b5'; g.strokeStyle = '#3a4356'; g.lineWidth = 1;
    g.textAlign = 'right'; g.textBaseline = 'middle';
    [0, 0.5, 1].forEach(v => {
      const Y = y0 - v * (y0 - y1);
      g.beginPath(); g.moveTo(x0, Y); g.lineTo(x1, Y); g.stroke();
      g.fillText(v.toFixed(1), x0 - 6, Y);
    });
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (let k = 0; k <= 5; k++) {
      const X = x0 + k / 5 * (x1 - x0);
      if (k === 0) { g.textBaseline = 'middle'; g.fillText('0', X, y0 + 20); }
    }
    g.strokeStyle = '#8a93a6';
    g.beginPath(); g.moveTo(x0, y1); g.lineTo(x0, y0); g.lineTo(x1, y0); g.stroke();
    {   // exact Bayes posteriors: dashed horizontal lines (between the clamping of o2, o0 and o1)
      const cs = D.params.clamp_starts || {}, key = p.bits[0] + ',' + p.bits[1];
      [[D.bayes, '12', '0'], [D.bayes2, '0', '6'], [D.bayes3, '6', null]].forEach(([tab, c0, c1]) => {
        const bz = (tab || {})[key];
        if (!bz || cs[c0] === undefined || (c1 !== null && cs[c1] === undefined)) return;
        const xa = x0 + cs[c0] / ctx.END_FRAME * (x1 - x0), xb = c1 === null ? x1 : x0 + cs[c1] / ctx.END_FRAME * (x1 - x0);
        g.save(); g.setLineDash([6, 4]); g.lineWidth = 1.6;
        bz.forEach((v, k) => { const Y = y0 - v * (y0 - y1); g.strokeStyle = PL_COLS[k]; g.beginPath(); g.moveTo(xa, Y); g.lineTo(xb, Y); g.stroke(); });
        g.restore();
      });
    }
    PANELS.filter(q => q.name.startsWith('OBSERVATION')).forEach(q => {   // ticks at the clamp times of the three o
      const cs = (D.params.clamp_starts || {})[String(q.bits[0])];
      if (cs === undefined) return;
      const X = x0 + cs / ctx.END_FRAME * (x1 - x0);
      g.save(); g.strokeStyle = 'rgba(255,107,107,0.6)'; g.lineWidth = 1;   // fine reddish line through the whole plot
      g.beginPath(); g.moveTo(X, y1); g.lineTo(X, y0 + 8); g.stroke(); g.restore();
      const dg = '₀₁₂₃₄₅₆₇₈₉'.indexOf(q.name.slice(-1));   // label: o_i clamped (small subscript index)
      const fs = 11, sf = fs * 0.78, t1 = 'o', t2 = String(dg), t3 = ' clamped';
      g.font = fs + 'px -apple-system, sans-serif'; const w1 = g.measureText(t1).width, w3 = g.measureText(t3).width;
      g.font = sf + 'px -apple-system, sans-serif'; const w2 = g.measureText(t2).width + 1;
      const lx = Math.min(Math.max(X - (w1 + w2 + w3) / 2, 2), w - (w1 + w2 + w3) - 2);
      g.save(); g.fillStyle = '#9aa4b5'; g.textAlign = 'left'; g.textBaseline = 'middle';
      g.font = fs + 'px -apple-system, sans-serif'; g.fillText(t1, lx, y0 + 20);
      g.font = sf + 'px -apple-system, sans-serif'; g.fillText(t2, lx + w1 + 1, y0 + 20 + fs * 0.28);
      g.font = fs + 'px -apple-system, sans-serif'; g.fillText(t3, lx + w1 + w2, y0 + 20); g.restore();
    });
    g.font = '11px -apple-system, sans-serif'; g.fillStyle = '#9aa4b5'; g.textAlign = 'center';
    g.save(); g.translate(x0 - 6 - g.measureText('0.0').width - 8 - 4, (y0 + y1) / 2); g.rotate(-Math.PI / 2);
    g.textBaseline = 'middle'; g.fillText('Probability', 0, 0); g.restore();
    g.save(); g.translate(x0 - 62, h / 2); g.rotate(-Math.PI / 2);
    g.font = '600 13px sans-serif'; g.fillStyle = '#cbd5e1'; g.textAlign = 'center'; g.textBaseline = 'middle';
    {
      const FS = 13, SUBFS = FS * 0.78;
      const parts = [...p.name].map(c => ({ c: '₀₁₂₃₄₅₆₇₈₉'.indexOf(c) >= 0 ? String('₀₁₂₃₄₅₆₇₈₉'.indexOf(c)) : c, sub: '₀₁₂₃₄₅₆₇₈₉'.indexOf(c) >= 0 }));
      const wOf = q => { g.font = '600 ' + (q.sub ? SUBFS : FS) + 'px sans-serif'; return g.measureText(q.c).width + (q.sub ? 1 : 0); };
      const total = parts.reduce((a, q) => a + wOf(q), 0);
      let cx = -total / 2; g.textAlign = 'left';
      parts.forEach(q => { g.font = '600 ' + (q.sub ? SUBFS : FS) + 'px sans-serif'; g.fillText(q.c, cx + (q.sub ? 1 : 0), q.sub ? FS * 0.28 : 0); cx += wOf(q); });
    }
    g.restore();
    g.font = '11px -apple-system, sans-serif'; g.fillStyle = '#9aa4b5'; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText('Time', (x0 + x1) / 2, y0 + 20);
    const labs = p.name.toLowerCase().startsWith('action') ? ['\u2191', '\u2192', '\u2193', '\u2190'] : ['A', 'B', 'C', 'D'];
    const cols = ['#ff4fd8', '#5ec8ff', '#ffd23f', '#6ee7a0'];
    const LY = 21, STEP = 62, FS = 18;
    g.textAlign = 'left'; g.textBaseline = 'middle';
    let X = x1 - labs.length * STEP;
    labs.forEach((l, i) => {   // four colored entries (larger so the assignment is quickly readable)
      g.strokeStyle = cols[i]; g.lineWidth = 4;
      g.beginPath(); g.moveTo(X, LY); g.lineTo(X + 22, LY); g.stroke();
      g.fillStyle = cols[i]; const isArr = i >= 0 && p.name.toLowerCase().startsWith('action'); g.font = '700 ' + (isArr ? Math.round(FS * 1.45) : FS) + 'px -apple-system, sans-serif'; g.fillText(l, X + 28, LY + (isArr ? 1 : 0));
      X += STEP;
    });
    {   // gray entries: solid = Langevin ensemble, dashed = exact Bayes posterior
      g.font = '600 13px -apple-system, sans-serif'; g.fillStyle = '#cbd5e1'; g.strokeStyle = '#cbd5e1'; g.lineWidth = 2.5;
      const items = [['Langevin ensemble', []], ['Bayes posterior', [6, 4]]];
      let xr = x1 - labs.length * STEP - 24;
      for (let n = items.length - 1; n >= 0; n--) {
        const [txt, dash] = items[n], tw = g.measureText(txt).width;
        xr -= tw; g.fillText(txt, xr, LY);
        xr -= 8; g.save(); g.setLineDash(dash); g.beginPath(); g.moveTo(xr - 26, LY); g.lineTo(xr, LY); g.stroke(); g.restore();
        xr -= 26 + 22;
      }
    }
  });
}
export function updatePlots() {
  const f = Math.min(ctx.getFrame(), ctx.END_FRAME) / ctx.END_FRAME;
  plotItems.forEach(it => {
    const tr = document.getElementById('t').getBoundingClientRect(), bl = it.box.getBoundingClientRect().left;
    it.cur.style.left = (tr.left - bl + THUMB_HALF + f * (tr.width - 2 * THUMB_HALF)) + 'px';
    it.cur.style.top = '26px'; it.cur.style.bottom = '24px';
    drawData(it);
  });
}
new ResizeObserver(drawPlots).observe(plotsEl);
window.addEventListener('resize', drawPlots);
drawPlots();
