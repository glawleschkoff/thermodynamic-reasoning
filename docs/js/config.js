// Viewer constants: colors, panel layout, display parameters (read-only, never modified).

// Colors
export const C_BALL = 0xff8c1a;                          // orange
export const C_GOAL = 0xff8c1a;                          // clamped observation: ball/label
export const C_MAGENTA = 0xff00cc;                       // balls: bright magenta
export const C_ARROW_OFF = 0x7d838f, C_ARROW_ON = 0xffffff;   // arrows: dimmed / lit
export const SQ_SHADOW = [0x8a8e96, 0x14161b], SQ_LIT = [0xffffff, 0x14161b];   // chessboard squares [white, dark]
export const LETTER_SHADOW = 0x6a6e76, LETTER_LIT = 0xffffff;                // letter multiplier
export const C_S = 0x4fa3ff, C_O = 0xffa94f, C_A = 0x5fd38d;                 // panel colors: hidden state, observation, action

// Panel layout as a POMDP wall: time (col) runs to the right; row 0 = actions, 1 = hidden states, 2 = observations.
// Actions sit on half columns (between s_t and s_t+1). bits = (x axis, y axis).
export const PANELS = [
  { name: 'ACTION a₀',       bits: [4, 5], col: 0.5, row: 0, color: C_A },
  { name: 'ACTION a₁',       bits: [10, 11], col: 1.5, row: 0, color: C_A },
  { name: 'HIDDEN STATE s₀', bits: [2, 3], col: 0, row: 1, color: C_S },
  { name: 'HIDDEN STATE s₁', bits: [8, 9], col: 1, row: 1, color: C_S },
  { name: 'HIDDEN STATE s₂', bits: [14, 15], col: 2, row: 1, color: C_S },
  { name: 'OBSERVATION o₀',  bits: [0, 1], col: 0, row: 2, color: C_O },
  { name: 'OBSERVATION o₁',  bits: [6, 7], col: 1, row: 2, color: C_O },
  { name: 'OBSERVATION o₂',  bits: [12, 13], col: 2, row: 2, color: C_O },
];

// Display
export const SPACING = 4.6, COL_SPACING = 7.4, RANGE = 1.35, RES = 56;
export const HSCALE = 0.5, EC = 2;   // height = HSCALE·EC·ln(1 + e/EC): linear for small e (valleys), compressed for large e
export const SHOW_WALLS = false;

// Playback rate: fixed simulation time per sample (independent of dt) so the choreography keeps the same length
export const SIM_RATE = 2.0;   // simulation time per second at speed 1
export const dtFrameOf = (stride) => 0.01 * stride;
