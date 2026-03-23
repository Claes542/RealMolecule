// molecule2.js — Faithful GPU port of original RealQM for arbitrary molecules
// N atoms at any 3D positions, 1 valence electron per atom, r_c pseudopotential
// Exactly the original algorithm: w level set, w-weighted Laplacian, coupled Poisson
// c[m] = 0.5*(u[m] - max(u[others])), P[m] sourced by sum of other u^2
"use strict";

// --- Configuration ---
const NN = window.USER_NN || 100;
const screenAu = window.USER_SCREEN || 10;
const MAX_STEPS = window.USER_STEPS || 2000;
const NORM_INTERVAL = 1;

const S = NN + 1;
const S2 = S * S;
const S3 = S * S * S;
const N2 = Math.floor(NN / 2);
const h = screenAu / NN;
const h2 = h * h;
const h3 = h * h * h;
const dv = 0.1;
const dt = dv * h2;
const half_d = 0.5 * dv;
const TWO_PI = 2 * Math.PI;

// 2D dispatch config
const WG_SIZE = 256;
const INTERIOR = (NN - 1) * (NN - 1) * (NN - 1);
const THREADS_X = 512;
const DISPATCH_Y = Math.ceil(Math.ceil(INTERIOR / WG_SIZE) / THREADS_X);
const DISPATCH_X = Math.min(THREADS_X, Math.ceil(INTERIOR / WG_SIZE));

// Reduction config
const REDUCE_WG = 128;
const N_REDUCE_WG = Math.min(256, Math.ceil(INTERIOR / REDUCE_WG));

// --- Atom configuration ---
// USER_ATOMS: array of { i, j, k, Z, rc } in grid coordinates
// One valence electron per atom
const atoms = window.USER_ATOMS || [
  { i: N2 - 7, j: N2, k: N2, Z: 1, rc: 0 },
  { i: N2 + 7, j: N2, k: N2, Z: 1, rc: 0 },
];
// Z_nuc: nuclear charge (defaults to Z). Bare protons: Z=0, Z_nuc=1
const _z_nuc = window.USER_Z_NUC || atoms.map(a => a.Z);
// Init positions for electrons (override wavefunction center)
const _initPos = window.USER_INIT_POS || atoms.map(() => null);
// NELEC counts atoms with electrons (Z>0)
const NELEC = atoms.filter(a => a.Z > 0).length;
// All atoms including bare protons (for K potential)
const ALL_ATOMS = atoms.length;

// Nuclear repulsion
let V_KK = 0;
if (!window.NO_NUC_REPULSION) {
  for (let a = 0; a < atoms.length; a++)
    for (let b = a + 1; b < atoms.length; b++) {
      const za = _z_nuc[a] || atoms[a].Z;
      const zb = _z_nuc[b] || atoms[b].Z;
      if (za === 0 || zb === 0) continue;
      const di = (atoms[a].i - atoms[b].i) * h;
      const dj = (atoms[a].j - atoms[b].j) * h;
      const dk = (atoms[a].k - atoms[b].k) * h;
      V_KK += za * zb / Math.sqrt(di*di + dj*dj + dk*dk + h2);
    }
}

console.log(`molecule2: NN=${NN}, ${NELEC} atoms/electrons, h=${h.toFixed(4)}, dt=${dt.toExponential(3)}`);
console.log(`Dispatch: ${DISPATCH_X} x ${DISPATCH_Y} workgroups of ${WG_SIZE}`);
console.log(`V_KK = ${V_KK.toFixed(6)}`);
for (const a of atoms)
  console.log(`  Atom Z=${a.Z} at (${a.i},${a.j},${a.k}) rc=${a.rc||0}`);

// --- WGSL Shaders ---

const paramStructWGSL = `
struct P {
  NN: u32, S: u32, S2: u32, S3: u32,
  N2: u32, dt: f32, half_d: f32, TWO_PI: f32,
  h: f32, h2: f32, inv_h: f32, inv_h2: f32,
  h3: f32, _p0: u32, _p1: u32, _p2: u32,
}`;

const CELLS_PER_ROW = DISPATCH_X * WG_SIZE;
const cellIdxWGSL = `
fn cellIdx(gid: vec3<u32>) -> u32 {
  return gid.x + gid.y * ${CELLS_PER_ROW}u;
}`;

// Init K shader: accumulates Z/r for one nucleus at a time
const initK_WGSL = `
${paramStructWGSL}
struct NucCfg { posI: f32, posJ: f32, posK: f32, Z: f32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> K: array<f32>;
@group(0) @binding(2) var<uniform> nuc: NucCfg;

${cellIdxWGSL}
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let dx = (f32(i) - nuc.posI) * p.h;
  let dy = (f32(j) - nuc.posJ) * p.h;
  let dz = (f32(k) - nuc.posK) * p.h;
  let r = sqrt(dx*dx + dy*dy + dz*dz + p.h2);
  K[id] += nuc.Z / r;
}
`;

// Init one electron: u = exp(-r), w = 1 in Voronoi domain, w = 0 inside r_c
// Domain via precomputed Voronoi mask
const initElec_WGSL = `
${paramStructWGSL}
struct ElecCfg { nucI: f32, nucJ: f32, nucK: f32, nucRC: f32, initR: f32, _p0: f32, _p1: f32, _p2: f32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> u_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> w_out: array<f32>;
@group(0) @binding(3) var<uniform> cfg: ElecCfg;
@group(0) @binding(4) var<storage, read> domain: array<f32>;

${cellIdxWGSL}
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  if (domain[id] < 0.5) { return; }

  let dx = (f32(i) - cfg.nucI) * p.h;
  let dy = (f32(j) - cfg.nucJ) * p.h;
  let dz = (f32(k) - cfg.nucK) * p.h;
  let r_raw = sqrt(dx*dx + dy*dy + dz*dz);
  let r_soft = sqrt(dx*dx + dy*dy + dz*dz + p.h2);

  if (cfg.nucRC > 0.0 && r_raw < cfg.nucRC) {
    w_out[id] = 0.0;
  } else {
    w_out[id] = 1.0;
  }
  if (cfg.initR > 0.0 && r_raw > cfg.initR) {
    u_out[id] = 0.0;
  } else {
    u_out[id] = exp(-r_soft);
  }
}
`;

// Precompute c and rho_other for one electron
// c[m] = 0.5*(u[m] - max(u[others]))
// rho_other = sum of u[n]^2 for n != m
function genPrecomputeWGSL() {
  if (NELEC <= 5) {
    let bindings = `@group(0) @binding(0) var<uniform> p: P;\n`;
    for (let m = 0; m < NELEC; m++)
      bindings += `@group(0) @binding(${m+1}) var<storage, read> u${m}: array<f32>;\n`;
    bindings += `@group(0) @binding(${NELEC+1}) var<storage, read_write> c_out: array<f32>;\n`;
    bindings += `@group(0) @binding(${NELEC+2}) var<storage, read_write> rho_out: array<f32>;\n`;
    bindings += `@group(0) @binding(${NELEC+3}) var<uniform> ecfg: ElecIdx;\n`;

    let selfExpr = `u${NELEC-1}[id]`;
    for (let m = NELEC - 2; m >= 0; m--)
      selfExpr = `select(${selfExpr}, u${m}[id], ecfg.idx == ${m}u)`;

    let body = '';
    for (let m = 0; m < NELEC; m++)
      body += `  if (ecfg.idx != ${m}u) { max_other = max(max_other, u${m}[id]); sum_rho += u${m}[id] * u${m}[id]; }\n`;

    return `${paramStructWGSL}
struct ElecIdx { idx: u32, _p0: u32, _p1: u32, _p2: u32, }
${bindings}
${cellIdxWGSL}
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let u_self_val = ${selfExpr};
  var max_other: f32 = -1e30;
  var sum_rho: f32 = 0.0;
${body}
  c_out[id] = 0.5 * (u_self_val - max_other);
  rho_out[id] = sum_rho;
}`;
  } else {
    // For NELEC > 5: packed buffer with loop
    return `${paramStructWGSL}
struct ElecIdx { idx: u32, nelec: u32, _p0: u32, _p1: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> packed_u: array<f32>;
@group(0) @binding(2) var<storage, read_write> c_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> rho_out: array<f32>;
@group(0) @binding(4) var<uniform> ecfg: ElecIdx;

${cellIdxWGSL}
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let u_self_val = packed_u[ecfg.idx * p.S3 + id];
  var max_other: f32 = -1e30;
  var sum_rho: f32 = 0.0;
  for (var m: u32 = 0u; m < ecfg.nelec; m++) {
    if (m != ecfg.idx) {
      let uv = packed_u[m * p.S3 + id];
      max_other = max(max_other, uv);
      sum_rho += uv * uv;
    }
  }
  c_out[id] = 0.5 * (u_self_val - max_other);
  rho_out[id] = sum_rho;
}`;
  }
}

// Fused w+u update — reads precomputed c_buf
const fusedWU_WGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> u_self: array<f32>;
@group(0) @binding(2) var<storage, read> c_buf: array<f32>;
@group(0) @binding(3) var<storage, read> w_self: array<f32>;
@group(0) @binding(4) var<storage, read> Kbuf: array<f32>;
@group(0) @binding(5) var<storage, read> Pot: array<f32>;
@group(0) @binding(6) var<storage, read_write> w_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> u_out: array<f32>;

@group(1) @binding(0) var<storage, read> rcMask: array<f32>;

${cellIdxWGSL}
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let mask = rcMask[id];
  let c = c_buf[id];
  let wc = w_self[id];

  // --- Update w (level set) ---
  let lap_w = w_self[id + p.S2] + w_self[id - p.S2]
            + w_self[id + p.S]  + w_self[id - p.S]
            + w_self[id + 1u]   + w_self[id - 1u] - 6.0 * wc;

  let gwx = (w_self[id + p.S2] - w_self[id - p.S2]) * p.inv_h;
  let gwy = (w_self[id + p.S]  - w_self[id - p.S])  * p.inv_h;
  let gwz = (w_self[id + 1u]   - w_self[id - 1u])   * p.inv_h;
  let grad_w = sqrt(gwx*gwx + gwy*gwy + gwz*gwz);

  var new_w = select(clamp(wc + 2.0 * p.dt * abs(c) * lap_w * p.inv_h2 + 10.0 * p.dt * c * grad_w, 0.0, 1.0), wc, ${window.FREEZE_BOUNDARY ? 'true' : 'false'});
  new_w *= mask;
  w_out[id] = new_w;

  // --- Update u using new w at center ---
  let uc = u_self[id];

  let flux_xp = (u_self[id + p.S2] - uc) * (w_self[id + p.S2] + new_w) * 0.5;
  let flux_xm = (uc - u_self[id - p.S2]) * (new_w + w_self[id - p.S2]) * 0.5;
  let flux_yp = (u_self[id + p.S] - uc) * (w_self[id + p.S] + new_w) * 0.5;
  let flux_ym = (uc - u_self[id - p.S]) * (new_w + w_self[id - p.S]) * 0.5;
  let flux_zp = (u_self[id + 1u] - uc) * (w_self[id + 1u] + new_w) * 0.5;
  let flux_zm = (uc - u_self[id - 1u]) * (new_w + w_self[id - 1u]) * 0.5;

  let wlap = (flux_xp - flux_xm) + (flux_yp - flux_ym) + (flux_zp - flux_zm);

  u_out[id] = uc + p.half_d * wlap + p.dt * (Kbuf[id] - 2.0 * Pot[id]) * uc * new_w;
}
`;

// Single-electron P update
const updateP_WGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> P_in: array<f32>;
@group(0) @binding(2) var<storage, read> rho_src: array<f32>;
@group(0) @binding(3) var<storage, read_write> P_out: array<f32>;

${cellIdxWGSL}
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let Pc = P_in[id];
  let lap_P = P_in[id + p.S2] + P_in[id - p.S2]
            + P_in[id + p.S]  + P_in[id - p.S]
            + P_in[id + 1u]   + P_in[id - 1u] - 6.0 * Pc;
  P_out[id] = Pc + p.dt * (lap_P * p.inv_h2 + p.TWO_PI * rho_src[id]);
}
`;

// Accumulate rho_total: rhoOut[id] += 2 * u[id]^2
// Factor of 2 so that existing 2π Poisson gives Lap(P)=-4π*rho
const accumRhoWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> rhoOut: array<f32>;

${cellIdxWGSL}
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;
  let v = u[id];
  rhoOut[id] += 2.0 * v * v;
}
`;

// Norm reduction
const reduceNormWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;

var<workgroup> sdata: array<f32, ${REDUCE_WG}>;

@compute @workgroup_size(${REDUCE_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(workgroup_id) wgid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let stride = ${N_REDUCE_WG}u * ${REDUCE_WG}u;

  var acc: f32 = 0.0;
  var cell = gid.x;
  loop {
    if (cell >= tot) { break; }
    let k = (cell % NM) + 1u;
    let j = ((cell / NM) % NM) + 1u;
    let i = (cell / (NM * NM)) + 1u;
    let id = i * p.S2 + j * p.S + k;
    let v = u[id];
    acc += v * v * p.h3;
    cell += stride;
  }
  sdata[lid] = acc;
  workgroupBarrier();
  var s: u32 = ${REDUCE_WG >> 1}u;
  loop {
    if (s == 0u) { break; }
    if (lid < s) { sdata[lid] += sdata[lid + s]; }
    workgroupBarrier();
    s = s >> 1u;
  }
  if (lid == 0u) { partials[wgid.x] = sdata[0]; }
}
`;

// Normalize
const normalizeWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read> normBuf: array<f32>;

${cellIdxWGSL}
@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  var norm: f32 = 0.0;
  for (var q: u32 = 0u; q < ${N_REDUCE_WG}u; q++) { norm += normBuf[q]; }

  if (norm > 0.0) { u[id] *= 1.0 / sqrt(norm); }
}
`;

// Extract 2D slice for visualization (up to 4 electrons)
const MAX_VIS = Math.min(NELEC, 4);
const SLICE_STRIDE = MAX_VIS * 3 + 1;
const extractSliceWGSL = `
${paramStructWGSL}
struct SliceCfg { offset: u32, _pad0: u32, _pad1: u32, _pad2: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> u_el: array<f32>;
@group(0) @binding(2) var<storage, read> w_el: array<f32>;
@group(0) @binding(3) var<storage, read> P_el: array<f32>;
@group(0) @binding(4) var<storage, read> Kbuf: array<f32>;
@group(0) @binding(5) var<storage, read_write> slice: array<f32>;
@group(0) @binding(6) var<uniform> scfg: SliceCfg;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let N1 = p.NN + 1u;
  let tot = N1 * N1;
  if (gid.x >= tot) { return; }
  let j = gid.x / N1;
  let i = gid.x % N1;
  let id3 = i * p.S2 + j * p.S + p.N2;
  let base = gid.x * ${SLICE_STRIDE}u + scfg.offset;
  slice[base]      = u_el[id3];
  slice[base + 1u] = w_el[id3];
  slice[base + 2u] = P_el[id3];
  if (scfg.offset == 0u) {
    slice[gid.x * ${SLICE_STRIDE}u + ${MAX_VIS * 3}u] = Kbuf[id3];
  }
}
`;

// --- GPU State ---
let device, gpuReady = false, gpuError = null;
let paramsBuf, nucCfgBuf, elecCfgBuf;
let elecIdxBufs = [];  // per-electron uniform buffers (avoids writeBuffer race)
let K_buf;
let u_buf = [], w_buf = [], P_buf = [];
let normPartialBuf = [];
let c_buf, rho_buf;
let rcMask_buf = [];
let domain_buf = [];
let rhoTotal_buf, Ptot_buf, Ptot_ping = 0;
let accumRhoBG = [];
let updatePtotBG = [];
let sliceBuf, sliceReadBuf, sliceCfgBuf;
let packedU_buf = null;
let cur = 0;

let initK_PL, initElec_PL, precompPL, fusedWU_PL, updateP_PL;
let accumRhoPL;
let reduceNormPL, normalizePL, extractSlicePL;
let initK_BG;
let precompBG = [];
let fusedWU_BG = [], fusedWU_mask_BG = [];
let updateP_BG = [];
let reduceNormBG = [], normalizeBG = [];
let extractSliceBG = [];

// Energy values
let E_T = 0, E_eK = 0, E_ee = 0, E_pot = 0, E_tot = 0;
let stepCount = 0;
let sliceData = null;

// Compute Voronoi domain for electron m
// Uses init positions (_initPos) when available, so He electrons get separate half-spaces
function computeVoronoiDomain(m) {
  const domain = new Float32Array(S3);
  const am = atoms[m];
  // Use init position for domain center if available (e.g. He half-space)
  const ipM = _initPos[m];
  const mi = ipM ? ipM[0] : am.i;
  const mj = ipM ? ipM[1] : am.j;
  const mk = ipM && ipM[2] !== undefined ? ipM[2] : (am.k || N2);
  for (let i = 0; i <= NN; i++) {
    for (let j = 0; j <= NN; j++) {
      for (let k = 0; k <= NN; k++) {
        const dx0 = (i - mi) * h, dy0 = (j - mj) * h, dz0 = (k - mk) * h;
        let myDist2 = dx0*dx0 + dy0*dy0 + dz0*dz0;
        let closest = true;
        for (let n = 0; n < ALL_ATOMS; n++) {
          if (n === m || atoms[n].Z <= 0) continue; // skip bare protons and self
          const ipN = _initPos[n];
          const ni = ipN ? ipN[0] : atoms[n].i;
          const nj = ipN ? ipN[1] : atoms[n].j;
          const nk = ipN && ipN[2] !== undefined ? ipN[2] : (atoms[n].k || N2);
          const dx = (i - ni) * h, dy = (j - nj) * h, dz = (k - nk) * h;
          if (dx*dx + dy*dy + dz*dz < myDist2) { closest = false; break; }
        }
        if (closest) domain[i * S2 + j * S + k] = 1.0;
      }
    }
  }
  return domain;
}

// Compute r_c mask for electron m
function computeRcMask(m) {
  const mask = new Float32Array(S3);
  mask.fill(1.0);
  const a = atoms[m];
  const rc = a.rc || 0;
  if (rc > 0) {
    const rc2 = rc * rc;
    const rcCells = Math.ceil(rc / h) + 1;
    for (let di = -rcCells; di <= rcCells; di++)
      for (let dj = -rcCells; dj <= rcCells; dj++)
        for (let dk = -rcCells; dk <= rcCells; dk++) {
          const gi = a.i + di, gj = a.j + dj, gk = a.k + dk;
          if (gi < 0 || gi >= S || gj < 0 || gj >= S || gk < 0 || gk >= S) continue;
          const r2 = (di * h) * (di * h) + (dj * h) * (dj * h) + (dk * h) * (dk * h);
          if (r2 < rc2) mask[gi * S2 + gj * S + gk] = 0.0;
        }
    console.log(`Electron ${m}: w=0 inside r_c=${rc} of atom ${m}`);
  }
  return mask;
}

async function initGPU() {
  try {
    if (!navigator.gpu) { gpuError = "WebGPU not supported in this browser"; return; }
    let adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) adapter = await navigator.gpu.requestAdapter();
    if (!adapter) { gpuError = "No WebGPU adapter found"; return; }
    const maxBuf = Math.min(adapter.limits.maxStorageBufferBindingSize,
      S3 * 4 * (NELEC > 5 ? NELEC : 1));
    device = await adapter.requestDevice({
      requiredLimits: { maxStorageBufferBindingSize: maxBuf, maxBufferSize: maxBuf }
    });
    device.lost.then(info => { gpuError = "GPU lost: " + info.message; gpuReady = false; });

    const bs = S3 * 4;
    const STOR = GPUBufferUsage.STORAGE;
    const COPY = GPUBufferUsage.COPY_DST;
    const COPY_SRC = GPUBufferUsage.COPY_SRC;

    // Params uniform
    paramsBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | COPY });
    const pb = new ArrayBuffer(64);
    const pu = new Uint32Array(pb);
    const pf = new Float32Array(pb);
    pu[0] = NN; pu[1] = S; pu[2] = S2; pu[3] = S3;
    pu[4] = N2; pf[5] = dt; pf[6] = half_d; pf[7] = TWO_PI;
    pf[8] = h; pf[9] = h2; pf[10] = 1/h; pf[11] = 1/h2;
    pf[12] = h3; pu[13] = 0; pu[14] = 0; pu[15] = 0;
    device.queue.writeBuffer(paramsBuf, 0, pb);

    // Config uniforms
    nucCfgBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | COPY });
    elecCfgBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | COPY });
    // Per-electron index buffers (pre-filled, avoids writeBuffer inside command encoder loop)
    for (let m = 0; m < NELEC; m++) {
      elecIdxBufs[m] = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | COPY });
      device.queue.writeBuffer(elecIdxBufs[m], 0, new Uint32Array([m, NELEC, 0, 0]));
    }

    // Buffers
    K_buf = device.createBuffer({ size: bs, usage: STOR | COPY | COPY_SRC });
    const zeros = new Float32Array(S3);
    device.queue.writeBuffer(K_buf, 0, zeros);

    c_buf = device.createBuffer({ size: bs, usage: STOR | COPY });
    rho_buf = device.createBuffer({ size: bs, usage: STOR | COPY });

    // Per-electron buffers
    for (let m = 0; m < NELEC; m++) {
      u_buf[m] = [
        device.createBuffer({ size: bs, usage: STOR | COPY | COPY_SRC }),
        device.createBuffer({ size: bs, usage: STOR | COPY | COPY_SRC }),
      ];
      w_buf[m] = [
        device.createBuffer({ size: bs, usage: STOR | COPY | COPY_SRC }),
        device.createBuffer({ size: bs, usage: STOR | COPY | COPY_SRC }),
      ];
      P_buf[m] = [
        device.createBuffer({ size: bs, usage: STOR | COPY | COPY_SRC }),
        device.createBuffer({ size: bs, usage: STOR | COPY | COPY_SRC }),
      ];
      for (let c = 0; c < 2; c++) {
        device.queue.writeBuffer(u_buf[m][c], 0, zeros);
        device.queue.writeBuffer(w_buf[m][c], 0, zeros);
        device.queue.writeBuffer(P_buf[m][c], 0, zeros);
      }
      normPartialBuf[m] = device.createBuffer({ size: N_REDUCE_WG * 4, usage: STOR | COPY });

      const maskData = computeRcMask(m);
      rcMask_buf[m] = device.createBuffer({ size: bs, usage: STOR | COPY });
      device.queue.writeBuffer(rcMask_buf[m], 0, maskData);

      const domainData = computeVoronoiDomain(m);
      domain_buf[m] = device.createBuffer({ size: bs, usage: STOR | COPY });
      device.queue.writeBuffer(domain_buf[m], 0, domainData);
    }

    // Packed u buffer for NELEC > 5
    if (NELEC > 5) {
      packedU_buf = device.createBuffer({ size: bs * NELEC, usage: STOR | COPY | COPY_SRC });
    }

    // Total rho and Ptot buffers for force computation
    rhoTotal_buf = device.createBuffer({ size: bs, usage: STOR | COPY });
    Ptot_buf = [
      device.createBuffer({ size: bs, usage: STOR | COPY | COPY_SRC }),
      device.createBuffer({ size: bs, usage: STOR | COPY | COPY_SRC }),
    ];
    device.queue.writeBuffer(rhoTotal_buf, 0, zeros);
    device.queue.writeBuffer(Ptot_buf[0], 0, zeros);
    device.queue.writeBuffer(Ptot_buf[1], 0, zeros);

    // Slice buffer
    const sliceSize = S * S * SLICE_STRIDE * 4;
    sliceBuf = device.createBuffer({ size: sliceSize, usage: STOR | COPY_SRC | COPY });
    sliceReadBuf = device.createBuffer({ size: sliceSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    sliceCfgBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | COPY });

    // Compile shaders
    async function compile(name, code) {
      const mod = device.createShaderModule({ code });
      const info = await mod.getCompilationInfo();
      for (const msg of info.messages)
        console[msg.type === 'error' ? 'error' : 'warn'](
          `${name} ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
      if (info.messages.some(m => m.type === 'error'))
        throw new Error(`Shader ${name} compilation failed`);
      console.log(`${name} compiled OK`);
      return mod;
    }
    const initK_Mod = await compile('initK', initK_WGSL);
    const initElec_Mod = await compile('initElec', initElec_WGSL);
    const precompMod = await compile('precompute', genPrecomputeWGSL());
    const fusedWU_Mod = await compile('fusedWU', fusedWU_WGSL);
    const updateP_Mod = await compile('updateP', updateP_WGSL);
    // const accumRhoMod = await compile('accumRho', accumRhoWGSL);
    const reduceNormMod = await compile('reduceNorm', reduceNormWGSL);
    const normalizeMod = await compile('normalize', normalizeWGSL);
    const extractSliceMod = await compile('extractSlice', extractSliceWGSL);

    // Pipelines
    initK_PL = await device.createComputePipelineAsync({
      layout: 'auto', compute: { module: initK_Mod, entryPoint: 'main' } });
    initElec_PL = await device.createComputePipelineAsync({
      layout: 'auto', compute: { module: initElec_Mod, entryPoint: 'main' } });
    precompPL = await device.createComputePipelineAsync({
      layout: 'auto', compute: { module: precompMod, entryPoint: 'main' } });
    fusedWU_PL = await device.createComputePipelineAsync({
      layout: 'auto', compute: { module: fusedWU_Mod, entryPoint: 'main' } });
    updateP_PL = await device.createComputePipelineAsync({
      layout: 'auto', compute: { module: updateP_Mod, entryPoint: 'main' } });
    // accumRhoPL = await device.createComputePipelineAsync({
    //   layout: 'auto', compute: { module: accumRhoMod, entryPoint: 'main' } });
    reduceNormPL = await device.createComputePipelineAsync({
      layout: 'auto', compute: { module: reduceNormMod, entryPoint: 'main' } });
    normalizePL = await device.createComputePipelineAsync({
      layout: 'auto', compute: { module: normalizeMod, entryPoint: 'main' } });
    extractSlicePL = await device.createComputePipelineAsync({
      layout: 'auto', compute: { module: extractSliceMod, entryPoint: 'main' } });
    console.log("All pipelines created");

    await runInit();
    createSimBindGroups();

    gpuReady = true;
    console.log(`molecule2 GPU initialized — ${NELEC} atoms/electrons`);
  } catch (e) {
    gpuError = e.message;
    console.error("GPU init failed:", e);
  }
}

async function runInit() {
  const zeros = new Float32Array(S3);
  device.queue.writeBuffer(K_buf, 0, zeros);

  // Init K: one dispatch per nucleus
  initK_BG = device.createBindGroup({ layout: initK_PL.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: paramsBuf } },
    { binding: 1, resource: { buffer: K_buf } },
    { binding: 2, resource: { buffer: nucCfgBuf } },
  ]});

  for (let n = 0; n < ALL_ATOMS; n++) {
    const a = atoms[n];
    const zn = _z_nuc[n] || a.Z;
    if (zn <= 0) continue; // skip atoms with no nuclear charge
    device.queue.writeBuffer(nucCfgBuf, 0, new Float32Array([a.i, a.j, a.k, zn]));
    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(initK_PL);
    cp.setBindGroup(0, initK_BG);
    cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
    cp.end();
    device.queue.submit([enc.finish()]);
  }

  // Init electrons — only for atoms with Z>0 (have electrons)
  let elecIdx = 0;
  for (let n = 0; n < ALL_ATOMS; n++) {
    const a = atoms[n];
    if (a.Z <= 0) continue; // skip bare protons
    const m = elecIdx++;
    const initR = window.USER_INIT_R || 0;
    // Use alternate init position if specified
    const ip = _initPos[n];
    const initI = ip ? ip[0] : a.i;
    const initJ = ip ? ip[1] : a.j;
    const initK = ip && ip[2] !== undefined ? ip[2] : (a.k || N2);
    device.queue.writeBuffer(elecCfgBuf, 0, new Float32Array([initI, initJ, initK, a.rc || 0, initR, 0, 0, 0]));

    const bg = device.createBindGroup({ layout: initElec_PL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: u_buf[m][0] } },
      { binding: 2, resource: { buffer: w_buf[m][0] } },
      { binding: 3, resource: { buffer: elecCfgBuf } },
      { binding: 4, resource: { buffer: domain_buf[m] } },
    ]});

    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(initElec_PL);
    cp.setBindGroup(0, bg);
    cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
    cp.end();
    device.queue.submit([enc.finish()]);
    console.log(`Electron ${m}: atom Z=${a.Z} at (${a.i},${a.j},${a.k})` +
      (ip ? ` init at (${initI},${initJ})` : ''));
  }
  await device.queue.onSubmittedWorkDone();
  console.log("runInit complete — " + NELEC + " electrons initialized");
}

function createSimBindGroups() {
  // Precompute bind groups — one per electron per ping-pong state
  if (NELEC <= 5) {
    for (let m = 0; m < NELEC; m++) {
      precompBG[m] = [];
      for (let c = 0; c < 2; c++) {
        const entries = [{ binding: 0, resource: { buffer: paramsBuf } }];
        for (let n = 0; n < NELEC; n++)
          entries.push({ binding: n + 1, resource: { buffer: u_buf[n][c] } });
        entries.push({ binding: NELEC + 1, resource: { buffer: c_buf } });
        entries.push({ binding: NELEC + 2, resource: { buffer: rho_buf } });
        entries.push({ binding: NELEC + 3, resource: { buffer: elecIdxBufs[m] } });
        precompBG[m][c] = device.createBindGroup({
          layout: precompPL.getBindGroupLayout(0), entries });
      }
    }
  } else {
    for (let m = 0; m < NELEC; m++) {
      precompBG[m] = [];
      for (let c = 0; c < 2; c++) {
        precompBG[m][c] = device.createBindGroup({
          layout: precompPL.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: packedU_buf } },
            { binding: 2, resource: { buffer: c_buf } },
            { binding: 3, resource: { buffer: rho_buf } },
            { binding: 4, resource: { buffer: elecIdxBufs[m] } },
          ]});
      }
    }
  }

  // Fused w+u bind groups
  for (let m = 0; m < NELEC; m++) {
    fusedWU_BG[m] = [];
    fusedWU_mask_BG[m] = [];
    for (let c = 0; c < 2; c++) {
      fusedWU_BG[m][c] = device.createBindGroup({
        layout: fusedWU_PL.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: paramsBuf } },
          { binding: 1, resource: { buffer: u_buf[m][c] } },
          { binding: 2, resource: { buffer: c_buf } },
          { binding: 3, resource: { buffer: w_buf[m][c] } },
          { binding: 4, resource: { buffer: K_buf } },
          { binding: 5, resource: { buffer: P_buf[m][0] } },
          { binding: 6, resource: { buffer: w_buf[m][1-c] } },
          { binding: 7, resource: { buffer: u_buf[m][1-c] } },
        ]});
      fusedWU_mask_BG[m][c] = device.createBindGroup({
        layout: fusedWU_PL.getBindGroupLayout(1), entries: [
          { binding: 0, resource: { buffer: rcMask_buf[m] } },
        ]});
    }
  }

  // P update bind groups
  for (let m = 0; m < NELEC; m++) {
    updateP_BG[m] = [];
    for (let ping = 0; ping < 2; ping++) {
      updateP_BG[m][ping] = device.createBindGroup({
        layout: updateP_PL.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: paramsBuf } },
          { binding: 1, resource: { buffer: P_buf[m][ping] } },
          { binding: 2, resource: { buffer: rho_buf } },
          { binding: 3, resource: { buffer: P_buf[m][1 - ping] } },
        ]});
    }
  }

  // accumRho and Ptot bind groups disabled for now

  // Norm + normalize
  for (let m = 0; m < NELEC; m++) {
    reduceNormBG[m] = [];
    normalizeBG[m] = [];
    for (let c = 0; c < 2; c++) {
      reduceNormBG[m][c] = device.createBindGroup({
        layout: reduceNormPL.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: paramsBuf } },
          { binding: 1, resource: { buffer: u_buf[m][1-c] } },
          { binding: 2, resource: { buffer: normPartialBuf[m] } },
        ]});
      normalizeBG[m][c] = device.createBindGroup({
        layout: normalizePL.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: paramsBuf } },
          { binding: 1, resource: { buffer: u_buf[m][1-c] } },
          { binding: 2, resource: { buffer: normPartialBuf[m] } },
        ]});
    }
  }

  // Extract slice
  for (let m = 0; m < MAX_VIS; m++) {
    extractSliceBG[m] = [];
    for (let c = 0; c < 2; c++) {
      extractSliceBG[m][c] = device.createBindGroup({
        layout: extractSlicePL.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: paramsBuf } },
          { binding: 1, resource: { buffer: u_buf[m][c] } },
          { binding: 2, resource: { buffer: w_buf[m][c] } },
          { binding: 3, resource: { buffer: P_buf[m][0] } },
          { binding: 4, resource: { buffer: K_buf } },
          { binding: 5, resource: { buffer: sliceBuf } },
          { binding: 6, resource: { buffer: sliceCfgBuf } },
        ]});
    }
  }
}

let P_ping = 0;

function doSteps(nSteps) {
  if (!gpuReady) return;
  const enc = device.createCommandEncoder();

  for (let s = 0; s < nSteps; s++) {
    // Pack u for NELEC > 5
    if (NELEC > 5) {
      for (let m = 0; m < NELEC; m++)
        enc.copyBufferToBuffer(u_buf[m][cur], 0, packedU_buf, m * S3 * 4, S3 * 4);
    }

    for (let m = 0; m < NELEC; m++) {
      // Precompute c and rho_other for electron m
      {
        const cp = enc.beginComputePass();
        cp.setPipeline(precompPL);
        cp.setBindGroup(0, precompBG[m][cur]);
        cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
        cp.end();
      }
      // Fused w+u
      {
        const cp = enc.beginComputePass();
        cp.setPipeline(fusedWU_PL);
        cp.setBindGroup(0, fusedWU_BG[m][cur]);
        cp.setBindGroup(1, fusedWU_mask_BG[m][cur]);
        cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
        cp.end();
      }
      // P update
      {
        const cp = enc.beginComputePass();
        cp.setPipeline(updateP_PL);
        cp.setBindGroup(0, updateP_BG[m][P_ping]);
        cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
        cp.end();
      }
    }
    P_ping = 1 - P_ping;

    // Normalize all electrons
    if (s % NORM_INTERVAL === 0) {
      for (let m = 0; m < NELEC; m++) {
        const rp = enc.beginComputePass();
        rp.setPipeline(reduceNormPL);
        rp.setBindGroup(0, reduceNormBG[m][cur]);
        rp.dispatchWorkgroups(N_REDUCE_WG);
        rp.end();

        const np = enc.beginComputePass();
        np.setPipeline(normalizePL);
        np.setBindGroup(0, normalizeBG[m][cur]);
        np.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
        np.end();
      }
    }

    // Ptot computation disabled for now (done in dynamics addon if needed)

    cur = 1 - cur;
    stepCount++;
  }

  device.queue.submit([enc.finish()]);
}

// Reinitialize with new atom positions
async function reinitSim(newAtoms) {
  if (!gpuReady) return;

  if (newAtoms) {
    for (let m = 0; m < atoms.length; m++) {
      if (newAtoms[m]) Object.assign(atoms[m], newAtoms[m]);
    }
  }

  // Recompute V_KK
  V_KK = 0;
  for (let a = 0; a < atoms.length; a++)
    for (let b = a + 1; b < atoms.length; b++) {
      const di = (atoms[a].i - atoms[b].i) * h;
      const dj = (atoms[a].j - atoms[b].j) * h;
      const dk = (atoms[a].k - atoms[b].k) * h;
      V_KK += atoms[a].Z * atoms[b].Z / Math.sqrt(di*di + dj*dj + dk*dk + h2);
    }

  const zeros = new Float32Array(S3);
  for (let m = 0; m < NELEC; m++) {
    for (let c = 0; c < 2; c++) {
      device.queue.writeBuffer(u_buf[m][c], 0, zeros);
      device.queue.writeBuffer(w_buf[m][c], 0, zeros);
      device.queue.writeBuffer(P_buf[m][c], 0, zeros);
    }
    const maskData = computeRcMask(m);
    device.queue.writeBuffer(rcMask_buf[m], 0, maskData);
    const domainData = computeVoronoiDomain(m);
    device.queue.writeBuffer(domain_buf[m], 0, domainData);
  }
  device.queue.writeBuffer(rhoTotal_buf, 0, zeros);
  device.queue.writeBuffer(Ptot_buf[0], 0, zeros);
  device.queue.writeBuffer(Ptot_buf[1], 0, zeros);
  Ptot_ping = 0;

  await runInit();
  cur = 0;
  P_ping = 0;
  stepCount = 0;
  E_T = 0; E_eK = 0; E_ee = 0; E_pot = 0; E_tot = 0;
  sliceData = null;
  energyReadBufs = null;
  console.log(`Reinit: V_KK=${V_KK.toFixed(4)}`);
}

// Full 3D CPU readback for energy
let energyReadBufs = null;
function initEnergyReadback() {
  const bs = S3 * 4;
  energyReadBufs = {
    u: [], w: [], P: [],
    K_read: device.createBuffer({ size: bs,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
  };
  for (let m = 0; m < NELEC; m++) {
    energyReadBufs.u[m] = device.createBuffer({ size: bs,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    energyReadBufs.w[m] = device.createBuffer({ size: bs,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    energyReadBufs.P[m] = device.createBuffer({ size: bs,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }
}

async function computeEnergy() {
  if (!gpuReady) return;
  if (!energyReadBufs) initEnergyReadback();

  const bs = S3 * 4;
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(K_buf, 0, energyReadBufs.K_read, 0, bs);
  for (let m = 0; m < NELEC; m++) {
    enc.copyBufferToBuffer(u_buf[m][cur], 0, energyReadBufs.u[m], 0, bs);
    enc.copyBufferToBuffer(w_buf[m][cur], 0, energyReadBufs.w[m], 0, bs);
    enc.copyBufferToBuffer(P_buf[m][0], 0, energyReadBufs.P[m], 0, bs);
  }
  device.queue.submit([enc.finish()]);

  const maps = [energyReadBufs.K_read.mapAsync(GPUMapMode.READ)];
  for (let m = 0; m < NELEC; m++) {
    maps.push(energyReadBufs.u[m].mapAsync(GPUMapMode.READ));
    maps.push(energyReadBufs.w[m].mapAsync(GPUMapMode.READ));
    maps.push(energyReadBufs.P[m].mapAsync(GPUMapMode.READ));
  }
  await Promise.all(maps);

  const K = new Float32Array(energyReadBufs.K_read.getMappedRange());
  const uArr = [], wArr = [], PArr = [];
  for (let m = 0; m < NELEC; m++) {
    uArr[m] = new Float32Array(energyReadBufs.u[m].getMappedRange());
    wArr[m] = new Float32Array(energyReadBufs.w[m].getMappedRange());
    PArr[m] = new Float32Array(energyReadBufs.P[m].getMappedRange());
  }

  E_T = 0; E_eK = 0; E_ee = 0; E_pot = 0;
  for (let i = 1; i < NN; i++) {
    for (let j = 1; j < NN; j++) {
      for (let k = 1; k < NN; k++) {
        const id = i * S2 + j * S + k;
        for (let m = 0; m < NELEC; m++) {
          const v = uArr[m][id];
          const rho = v * v;
          const wv = wArr[m][id];
          if (wv > 0.1) {
            const gx = uArr[m][id + S2] - v;
            const gy = uArr[m][id + S] - v;
            const gz = uArr[m][id + 1] - v;
            E_T += 0.5 * (gx*gx + gy*gy + gz*gz) * h;
          }
          E_eK += -K[id] * rho * h3;
          E_ee += PArr[m][id] * rho * h3;
          E_pot += (PArr[m][id] - K[id]) * rho * h3;
        }
      }
    }
  }
  E_tot = E_T + E_eK + E_ee + V_KK;

  energyReadBufs.K_read.unmap();
  for (let m = 0; m < NELEC; m++) {
    energyReadBufs.u[m].unmap();
    energyReadBufs.w[m].unmap();
    energyReadBufs.P[m].unmap();
  }
}

async function extractSlice() {
  if (!gpuReady) return;
  const sliceWGs = Math.ceil(S * S / WG_SIZE);
  for (let m = 0; m < MAX_VIS; m++) {
    const offset = m * 3;
    device.queue.writeBuffer(sliceCfgBuf, 0, new Uint32Array([offset, 0, 0, 0]));
    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(extractSlicePL);
    cp.setBindGroup(0, extractSliceBG[m][cur]);
    cp.dispatchWorkgroups(sliceWGs);
    cp.end();
    device.queue.submit([enc.finish()]);
  }
  const sliceBytes = S * S * SLICE_STRIDE * 4;
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(sliceBuf, 0, sliceReadBuf, 0, sliceBytes);
  device.queue.submit([enc.finish()]);
  await sliceReadBuf.mapAsync(GPUMapMode.READ);
  sliceData = new Float32Array(sliceReadBuf.getMappedRange().slice(0));
  sliceReadBuf.unmap();
}


// --- p5.js Integration ---
const STEPS_PER_FRAME = window.USER_SPF || (NN <= 100 ? 10 : 2);
let initDone = false;
let readbackPending = false;
let lastLogStep = -1;

const ELEC_COLORS = [
  [255, 50, 50], [50, 50, 255], [50, 200, 50], [200, 50, 200],
];

window.setup = function() {
  createCanvas(700, 730);
  initGPU().then(() => {
    initDone = true;
    console.log("GPU init complete");
  }).catch(e => {
    gpuError = e.message;
    console.error("GPU init error:", e);
  });
};

window.draw = function() {
  background(220);
  if (gpuError) {
    fill(255, 0, 0); textSize(14);
    text("GPU Error: " + gpuError, 10, 200);
    return;
  }
  if (!initDone || !gpuReady) {
    fill(0); text("Initializing GPU...", 10, 200);
    return;
  }

  if (stepCount < MAX_STEPS) {
    doSteps(STEPS_PER_FRAME);
  }

  if (!readbackPending) {
    readbackPending = true;
    (async function() {
      try {
        await device.queue.onSubmittedWorkDone();
        await computeEnergy();
        await extractSlice();
      } catch(e) {
        console.error("Readback error:", e);
        gpuError = "Readback: " + e.message;
      }
      readbackPending = false;
    })();
  }

  // Draw density
  noStroke();
  const imgW = 350, imgH = 350;
  const sx = imgW / S, sy = imgH / S;

  if (sliceData) {
    for (let i = 1; i < NN; i++) {
      for (let j = 1; j < NN; j++) {
        const idx = (j * S + i) * SLICE_STRIDE;
        for (let m = 0; m < MAX_VIS; m++) {
          const off = idx + m * 3;
          const wu = sliceData[off] * sliceData[off + 1];
          const a = Math.min(255, 500 * Math.abs(wu));
          if (a > 2) {
            const cl = ELEC_COLORS[m % ELEC_COLORS.length];
            fill(cl[0], cl[1], cl[2], a);
            square(i * sx, j * sy, Math.max(sx, 1));
          }
        }
      }
    }

    // r_c circles and nuclei
    noFill();
    for (const a of atoms) {
      const rc = a.rc || 0;
      if (rc > 0) {
        stroke(255, 200, 0, 150); strokeWeight(1.5);
        circle(a.i * sx, a.j * sy, rc / h * sx * 2);
      }
    }
    fill(0); noStroke();
    for (const a of atoms) circle(a.i * sx, a.j * sy, 8);

    // Force arrows from dynamics addon
    if (window._dyn && window._dyn.enabled()) {
      var eForces = window._dyn.elecForces();
      var nForces = window._dyn.nucRepForces();
      var tForces = window._dyn.forces();
      var arrowScale = 500;
      for (var fa = 0; fa < atoms.length; fa++) {
        var ax = atoms[fa].i * sx, ay = atoms[fa].j * sy;
        // Total force (green)
        var tx = tForces[fa][0], ty = tForces[fa][1];
        if (Math.sqrt(tx*tx + ty*ty) > 1e-8) {
          stroke(0, 255, 0); strokeWeight(2);
          line(ax, ay, ax + tx * arrowScale, ay + ty * arrowScale);
        }
      }
      noStroke();
      fill(0, 200, 200); textSize(11);
      text("Dynamics ON  nucStep=" + window._dyn.steps(), imgW + 15, 186 + atoms.length * 14 + 14);
      // Log force magnitudes
      for (var fa = 0; fa < atoms.length; fa++) {
        var ef = eForces[fa], nf = window._dyn.nucRepForces()[fa], tf = window._dyn.forces()[fa];
        text("#" + fa + " elec=" + Math.sqrt(ef[0]*ef[0]+ef[1]*ef[1]+ef[2]*ef[2]).toFixed(4) +
          " nuc=" + Math.sqrt(nf[0]*nf[0]+nf[1]*nf[1]+nf[2]*nf[2]).toFixed(4) +
          " tot=" + Math.sqrt(tf[0]*tf[0]+tf[1]*tf[1]+tf[2]*tf[2]).toFixed(4),
          imgW + 15, 186 + atoms.length * 14 + 28 + fa * 14);
      }
    }

    // Display inter-nuclear distances and angles
    if (atoms.length >= 2) {
      fill(255, 200, 0); textSize(11);
      var dIdx = 0;
      for (var da = 0; da < atoms.length; da++) {
        for (var db = da + 1; db < atoms.length; db++) {
          var ddi = (atoms[da].i - atoms[db].i) * h;
          var ddj = (atoms[da].j - atoms[db].j) * h;
          var ddk = (atoms[da].k - atoms[db].k) * h;
          var dist = Math.sqrt(ddi*ddi + ddj*ddj + ddk*ddk);
          text("d(" + da + "-" + db + ") = " + dist.toFixed(3) + " au", imgW + 15, 186 + atoms.length * 14 + 28 + dIdx * 14);
          dIdx++;
        }
      }
      // Display angle for 3-atom systems (angle at atom 0)
      if (atoms.length >= 3) {
        var v1 = [(atoms[1].i-atoms[0].i)*h, (atoms[1].j-atoms[0].j)*h, (atoms[1].k-atoms[0].k)*h];
        var v2 = [(atoms[2].i-atoms[0].i)*h, (atoms[2].j-atoms[0].j)*h, (atoms[2].k-atoms[0].k)*h];
        var dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
        var m1 = Math.sqrt(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2]);
        var m2 = Math.sqrt(v2[0]*v2[0]+v2[1]*v2[1]+v2[2]*v2[2]);
        var ang = Math.acos(Math.max(-1, Math.min(1, dot/(m1*m2)))) * 180 / Math.PI;
        text("angle(1-0-2) = " + ang.toFixed(1) + "\u00B0", imgW + 15, 186 + atoms.length * 14 + 28 + dIdx * 14);
      }
    }

    // Line plots
    const plotW = imgW;
    const plotH_w = 80, plotH_u = 80, plotH_P = 80;
    const gap = 15;
    const baseW = imgH + gap + plotH_w;
    const baseU = baseW + gap + plotH_u;
    const baseP = baseU + gap + plotH_P;

    for (let i = 0; i < S; i++) {
      const idx = (N2 * S + i) * SLICE_STRIDE;
      const x = i * (plotW / S);
      for (let m = 0; m < MAX_VIS; m++) {
        const off = idx + m * 3;
        const uv = sliceData[off], wv = sliceData[off + 1], Pv = sliceData[off + 2];
        const cl = ELEC_COLORS[m % ELEC_COLORS.length];

        fill(cl[0], cl[1], cl[2]); noStroke();
        ellipse(x, baseW - plotH_w * wv, 3);
        ellipse(x, baseU - plotH_u * uv * wv, 3);
        fill(cl[0], cl[1], cl[2], 180);
        ellipse(x, baseP - plotH_P * Pv, 2);
      }
      const Kv = sliceData[idx + MAX_VIS * 3];
      fill(0, 180, 0);
      ellipse(x, baseP - 30 * Kv, 3);
    }
  }

  // Info
  const rx = imgW + 15;
  fill(0); textSize(13);
  text("Molecule RealQM GPU", rx, 20);
  text(NN + "\u00B3 / " + screenAu + " au", rx, 38);
  text(NELEC + " atoms/electrons", rx, 56);
  const rcStr = atoms.some(a => a.rc > 0) ? " r_c=" + atoms[0].rc : "";
  text("step " + stepCount + rcStr, rx, 74);

  textSize(12);
  text("T = " + E_T.toFixed(4), rx, 96);
  text("V_eK = " + E_eK.toFixed(4), rx, 112);
  text("V_ee = " + E_ee.toFixed(4), rx, 128);
  text("V_KK = " + V_KK.toFixed(4), rx, 144);
  textSize(13);
  text("E = " + E_tot.toFixed(4), rx, 166);

  textSize(10);
  for (let m = 0; m < atoms.length; m++) {
    const a = atoms[m];
    text(`#${m}: Z=${a.Z} (${a.i},${a.j},${a.k})`, rx, 186 + m * 14);
  }

  if (stepCount > 0 && stepCount % 100 === 0 && stepCount !== lastLogStep) {
    lastLogStep = stepCount;
    console.log("step=" + stepCount + " T=" + E_T.toFixed(4) + " V_eK=" + E_eK.toFixed(4) +
      " V_ee=" + E_ee.toFixed(4) + " V_KK=" + V_KK.toFixed(4) + " E=" + E_tot.toFixed(4));
  }
};

// Expose internals for dynamics addon
window._m2 = {
  atoms: atoms,
  get device() { return device; }, get cur() { return cur; },
  get stepCount() { return stepCount; }, set stepCount(v) { stepCount = v; }, get MAX_STEPS() { return MAX_STEPS; },
  get gpuReady() { return gpuReady; }, get V_KK() { return V_KK; }, set V_KK(v) { V_KK = v; },
  get E_tot() { return E_tot; }, get E_T() { return E_T; }, get E_eK() { return E_eK; }, get E_ee() { return E_ee; },
  get u_buf() { return u_buf; }, get w_buf() { return w_buf; }, get P_buf() { return P_buf; }, get K_buf() { return K_buf; }, get Ptot_buf() { return Ptot_buf; },
  get nucCfgBuf() { return nucCfgBuf; }, get initK_PL() { return initK_PL; }, get initK_BG() { return initK_BG; },
  get energyReadBufs() { return energyReadBufs; },
  initEnergyReadback: function() { initEnergyReadback(); },
  computeRcMask: computeRcMask,
  computeVoronoiDomain: computeVoronoiDomain,
  get rcMask_buf() { return rcMask_buf; },
  get domain_buf() { return domain_buf; },
  NELEC: NELEC, NN: NN, S: S, S2: S2, S3: S3,
  h: h, h2: h2, h3: h3,
  DISPATCH_X: DISPATCH_X, DISPATCH_Y: DISPATCH_Y,
};

