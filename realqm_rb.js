// realqm_rb.js — Red-Black Gauss-Seidel GPU solver
// Supports NELEC >= 2 electron domains with arbitrary nuclear positions
// In-place updates with red-black coloring for Gauss-Seidel convergence
// Parameters match p5: c=0.1, w diffusion=2, advection=10, K=Z/sqrt(r²+1.2h²)

"use strict";

const NN = window.USER_NN || 100;
const screenAu = window.USER_SCREEN || 8;
const D_CELLS = (window.USER_D !== undefined) ? window.USER_D : 0;
const MAX_STEPS = window.USER_STEPS || 10000;
const NELEC = window.USER_NELEC || 2;
const NORM_INTERVAL = 5;

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

const WG_SIZE = 256;
const INTERIOR = (NN - 1) * (NN - 1) * (NN - 1);
const THREADS_X = 512;
const DISPATCH_Y = Math.ceil(Math.ceil(INTERIOR / WG_SIZE) / THREADS_X);
const DISPATCH_X = Math.min(THREADS_X, Math.ceil(INTERIOR / WG_SIZE));
const REDUCE_WG = 128;
const N_REDUCE_WG = Math.min(256, Math.ceil(INTERIOR / REDUCE_WG));
const CELLS_PER_ROW = DISPATCH_X * WG_SIZE;

console.log(`RealQM RB: NN=${NN}, h=${h.toFixed(4)}, dt=${dt.toExponential(3)}, NELEC=${NELEC}`);

// Nuclear positions: prefer window.USER_NUCLEI, fall back to 2-nuclei D_CELLS setup
// Each nucleus: { i, j, k, Z, rc, ne }  (rc, ne optional)
let nuclei;
if (window.USER_NUCLEI) {
  nuclei = window.USER_NUCLEI;
} else {
  nuclei = [
    { i: N2 - Math.round(D_CELLS / 2), j: N2, k: N2, Z: 1, rc: 0, ne: 1 },
    { i: N2 + Math.round(D_CELLS / 2), j: N2, k: N2, Z: 1, rc: 0, ne: 1 },
  ];
}

// Ensure each nucleus has ne (electron count, defaults to 1)
for (const nuc of nuclei) {
  if (nuc.ne === undefined) nuc.ne = 1;
  if (nuc.rc === undefined) nuc.rc = 0;
}

// Nuclear repulsion energy (all pairs)
let V_KK = 0;
if (!window.NO_NUC_REPULSION) {
  for (let a = 0; a < nuclei.length; a++) {
    for (let b = a + 1; b < nuclei.length; b++) {
      const dx = (nuclei[a].i - nuclei[b].i) * h;
      const dy = (nuclei[a].j - nuclei[b].j) * h;
      const dz = (nuclei[a].k - nuclei[b].k) * h;
      const r = Math.sqrt(dx*dx + dy*dy + dz*dz);
      if (r > 0) V_KK += nuclei[a].Z * nuclei[b].Z / Math.sqrt(r*r + h2);
    }
  }
}

// Map each electron index m -> which nucleus it belongs to.
// Electrons are assigned to nuclei in order, each nucleus getting ne[n] electrons.
// e.g. ne=[2,1,1]: electrons 0,1 -> nucleus 0; electron 2 -> nucleus 1; electron 3 -> nucleus 2
const electronNucleus = [];  // electronNucleus[m] = nucleus index for electron m
{
  let idx = 0;
  for (let n = 0; n < nuclei.length; n++) {
    for (let e = 0; e < nuclei[n].ne; e++) {
      electronNucleus[idx++] = n;
    }
  }
}
console.log("electronNucleus:", electronNucleus);

// --- WGSL Shaders ---

const paramStructWGSL = `
struct P {
  NN: u32, S: u32, S2: u32, S3: u32,
  N2: u32, dt: f32, half_d: f32, TWO_PI: f32,
  h: f32, h2: f32, inv_h: f32, inv_h2: f32,
  h3: f32, _p0: u32, _p1: u32, _p2: u32,
}`;

const cellIdxWGSL = `
fn cellIdx(gid: vec3<u32>) -> u32 {
  return gid.x + gid.y * ${CELLS_PER_ROW}u;
}`;

// Init shader: accumulates K from one nucleus, sets u and w for one electron domain
// Domain assignment uses Voronoi (nearest nucleus) based on nucI/nucJ/nucK.
// Multi-rival: cell is in own Voronoi cell iff it's closer to own than to every
// other nucleus. rivCount = number of rival entries in rivals[] (0..MAX_RIVALS-1).
const MAX_RIVALS = 8;
const initWGSL = `
${paramStructWGSL}
struct InitCfg {
  nucI: f32, nucJ: f32, nucK: f32, nucZ: f32,
  rc: f32, _pa: f32, _pb: f32, _pc: f32,
  splitDir: u32, writeK: u32, rivCount: u32, _pd: u32,
  rivals: array<vec4<f32>, ${MAX_RIVALS}>,  // (i, j, k, _)
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> K: array<f32>;
@group(0) @binding(2) var<storage, read_write> u: array<f32>;
@group(0) @binding(3) var<storage, read_write> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> Pot: array<f32>;
@group(0) @binding(5) var<uniform> cfg: InitCfg;

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

  // dx,dy,dz in au; r2 in au^2
  let dx = (f32(i) - cfg.nucI) * p.h;
  let dy = (f32(j) - cfg.nucJ) * p.h;
  let dz = (f32(k) - cfg.nucK) * p.h;
  let r2 = dx*dx + dy*dy + dz*dz;
  let r_soft = sqrt(r2 + 1.2 * p.h2);  // 1.2h² smoothing
  let r_au = sqrt(r2);  // distance in au

  // K accumulates Z/r_soft for each nucleus (only when writeK flag is set)
  if (cfg.writeK != 0u) {
    K[id] += cfg.nucZ / r_soft;
  }

  // Voronoi: cell must be closer to own nucleus than to EVERY rival.
  var inVoronoi: bool = true;
  for (var rv: u32 = 0u; rv < cfg.rivCount; rv = rv + 1u) {
    let riv = cfg.rivals[rv];
    let drx = (f32(i) - riv.x) * p.h;
    let dry = (f32(j) - riv.y) * p.h;
    let drz = (f32(k) - riv.z) * p.h;
    let riv2 = drx*drx + dry*dry + drz*drz;
    if (r2 > riv2) { inVoronoi = false; }
  }

  // For rc > 0: restrict domain to sphere of radius rc (in au)
  let inRcSphere = cfg.rc <= 0.0 || r_au < cfg.rc;

  // splitDir within Voronoi cell:
  //   0 = left half  (i < N2-1, 3-cell gap at midplane)
  //   1 = right half (i > N2+1)
  //   2 = full Voronoi cell (single electron at this nucleus)
  let inSplit = select(
    select(i > p.N2 + 1u, i < p.N2 - 1u, cfg.splitDir == 0u),
    true,
    cfg.splitDir == 2u
  );

  // Spherical cutoff at 3 au for u/w initialization
  let inSphere = r_au < 3.0;

  // w starts as a compact indicator inside the Voronoi ∩ rc ∩ 3au-sphere region.
  // u uses the same region but without the rc limit, so its exponential tail extends
  // beyond the initial w front. This gives u>0 just outside w=1, driving c>0 at the
  // front and letting advection push w outward from the first step.
  let wInDomain = inVoronoi && inSplit && inRcSphere && inSphere;
  let uInDomain = inVoronoi && inSplit && inSphere;

  if (wInDomain) { w[id] = 1.0; }
  if (uInDomain) {
    u[id] = exp(-0.5 * cfg.nucZ * r_soft);
  }

  // Initial P = 0.5/r_soft (rough guess for electron repulsion potential)
  Pot[id] = 0.5 / r_soft;
}
`;

// Compute rho_total = sum_m u[m]^2 into a scratch buffer
// Generated at runtime per NELEC (WGSL code sees actual number of u bindings)
function makeRhoTotalWGSL(nelec) {
  const bindings = [];
  for (let m = 0; m < nelec; m++) {
    bindings.push(`@group(0) @binding(${m + 1}) var<storage, read> u${m}: array<f32>;`);
  }
  const sumTerms = Array.from({length: nelec}, (_, m) => `u${m}[id]*u${m}[id]`).join(' + ');
  return `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
${bindings.join('\n')}
@group(0) @binding(${nelec + 1}) var<storage, read_write> rho_total: array<f32>;

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
  rho_total[id] = ${sumTerms};
}
`;
}

// Compute u_nearest[m] = max over n!=m of u[n] for front-tracking competition.
// "Nearest" = most competing electron at this cell (largest u), not the weakest.
// For nelec=2 this is identical to min (single term); for nelec>=3 max is required
// so the boundary feels its real neighbor, not a distant electron with u≈0.
// Generated at runtime per NELEC; separate shader per m index. Only declare bindings
// for the OTHER electrons — declaring the self buffer here would be stripped by
// layout:'auto' (unused), causing bind-group validation errors.
function makeUNearestWGSL(nelec, m) {
  const others = Array.from({length: nelec}, (_, n) => n).filter(n => n !== m);
  const bindings = others.map((n, i) => `@group(0) @binding(${i + 1}) var<storage, read> u${n}: array<f32>;`);
  let minExpr = `u${others[0]}[id]`;
  for (let i = 1; i < others.length; i++) {
    minExpr = `max(${minExpr}, u${others[i]}[id])`;
  }
  return `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
${bindings.join('\n')}
@group(0) @binding(${others.length + 1}) var<storage, read_write> u_nearest: array<f32>;

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
  u_nearest[id] = ${minExpr};
}
`;
}

// Red-black fused w+u update — IN-PLACE, only updates cells of matching color
// u_nearest holds min(u[n], n!=m) for front tracking competition
const fusedWU_RB_WGSL = `
${paramStructWGSL}
struct ColorCfg { color: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> u_self: array<f32>;
@group(0) @binding(2) var<storage, read> u_nearest: array<f32>;
@group(0) @binding(3) var<storage, read_write> w_self: array<f32>;
@group(0) @binding(4) var<storage, read> Kbuf: array<f32>;
@group(0) @binding(5) var<storage, read> Pot: array<f32>;
@group(0) @binding(6) var<uniform> ccfg: ColorCfg;
@group(0) @binding(7) var<storage, read> rcMask: array<f32>;

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

  // Red-black: skip wrong color
  if ((i + j + k) % 2u != ccfg.color) { return; }

  let c = 0.1 * (u_self[id] - u_nearest[id]);  // p5: 0.1 coefficient
  let wc = w_self[id];

  // Front tracking: diffusion=2, advection=10 (matching p5, validated on H2/He)
  let lap_w = w_self[id + p.S2] + w_self[id - p.S2]
            + w_self[id + p.S]  + w_self[id - p.S]
            + w_self[id + 1u]   + w_self[id - 1u] - 6.0 * wc;
  let gwx = (w_self[id + p.S2] - w_self[id - p.S2]) * p.inv_h;
  let gwy = (w_self[id + p.S]  - w_self[id - p.S])  * p.inv_h;
  let gwz = (w_self[id + 1u]   - w_self[id - 1u])   * p.inv_h;
  let grad_w = sqrt(gwx*gwx + gwy*gwy + gwz*gwz);
  w_self[id] = wc + 0.5 * p.dt * abs(c) * lap_w * p.inv_h2 + 10.0 * p.dt * c * grad_w;

  // u update: w-weighted Laplacian + potential (reads UPDATED w via w_self)
  let wc2 = w_self[id];
  let uc = u_self[id];
  let flux_xp = (u_self[id + p.S2] - uc) * (w_self[id + p.S2] + wc2) * 0.5;
  let flux_xm = (uc - u_self[id - p.S2]) * (wc2 + w_self[id - p.S2]) * 0.5;
  let flux_yp = (u_self[id + p.S] - uc) * (w_self[id + p.S] + wc2) * 0.5;
  let flux_ym = (uc - u_self[id - p.S]) * (wc2 + w_self[id - p.S]) * 0.5;
  let flux_zp = (u_self[id + 1u] - uc) * (w_self[id + 1u] + wc2) * 0.5;
  let flux_zm = (uc - u_self[id - 1u]) * (wc2 + w_self[id - 1u]) * 0.5;
  let wlap = (flux_xp - flux_xm) + (flux_yp - flux_ym) + (flux_zp - flux_zm);

  let u_new = uc + p.half_d * wlap + p.dt * (Kbuf[id] - 2.0 * Pot[id]) * uc * wc2;
  u_self[id] = u_new * rcMask[id];  // zero inside r_c
  w_self[id] = w_self[id] * rcMask[id];  // also zero w inside r_c
}
`;

// Red-black Poisson update for ONE electron — IN-PLACE.
// P[m] is driven by rho_total − selfFrac·u_self²: for singly-occupied orbitals
// (selfN ≤ 1) we fully remove self-interaction; for multi-occupancy (selfN > 1)
// we keep (n−1)/n of the self-density so the real intra-orbital pair repulsion
// stays in the potential. selfN = ∫u_self² is read from the norm-target uniform.
const updateP_RB_WGSL = `
${paramStructWGSL}
struct ColorCfg { color: u32, _p0: u32, _p1: u32, _p2: u32 }
struct NormCfg { tgt: f32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> Pm: array<f32>;
@group(0) @binding(2) var<storage, read> u_self: array<f32>;
@group(0) @binding(3) var<storage, read> rho_total: array<f32>;
@group(0) @binding(4) var<uniform> ccfg: ColorCfg;
@group(0) @binding(5) var<uniform> ncfg: NormCfg;

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

  if ((i + j + k) % 2u != ccfg.color) { return; }

  let Pc = Pm[id];
  let lap_P = Pm[id + p.S2] + Pm[id - p.S2]
            + Pm[id + p.S]  + Pm[id - p.S]
            + Pm[id + 1u]   + Pm[id - 1u] - 6.0 * Pc;
  let us = u_self[id];
  let selfN = ncfg.tgt;
  let selfFrac = select(1.0, 1.0 / selfN, selfN > 1.0);
  let rho_others = max(rho_total[id] - selfFrac * us * us, 0.0);
  Pm[id] = Pc + p.dt * (lap_P * p.inv_h2 + p.TWO_PI * rho_others);
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

// Normalize — scales u so integral(u^2 dV) = ne (electron count for this domain)
// normTarget is passed via a small uniform buffer containing target norm value
const normalizeWGSL = `
${paramStructWGSL}
struct NormCfg { tgt: f32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read> normBuf: array<f32>;
@group(0) @binding(3) var<uniform> ncfg: NormCfg;

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
  // norm currently holds integral(u^2 dV); scale so it equals ncfg.tgt
  if (norm > 0.0) { u[id] *= sqrt(ncfg.tgt / norm); }
}
`;

// Slice layout: NELEC*3 + 1 floats per pixel
// [m*3+0] = u[m], [m*3+1] = w[m], [m*3+2] = P[m], [NELEC*3] = K
const SLICE_STRIDE = NELEC * 3 + 1;

// Extract 2D slice for one electron (offset = m*3 in the per-pixel layout)
const extractSliceWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> u_el: array<f32>;
@group(0) @binding(2) var<storage, read> w_el: array<f32>;
@group(0) @binding(3) var<storage, read> P_el: array<f32>;
@group(0) @binding(4) var<storage, read> Kbuf: array<f32>;
@group(0) @binding(5) var<storage, read_write> slice: array<f32>;
struct SliceCfg { offset: u32, stride: u32, nelec: u32, _p0: u32 }
@group(0) @binding(6) var<uniform> scfg: SliceCfg;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let N1 = p.NN + 1u;
  let tot = N1 * N1;
  if (gid.x >= tot) { return; }
  let j = gid.x / N1;
  let i = gid.x % N1;
  let id3 = i * p.S2 + j * p.S + p.N2;
  let base = gid.x * scfg.stride + scfg.offset;
  slice[base]      = u_el[id3];
  slice[base + 1u] = w_el[id3];
  slice[base + 2u] = P_el[id3];
  // K slot is at offset NELEC*3 (last slot), written by electron 0 pass
  if (scfg.offset == 0u) {
    slice[gid.x * scfg.stride + scfg.stride - 1u] = Kbuf[id3];
  }
}
`;

// --- GPU State ---
let device, gpuReady = false, gpuError = null;
let paramsBuf, initCfgBuf, colorBuf = [];
let K_buf;
let u_buf = [], w_buf = [], P_buf = [], rcMask_buf = [];
let normPartialBuf = [], normCfgBuf = [];
let rhoTotalBuf;
let uNearestBuf = [];  // uNearestBuf[m] = min(u[n], n!=m)
let sliceBuf, sliceReadBuf, sliceCfgBuf;

let initPL, fusedWU_RB_PL, updateP_RB_PL;
let reduceNormPL, normalizePL, extractSlicePL;
let rhoTotalPL, uNearestPL = [];

let E_T = 0, E_eK = 0, E_ee = 0, E_tot = 0;
let stepCount = 0;
let sliceData = null;

async function initGPU() {
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) { gpuError = "No WebGPU adapter"; return; }
    const maxBuf = Math.min(adapter.limits.maxStorageBufferBindingSize, S3 * 4);
    device = await adapter.requestDevice({
      requiredLimits: { maxStorageBufferBindingSize: maxBuf, maxBufferSize: maxBuf }
    });
    device.lost.then(info => { gpuError = "GPU lost: " + info.message; gpuReady = false; });

    const bs = S3 * 4;
    const STOR = GPUBufferUsage.STORAGE;
    const COPY = GPUBufferUsage.COPY_DST;
    const COPY_SRC = GPUBufferUsage.COPY_SRC;

    // Params
    paramsBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | COPY });
    const pb = new ArrayBuffer(64);
    const pu = new Uint32Array(pb);
    const pf = new Float32Array(pb);
    pu[0] = NN; pu[1] = S; pu[2] = S2; pu[3] = S3;
    pu[4] = N2; pf[5] = dt; pf[6] = half_d; pf[7] = TWO_PI;
    pf[8] = h; pf[9] = h2; pf[10] = 1/h; pf[11] = 1/h2;
    pf[12] = h3;
    device.queue.writeBuffer(paramsBuf, 0, pb);

    // Init config: 48 bytes header + 16*MAX_RIVALS bytes for rival positions
    initCfgBuf = device.createBuffer({ size: 48 + 16 * MAX_RIVALS, usage: GPUBufferUsage.UNIFORM | COPY });

    // Color configs: red(0) and black(1)
    colorBuf = [
      device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | COPY }),
      device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | COPY }),
    ];
    device.queue.writeBuffer(colorBuf[0], 0, new Uint32Array([0, 0, 0, 0]));
    device.queue.writeBuffer(colorBuf[1], 0, new Uint32Array([1, 0, 0, 0]));

    // Per-electron buffers
    const zeros = new Float32Array(S3);
    K_buf = device.createBuffer({ size: bs, usage: STOR | COPY | COPY_SRC });
    device.queue.writeBuffer(K_buf, 0, zeros);

    rhoTotalBuf = device.createBuffer({ size: bs, usage: STOR | COPY | COPY_SRC });
    device.queue.writeBuffer(rhoTotalBuf, 0, zeros);

    // Precompute r_c exclusion masks: 1.0 outside r_c, 0.0 inside
    rcMask_buf = [];
    for (let m = 0; m < NELEC; m++) {
      const maskData = new Float32Array(S3);
      maskData.fill(1.0);
      const nuc = nuclei[electronNucleus[m]];
      const rc_excl = nuc.rc_excl || 0;
      if (rc_excl > 0) {
        const rcCells = Math.ceil(rc_excl / h) + 1;
        for (let di = -rcCells; di <= rcCells; di++)
          for (let dj = -rcCells; dj <= rcCells; dj++)
            for (let dk = -rcCells; dk <= rcCells; dk++) {
              const gi = nuc.i + di, gj = nuc.j + dj, gk = nuc.k + dk;
              if (gi < 0 || gi >= S || gj < 0 || gj >= S || gk < 0 || gk >= S) continue;
              const r2 = (di * h) * (di * h) + (dj * h) * (dj * h) + (dk * h) * (dk * h);
              if (r2 < rc_excl * rc_excl) maskData[gi * S2 + gj * S + gk] = 0.0;
            }
        console.log(`Electron ${m}: r_c exclusion ${rc_excl} au around nucleus (${nuc.i},${nuc.j},${nuc.k})`);
      }
      rcMask_buf[m] = device.createBuffer({ size: bs, usage: STOR | COPY });
      device.queue.writeBuffer(rcMask_buf[m], 0, maskData);
    }

    for (let m = 0; m < NELEC; m++) {
      u_buf[m] = device.createBuffer({ size: bs, usage: STOR | COPY | COPY_SRC });
      w_buf[m] = device.createBuffer({ size: bs, usage: STOR | COPY | COPY_SRC });
      P_buf[m] = device.createBuffer({ size: bs, usage: STOR | COPY | COPY_SRC });
      device.queue.writeBuffer(u_buf[m], 0, zeros);
      device.queue.writeBuffer(w_buf[m], 0, zeros);
      device.queue.writeBuffer(P_buf[m], 0, zeros);
      normPartialBuf[m] = device.createBuffer({ size: N_REDUCE_WG * 4, usage: STOR | COPY });
      normCfgBuf[m] = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | COPY });
      // Write normalization target: USER_NORM_TARGETS overrides ne
      const normTargets = window.USER_NORM_TARGETS;
      const ne = normTargets ? normTargets[m] : nuclei[electronNucleus[m]].ne;
      device.queue.writeBuffer(normCfgBuf[m], 0, new Float32Array([ne, 0, 0, 0]));

      uNearestBuf[m] = device.createBuffer({ size: bs, usage: STOR | COPY | COPY_SRC });
      device.queue.writeBuffer(uNearestBuf[m], 0, zeros);
    }

    const sliceSize = S * S * SLICE_STRIDE * 4;
    sliceBuf = device.createBuffer({ size: sliceSize, usage: STOR | COPY_SRC | COPY });
    sliceReadBuf = device.createBuffer({ size: sliceSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    sliceCfgBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | COPY });

    // Compile shaders
    async function compile(name, code) {
      const mod = device.createShaderModule({ code });
      const info = await mod.getCompilationInfo();
      for (const msg of info.messages) {
        console[msg.type === 'error' ? 'error' : 'warn'](`${name} ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
      }
      if (info.messages.some(m => m.type === 'error')) throw new Error(`Shader ${name} failed`);
      return mod;
    }
    const initMod = await compile('init', initWGSL);
    const fusedMod = await compile('fusedWU_RB', fusedWU_RB_WGSL);
    const poissonMod = await compile('updateP_RB', updateP_RB_WGSL);
    const reduceMod = await compile('reduceNorm', reduceNormWGSL);
    const normMod = await compile('normalize', normalizeWGSL);
    const extractMod = await compile('extractSlice', extractSliceWGSL);
    const rhoTotalMod = await compile('rhoTotal', makeRhoTotalWGSL(NELEC));

    initPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: initMod, entryPoint: 'main' } });
    fusedWU_RB_PL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: fusedMod, entryPoint: 'main' } });
    updateP_RB_PL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: poissonMod, entryPoint: 'main' } });
    reduceNormPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: reduceMod, entryPoint: 'main' } });
    normalizePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: normMod, entryPoint: 'main' } });
    extractSlicePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: extractMod, entryPoint: 'main' } });
    rhoTotalPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: rhoTotalMod, entryPoint: 'main' } });

    // Compile uNearest shaders (one per electron)
    for (let m = 0; m < NELEC; m++) {
      const mod = await compile(`uNearest_${m}`, makeUNearestWGSL(NELEC, m));
      uNearestPL[m] = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
    }

    // Initialize K and electron domains
    // K is accumulated by dispatching init shader once per nucleus (to add each nucleus's contribution)
    // u/w/P are initialized per electron (domain assignment)
    await runInitKernels();

    // Apply r_c exclusion masks to initial u and w
    for (let m = 0; m < NELEC; m++) {
      const nuc = nuclei[electronNucleus[m]];
      const rc_excl = nuc.rc_excl || 0;
      if (rc_excl > 0) {
        // Read u, zero inside r_c, write back
        const tmpBuf = device.createBuffer({ size: bs, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(u_buf[m], 0, tmpBuf, 0, bs);
        device.queue.submit([enc.finish()]);
        await tmpBuf.mapAsync(GPUMapMode.READ);
        const uData = new Float32Array(tmpBuf.getMappedRange().slice(0));
        tmpBuf.unmap(); tmpBuf.destroy();
        const wData = new Float32Array(S3);
        // Read w too
        const tmpBuf2 = device.createBuffer({ size: bs, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const enc2 = device.createCommandEncoder();
        enc2.copyBufferToBuffer(w_buf[m], 0, tmpBuf2, 0, bs);
        device.queue.submit([enc2.finish()]);
        await tmpBuf2.mapAsync(GPUMapMode.READ);
        const wOrig = new Float32Array(tmpBuf2.getMappedRange().slice(0));
        tmpBuf2.unmap(); tmpBuf2.destroy();
        const rcCells = Math.ceil(rc_excl / h) + 1;
        for (let di = -rcCells; di <= rcCells; di++)
          for (let dj = -rcCells; dj <= rcCells; dj++)
            for (let dk = -rcCells; dk <= rcCells; dk++) {
              const gi = nuc.i + di, gj = nuc.j + dj, gk = nuc.k + dk;
              if (gi < 0 || gi >= S || gj < 0 || gj >= S || gk < 0 || gk >= S) continue;
              const r2 = (di*h)*(di*h) + (dj*h)*(dj*h) + (dk*h)*(dk*h);
              if (r2 < rc_excl * rc_excl) {
                const idx = gi * S2 + gj * S + gk;
                uData[idx] = 0;
                wOrig[idx] = 0;
              }
            }
        device.queue.writeBuffer(u_buf[m], 0, uData);
        device.queue.writeBuffer(w_buf[m], 0, wOrig);
        console.log(`Init: zeroed u,w inside r_c=${rc_excl} for electron ${m}`);
      }
    }

    gpuReady = true;
    console.log("RealQM Red-Black GPU initialized, nuclei=" + nuclei.length + ", NELEC=" + NELEC);
    // Debug: read back initial u/w to confirm init worked
    for (let m = 0; m < NELEC; m++) {
      const dbgBuf = device.createBuffer({ size: S3*4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const dbgEnc = device.createCommandEncoder();
      dbgEnc.copyBufferToBuffer(u_buf[m], 0, dbgBuf, 0, S3*4);
      device.queue.submit([dbgEnc.finish()]);
      await dbgBuf.mapAsync(GPUMapMode.READ);
      const uInit = new Float32Array(dbgBuf.getMappedRange().slice(0));
      dbgBuf.unmap(); dbgBuf.destroy();
      let maxU = 0, nzU = 0;
      for (let i2 = 0; i2 < uInit.length; i2++) {
        if (uInit[i2] > maxU) maxU = uInit[i2];
        if (uInit[i2] > 1e-6) nzU++;
      }
      console.log('[init debug] electron ' + m + ': max_u=' + maxU.toFixed(4) + ', nonzero_cells=' + nzU);
    }
  } catch (e) {
    gpuError = e.message;
    console.error("GPU init failed:", e);
  }
}

// Find the nearest rival nucleus for Voronoi domain of nucleus n
// If only one nucleus, returns the same position (so Voronoi check is always true)
function findRivalNucleus(nucIdx) {
  if (nuclei.length === 1) return nuclei[nucIdx];
  let bestDist = Infinity, bestNuc = null;
  const nuc = nuclei[nucIdx];
  for (let b = 0; b < nuclei.length; b++) {
    if (b === nucIdx) continue;
    const dx = (nuc.i - nuclei[b].i) * h;
    const dy = (nuc.j - nuclei[b].j) * h;
    const dz = (nuc.k - nuclei[b].k) * h;
    const d2 = dx*dx + dy*dy + dz*dz;
    if (d2 < bestDist) { bestDist = d2; bestNuc = nuclei[b]; }
  }
  return bestNuc;
}

// Helper: write InitCfg uniform buffer
// Layout (48 bytes header + 16*MAX_RIVALS bytes for rivals array):
//   [0]  nucI   [1]  nucJ   [2]  nucK   [3]  nucZ
//   [4]  rc     [5]  _pa    [6]  _pb    [7]  _pc
//   [8]  splitDir(u32)  [9] writeK(u32)  [10] rivCount(u32)  [11] _pd
//   [12..] rivals: vec4<f32> each (i, j, k, _) × MAX_RIVALS
function writeInitCfg(nuc, rivals, splitDir, rc, writeK) {
  const size = 48 + 16 * MAX_RIVALS;
  const buf = new ArrayBuffer(size);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  f[0] = nuc.i;  f[1] = nuc.j;  f[2] = nuc.k;  f[3] = nuc.Z;
  f[4] = rc;     f[5] = 0;      f[6] = 0;      f[7] = 0;
  u[8] = splitDir;
  u[9] = writeK ? 1 : 0;
  u[10] = Math.min(rivals.length, MAX_RIVALS);
  u[11] = 0;
  for (let r = 0; r < u[10]; r++) {
    const off = 12 + r * 4;
    f[off + 0] = rivals[r].i;
    f[off + 1] = rivals[r].j;
    f[off + 2] = rivals[r].k;
    f[off + 3] = 0;
  }
  device.queue.writeBuffer(initCfgBuf, 0, buf);
}

function allOtherNuclei(ownIdx) {
  const out = [];
  for (let b = 0; b < nuclei.length; b++) if (b !== ownIdx) out.push(nuclei[b]);
  return out;
}

async function runInitKernels() {
  // Phase 1: Accumulate K from ALL nuclei
  // writeK=1, writeUW=false (u/w won't be written since rc=0 means only 3au sphere limit,
  // but inDomain check uses Voronoi AND splitDir — use splitDir=2 (full Voronoi),
  // rc=0 (no rc restriction). This means P and u/w ARE written for any K pass,
  // but they get overwritten in Phase 2 per-electron. That's fine.

  // First zero K
  device.queue.writeBuffer(K_buf, 0, new Float32Array(S3));

  // For each nucleus, dispatch to accumulate K (writeK=1). Voronoi unused (all cells accepted).
  for (let n = 0; n < nuclei.length; n++) {
    const nuc = nuclei[n];
    writeInitCfg(nuc, [] /*no rivals during K accum*/, 2 /*full Voronoi*/, 0 /*no rc limit*/, true /*writeK*/);

    const initBG = device.createBindGroup({ layout: initPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: K_buf } },
      { binding: 2, resource: { buffer: u_buf[0] } },  // scratch
      { binding: 3, resource: { buffer: w_buf[0] } },  // scratch
      { binding: 4, resource: { buffer: P_buf[0] } },  // scratch
      { binding: 5, resource: { buffer: initCfgBuf } },
    ]});

    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(initPL);
    cp.setBindGroup(0, initBG);
    cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
    cp.end();
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();

  // Phase 2: Initialize each electron's u/w/P domain (writeK=0)
  for (let m = 0; m < NELEC; m++) {
    const nucIdx = electronNucleus[m];
    const nuc = nuclei[nucIdx];
    const rivals = allOtherNuclei(nucIdx);  // multi-rival Voronoi

    // Determine which sibling this electron is within its nucleus
    let sibling = 0;
    for (let n = 0; n < m; n++) {
      if (electronNucleus[n] === nucIdx) sibling++;
    }
    const numSiblings = nuclei[nucIdx].ne;

    // splitDir: 0=left half (1st of 2), 1=right half (2nd of 2), 2=full Voronoi (sole)
    let splitDir;
    if (numSiblings === 1) {
      splitDir = 2;
    } else {
      splitDir = sibling === 0 ? 0 : 1;
    }

    writeInitCfg(nuc, rivals, splitDir, nuc.rc, false /*writeK=0*/);

    // Zero u/w/P for this electron first, then dispatch domain init
    const zeroF = new Float32Array(S3);
    device.queue.writeBuffer(u_buf[m], 0, zeroF);
    device.queue.writeBuffer(w_buf[m], 0, zeroF);
    device.queue.writeBuffer(P_buf[m], 0, zeroF);

    const initBG = device.createBindGroup({ layout: initPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: K_buf } },
      { binding: 2, resource: { buffer: u_buf[m] } },
      { binding: 3, resource: { buffer: w_buf[m] } },
      { binding: 4, resource: { buffer: P_buf[m] } },
      { binding: 5, resource: { buffer: initCfgBuf } },
    ]});

    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(initPL);
    cp.setBindGroup(0, initBG);
    cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
    cp.end();
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
}

// Create bind groups (called after init since we need single buffers)
let fusedBG = [], poissonBG = [], reduceNormBG = [], normalizeBG = [], extractSliceBG = [];
let rhoTotalBG, uNearestBG = [];

function createBindGroups() {
  // rhoTotal bind group
  {
    const entries = [{ binding: 0, resource: { buffer: paramsBuf } }];
    for (let m = 0; m < NELEC; m++) {
      entries.push({ binding: m + 1, resource: { buffer: u_buf[m] } });
    }
    entries.push({ binding: NELEC + 1, resource: { buffer: rhoTotalBuf } });
    rhoTotalBG = device.createBindGroup({ layout: rhoTotalPL.getBindGroupLayout(0), entries });
  }

  // uNearest bind groups: one per electron. Only bind the OTHER electrons' u_buf
  // (self buffer isn't referenced in the shader and would be stripped from the layout).
  for (let m = 0; m < NELEC; m++) {
    const others = [];
    for (let n = 0; n < NELEC; n++) if (n !== m) others.push(n);
    const entries = [{ binding: 0, resource: { buffer: paramsBuf } }];
    others.forEach((n, i) => entries.push({ binding: i + 1, resource: { buffer: u_buf[n] } }));
    entries.push({ binding: others.length + 1, resource: { buffer: uNearestBuf[m] } });
    uNearestBG[m] = device.createBindGroup({ layout: uNearestPL[m].getBindGroupLayout(0), entries });
  }

  // Fused w+u: per electron, per color — uses uNearestBuf[m] for competition
  for (let m = 0; m < NELEC; m++) {
    fusedBG[m] = [];
    for (let color = 0; color < 2; color++) {
      fusedBG[m][color] = device.createBindGroup({ layout: fusedWU_RB_PL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: u_buf[m] } },
        { binding: 2, resource: { buffer: uNearestBuf[m] } },
        { binding: 3, resource: { buffer: w_buf[m] } },
        { binding: 4, resource: { buffer: K_buf } },
        { binding: 5, resource: { buffer: P_buf[m] } },
        { binding: 6, resource: { buffer: colorBuf[color] } },
        { binding: 7, resource: { buffer: rcMask_buf[m] } },
      ]});
    }
  }

  // Poisson: per electron, per color. binding 5 carries the occupation (norm target)
  // used by the SIC (n-1)/n factor in the shader.
  for (let m = 0; m < NELEC; m++) {
    poissonBG[m] = [];
    for (let color = 0; color < 2; color++) {
      poissonBG[m][color] = device.createBindGroup({ layout: updateP_RB_PL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: P_buf[m] } },
        { binding: 2, resource: { buffer: u_buf[m] } },
        { binding: 3, resource: { buffer: rhoTotalBuf } },
        { binding: 4, resource: { buffer: colorBuf[color] } },
        { binding: 5, resource: { buffer: normCfgBuf[m] } },
      ]});
    }
  }

  // Norm + normalize
  for (let m = 0; m < NELEC; m++) {
    reduceNormBG[m] = device.createBindGroup({ layout: reduceNormPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: u_buf[m] } },
      { binding: 2, resource: { buffer: normPartialBuf[m] } },
    ]});
    normalizeBG[m] = device.createBindGroup({ layout: normalizePL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: u_buf[m] } },
      { binding: 2, resource: { buffer: normPartialBuf[m] } },
      { binding: 3, resource: { buffer: normCfgBuf[m] } },
    ]});
  }

  // Extract slices
  for (let m = 0; m < NELEC; m++) {
    extractSliceBG[m] = device.createBindGroup({ layout: extractSlicePL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: u_buf[m] } },
      { binding: 2, resource: { buffer: w_buf[m] } },
      { binding: 3, resource: { buffer: P_buf[m] } },
      { binding: 4, resource: { buffer: K_buf } },
      { binding: 5, resource: { buffer: sliceBuf } },
      { binding: 6, resource: { buffer: sliceCfgBuf } },
    ]});
  }
}

function doSteps(nSteps) {
  if (!gpuReady) return;
  const enc = device.createCommandEncoder();

  for (let s = 0; s < nSteps; s++) {
    // Step 1: compute rho_total = sum_m u[m]^2
    { const cp = enc.beginComputePass();
      cp.setPipeline(rhoTotalPL);
      cp.setBindGroup(0, rhoTotalBG);
      cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
      cp.end(); }

    // Step 2: compute u_nearest[m] = min(u[n], n!=m) for each electron
    for (let m = 0; m < NELEC; m++) {
      const cp = enc.beginComputePass();
      cp.setPipeline(uNearestPL[m]);
      cp.setBindGroup(0, uNearestBG[m]);
      cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
      cp.end();
    }

    // Red: update w+u for all electrons, then P for all electrons
    for (let m = 0; m < NELEC; m++) {
      const cp = enc.beginComputePass();
      cp.setPipeline(fusedWU_RB_PL);
      cp.setBindGroup(0, fusedBG[m][0]);  // red
      cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
      cp.end();
    }
    for (let m = 0; m < NELEC; m++) {
      const cp = enc.beginComputePass();
      cp.setPipeline(updateP_RB_PL);
      cp.setBindGroup(0, poissonBG[m][0]);  // red P
      cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
      cp.end();
    }

    // Black: update w+u for all electrons, then P for all electrons
    for (let m = 0; m < NELEC; m++) {
      const cp = enc.beginComputePass();
      cp.setPipeline(fusedWU_RB_PL);
      cp.setBindGroup(0, fusedBG[m][1]);  // black
      cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
      cp.end();
    }
    for (let m = 0; m < NELEC; m++) {
      const cp = enc.beginComputePass();
      cp.setPipeline(updateP_RB_PL);
      cp.setBindGroup(0, poissonBG[m][1]);  // black P
      cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
      cp.end();
    }

    // Normalize each electron to its target ne
    if (s % NORM_INTERVAL === 0) {
      for (let m = 0; m < NELEC; m++) {
        const rp = enc.beginComputePass();
        rp.setPipeline(reduceNormPL);
        rp.setBindGroup(0, reduceNormBG[m]);
        rp.dispatchWorkgroups(N_REDUCE_WG);
        rp.end();

        const np = enc.beginComputePass();
        np.setPipeline(normalizePL);
        np.setBindGroup(0, normalizeBG[m]);
        np.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
        np.end();
      }
    }

    stepCount++;
  }

  device.queue.submit([enc.finish()]);
}

// Energy readback
let energyReadBufs = null;
function initEnergyReadback() {
  const bs = S3 * 4;
  energyReadBufs = {
    u: [], w: [], P: [],
    K_read: device.createBuffer({ size: bs, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
  };
  for (let m = 0; m < NELEC; m++) {
    energyReadBufs.u[m] = device.createBuffer({ size: bs, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    energyReadBufs.w[m] = device.createBuffer({ size: bs, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    energyReadBufs.P[m] = device.createBuffer({ size: bs, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }
}

async function computeEnergy() {
  if (!gpuReady) return;
  if (!energyReadBufs) initEnergyReadback();

  const bs = S3 * 4;
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(K_buf, 0, energyReadBufs.K_read, 0, bs);
  for (let m = 0; m < NELEC; m++) {
    enc.copyBufferToBuffer(u_buf[m], 0, energyReadBufs.u[m], 0, bs);
    enc.copyBufferToBuffer(w_buf[m], 0, energyReadBufs.w[m], 0, bs);
    enc.copyBufferToBuffer(P_buf[m], 0, energyReadBufs.P[m], 0, bs);
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

  // Energy: T where w>0.5 (no w weighting), V bare (matching p5)
  E_T = 0; E_eK = 0; E_ee = 0;
  for (let i = 1; i < NN; i++) {
    for (let j = 1; j < NN; j++) {
      for (let k = 1; k < NN; k++) {
        const id = i * S2 + j * S + k;
        for (let m = 0; m < NELEC; m++) {
          const v = uArr[m][id];
          const rho = v * v;
          const wv = wArr[m][id];
          if (wv > 0.1) {
            // Skip gradient legs that straddle a hard boundary (rc_excl or domain edge)
            // where u is forced to 0 — otherwise the discontinuity inflates T.
            let g2 = 0;
            if (wArr[m][id + S2] > 0.1) { const gx = uArr[m][id + S2] - v; g2 += gx*gx; }
            if (wArr[m][id + S]  > 0.1) { const gy = uArr[m][id + S]  - v; g2 += gy*gy; }
            if (wArr[m][id + 1]  > 0.1) { const gz = uArr[m][id + 1]  - v; g2 += gz*gz; }
            E_T += 0.5 * g2 * h;
          }
          E_eK += -K[id] * rho * h3;
          E_ee += PArr[m][id] * rho * h3;
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
  for (let m = 0; m < NELEC; m++) {
    device.queue.writeBuffer(sliceCfgBuf, 0, new Uint32Array([m * 3, SLICE_STRIDE, NELEC, 0]));
    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(extractSlicePL);
    cp.setBindGroup(0, extractSliceBG[m]);
    cp.dispatchWorkgroups(sliceWGs);
    cp.end();
    device.queue.submit([enc.finish()]);
  }
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(sliceBuf, 0, sliceReadBuf, 0, S * S * SLICE_STRIDE * 4);
  device.queue.submit([enc.finish()]);
  await sliceReadBuf.mapAsync(GPUMapMode.READ);
  sliceData = new Float32Array(sliceReadBuf.getMappedRange().slice(0));
  sliceReadBuf.unmap();
}

// --- p5.js Integration ---
const DISPLAY_SIZE = 700;
const STEPS_PER_FRAME = NN <= 100 ? 10 : (NN <= 150 ? 6 : (NN <= 200 ? 4 : 2));
let initDone = false;
let readbackPending = false;

// Electron colors for visualization (cycles for more than 3)
const ELEC_COLORS = [
  [255, 50, 50],   // red
  [50, 50, 255],   // blue
  [50, 200, 50],   // green
  [255, 165, 0],   // orange
  [180, 0, 180],   // purple
];

window.setup = function() {
  createCanvas(700, 730);
  initGPU().then(() => {
    createBindGroups();
    initDone = true;
  }).catch(e => {
    gpuError = e.message;
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

  // DEBUG: log once to diagnose static display
  if (!window._debugLogged && sliceData) {
    window._debugLogged = true;
    var _maxU = 0, _maxW = 0, _maxUW = 0, _nonzero = 0;
    for (var _i = 0; _i < sliceData.length; _i++) {
      if (sliceData[_i] > _maxU) _maxU = sliceData[_i];
      if (sliceData[_i] > 1e-6) _nonzero++;
    }
    for (var _j = 0; _j < S*S; _j++) {
      for (var _m = 0; _m < NELEC; _m++) {
        var _u = sliceData[_j * SLICE_STRIDE + _m*3];
        var _w = sliceData[_j * SLICE_STRIDE + _m*3 + 1];
        if (_u * _w > _maxUW) _maxUW = _u * _w;
      }
    }
    console.log('[rb debug] sliceData len=' + sliceData.length + ' nonzero=' + _nonzero + ' max_val=' + _maxU.toFixed(4) + ' max_u*w=' + _maxUW.toFixed(4) + ' stepCount=' + stepCount);
  }
  if (!readbackPending && (frameCount % 3 === 0 || stepCount >= MAX_STEPS)) {
    readbackPending = true;
    (async function() {
      try {
        await device.queue.onSubmittedWorkDone();
        await computeEnergy();
        await extractSlice();
      } catch(e) {
        gpuError = "Readback: " + e.message;
      }
      readbackPending = false;
      // Expose latest energies + step count for external sweep pages
      window.__realqm = {
        stepCount, maxSteps: MAX_STEPS,
        E_T, E_eK, E_ee, V_KK, E_tot,
        done: stepCount >= MAX_STEPS,
      };
    })();
  }

  // Draw visualization
  noStroke();
  const imgW = 350, imgH = 350;
  const sx = imgW / S, sy = imgH / S;

  if (sliceData) {
    for (let i = 1; i < NN; i++) {
      for (let j = 1; j < NN; j++) {
        const idx = (j * S + i) * SLICE_STRIDE;
        for (let m = 0; m < NELEC; m++) {
          // Each electron occupies 3 slots: u at m*3, w at m*3+1, P at m*3+2
          const u_val = sliceData[idx + m * 3];
          const w_val = sliceData[idx + m * 3 + 1];
          const wu = u_val * w_val;
          const a = Math.min(255, 500 * Math.abs(wu));
          if (a > 2) {
            const col = ELEC_COLORS[m % ELEC_COLORS.length];
            fill(col[0], col[1], col[2], a);
            square(i * sx, j * sy, Math.max(sx, 1));
          }
        }
      }
    }

    fill(0); noStroke();
    for (const nuc of nuclei) {
      circle(nuc.i * sx, nuc.j * sy, 8);
    }

    // 1D line plots along i-axis at j=N2 — plot u and w for every electron on the
    // same baseline (y=200) with the same vertical scale (×100) so they overlay directly
    for (let i = 0; i < S; i++) {
      const idx = (N2 * S + i) * SLICE_STRIDE;
      fill(0); ellipse(i * (350/S), 300, 2);
      for (let m = 0; m < NELEC; m++) {
        const uVal = sliceData[idx + m * 3];
        const wVal = sliceData[idx + m * 3 + 1];
        fill(0);               ellipse(i * (350/S), 200 - 100*wVal, 3);  // w (black)
        fill(255, 255, 0);     ellipse(i * (350/S), 200 - 100*uVal, 3);  // u (yellow)
      }
      // Combined P from all electrons
      let pSum = 0;
      for (let m = 0; m < NELEC; m++) pSum += sliceData[idx + m * 3 + 2];
      fill(0, 200, 200); ellipse(i * (350/S), 300 - 100*pSum, 2);
      // K (last slot)
      fill(0, 0, 200); ellipse(i * (350/S), 300 - 30*sliceData[idx + SLICE_STRIDE - 1], 3);
    }
  }

  const rx = 370;
  fill(0); textSize(13);
  const molName = window.USER_MOL_NAME || (NELEC === 2 ? (D_CELLS < 1 ? "He RealQM RB GPU" : "H2 RealQM RB GPU") : `${NELEC}e RealQM RB GPU`);
  text(molName, rx, 20);
  text(NN + "^3 / " + screenAu + " au", rx, 38);
  if (!window.USER_NUCLEI) {
    text("R = " + (D_CELLS * h).toFixed(2) + " au", rx, 56);
  } else {
    text(`${nuclei.length} nuclei, ${NELEC} electrons`, rx, 56);
  }
  text("step " + stepCount, rx, 74);

  textSize(12);
  text("T = " + E_T.toFixed(4), rx, 110);
  text("V_eK = " + E_eK.toFixed(4), rx, 126);
  text("V_ee = " + E_ee.toFixed(4), rx, 142);
  text("V_KK = " + V_KK.toFixed(4), rx, 158);
  textSize(13);
  text("E = " + E_tot.toFixed(4), rx, 180);
  textSize(11);
  const refStr = window.USER_REF || (NELEC === 2 ? (D_CELLS < 1 ? "(He ref -2.903 Ha)" : "(H2 ref -1.17 Ha)") : "");
  if (refStr) text(refStr, rx, 196);

  if (stepCount > 0 && stepCount % 100 === 0) {
    console.log("step=" + stepCount + " T=" + E_T.toFixed(4) + " V_eK=" + E_eK.toFixed(4) +
      " V_ee=" + E_ee.toFixed(4) + " E=" + E_tot.toFixed(4));
  }
};
