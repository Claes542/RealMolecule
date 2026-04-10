// realqm_gpu.js — Faithful GPU port of original RealQM
// Per-electron wavefunctions u[m], level set functions w[m], Poisson potentials P[m]
// w-weighted Laplacian, coupled forward-Euler Poisson, front tracking
// Fused w+u kernel + sequential P updates to match CPU's per-cell coupling
// r_c pseudopotential: w=0 inside r_c provides Neumann BC via w-weighting, u free

"use strict";

// --- Configuration ---
const NN = window.USER_NN || 100;
const screenAu = window.USER_SCREEN || 10;
const D_CELLS = window.USER_D || 14;
const MAX_STEPS = window.USER_STEPS || 2000;
const NELEC = window.USER_NELEC || 2;
const USER_RC = window.USER_RC || [];
const NORM_INTERVAL = 1;
const POISSON_ITERS = window.POISSON_ITERS || 1;  // P sub-iterations per ITP step

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

console.log(`RealQM GPU: NN=${NN}, S=${S}, S3=${S3}, h=${h.toFixed(4)}, dt=${dt.toExponential(3)}, NELEC=${NELEC}`);
console.log(`Dispatch: ${DISPATCH_X} x ${DISPATCH_Y} workgroups of ${WG_SIZE}, interior=${INTERIOR}`);

// --- Nuclear positions ---
const nuclei = window.USER_NUCLEI || [
  { i: N2 - Math.round(D_CELLS / 2), j: N2, k: N2, Z: 1, rc: USER_RC[0] || 0 },
  { i: N2 + Math.round(D_CELLS / 2), j: N2, k: N2, Z: 1, rc: USER_RC[1] || 0 },
];
let V_KK = (function() {
  let e = 0;
  for (let a = 0; a < nuclei.length; a++)
    for (let b = a + 1; b < nuclei.length; b++) {
      const di = (nuclei[a].i - nuclei[b].i) * h;
      const dj = (nuclei[a].j - nuclei[b].j) * h;
      const dk = (nuclei[a].k - nuclei[b].k) * h;
      e += nuclei[a].Z * nuclei[b].Z / Math.sqrt(di*di + dj*dj + dk*dk);
    }
  return e;
})();
console.log(`Nuclear repulsion V_KK = ${V_KK.toFixed(6)}`);
for (const nuc of nuclei) {
  if (nuc.rc > 0) console.log(`Nucleus Z=${nuc.Z} at (${nuc.i},${nuc.j},${nuc.k}) r_c=${nuc.rc}`);
}

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

// Init shader: bare softened Coulomb for K; w=0 inside r_c, u free
const initWGSL = `
${paramStructWGSL}
struct InitCfg {
  nucI: f32, nucJ: f32, nucK: f32, nucZ: f32,
  otherI: f32, otherJ: f32, otherK: f32, otherZ: f32,
  splitAxis: u32, splitDir: u32, splitPos: u32, elecIdx: u32,
  nucRC2: f32, otherRC2: f32, _pad0: u32, _pad1: u32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> K: array<f32>;
@group(0) @binding(2) var<storage, read_write> u_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> w_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> P_out: array<f32>;
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

  // Distance to this electron's nucleus
  let dx1 = (f32(i) - cfg.nucI) * p.h;
  let dy1 = (f32(j) - cfg.nucJ) * p.h;
  let dz1 = (f32(k) - cfg.nucK) * p.h;
  let r1_raw = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1);
  let r1_soft = sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1 + p.h2);
  let nucRC = sqrt(cfg.nucRC2);

  // Distance to other nucleus
  let dx2 = (f32(i) - cfg.otherI) * p.h;
  let dy2 = (f32(j) - cfg.otherJ) * p.h;
  let dz2 = (f32(k) - cfg.otherK) * p.h;
  let r2_soft = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2 + p.h2);

  // Nuclear potential K — bare softened Coulomb (r_c handled by w=0 BC)
  K[id] += cfg.nucZ / r1_soft;

  // Domain assignment
  let inDomain = select(
    i > cfg.splitPos,
    i < cfg.splitPos,
    cfg.splitDir == 0u
  );

  if (inDomain) {
    if (nucRC > 0.0 && r1_raw < nucRC) {
      w_out[id] = 0.0;
    } else {
      w_out[id] = 1.0;
    }
    u_out[id] = exp(-r1_soft);
  }

  // Initial P: half-Coulomb from other nucleus
  P_out[id] = 0.5 * cfg.otherZ / r2_soft;
}
`;

// Fused w+u update — w-weighted Laplacian, front tracking, matching h2_clean.js
const fusedWU_WGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> u_self: array<f32>;
@group(0) @binding(2) var<storage, read> u_other: array<f32>;
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
  let c = 0.5 * (u_self[id] - u_other[id]);
  let wc = w_self[id];

  // Update w: front tracking (no clamp, matching p5/clean)
  let lap_w = w_self[id + p.S2] + w_self[id - p.S2]
            + w_self[id + p.S]  + w_self[id - p.S]
            + w_self[id + 1u]   + w_self[id - 1u] - 6.0 * wc;
  let gwx = (w_self[id + p.S2] - w_self[id - p.S2]) * p.inv_h;
  let gwy = (w_self[id + p.S]  - w_self[id - p.S])  * p.inv_h;
  let gwz = (w_self[id + 1u]   - w_self[id - 1u])   * p.inv_h;
  let grad_w = sqrt(gwx*gwx + gwy*gwy + gwz*gwz);
  var new_w = wc + 2.0 * p.dt * abs(c) * lap_w * p.inv_h2 + 10.0 * p.dt * c * grad_w;
  new_w *= mask;
  w_out[id] = new_w;

  // Update u: w-weighted Laplacian
  let uc = u_self[id];
  let flux_xp = (u_self[id + p.S2] - uc) * (w_self[id + p.S2] + wc) * 0.5;
  let flux_xm = (uc - u_self[id - p.S2]) * (wc + w_self[id - p.S2]) * 0.5;
  let flux_yp = (u_self[id + p.S] - uc) * (w_self[id + p.S] + wc) * 0.5;
  let flux_ym = (uc - u_self[id - p.S]) * (wc + w_self[id - p.S]) * 0.5;
  let flux_zp = (u_self[id + 1u] - uc) * (w_self[id + 1u] + wc) * 0.5;
  let flux_zm = (uc - u_self[id - 1u]) * (wc + w_self[id - 1u]) * 0.5;
  let wlap = (flux_xp - flux_xm) + (flux_yp - flux_ym) + (flux_zp - flux_zm);

  u_out[id] = uc + p.half_d * wlap + p.dt * (Kbuf[id] - 2.0 * Pot[id]) * uc * wc;
}
`;

// Forward Euler Poisson update (coupled with ITP, same dt)
const updateP_both_WGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> P0_in: array<f32>;
@group(0) @binding(2) var<storage, read> P1_in: array<f32>;
@group(0) @binding(3) var<storage, read> u_src0: array<f32>;
@group(0) @binding(4) var<storage, read> u_src1: array<f32>;
@group(0) @binding(5) var<storage, read_write> P0_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> P1_out: array<f32>;

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

  let P0c = P0_in[id];
  let lap_P0 = P0_in[id + p.S2] + P0_in[id - p.S2]
             + P0_in[id + p.S]  + P0_in[id - p.S]
             + P0_in[id + 1u]   + P0_in[id - 1u] - 6.0 * P0c;
  let rho0 = u_src0[id];
  P0_out[id] = P0c + p.dt * (lap_P0 * p.inv_h2 + p.TWO_PI * rho0 * rho0);

  let P1c = P1_in[id];
  let lap_P1 = P1_in[id + p.S2] + P1_in[id - p.S2]
             + P1_in[id + p.S]  + P1_in[id - p.S]
             + P1_in[id + 1u]   + P1_in[id - 1u] - 6.0 * P1c;
  let rho1 = u_src1[id];
  P1_out[id] = P1c + p.dt * (lap_P1 * p.inv_h2 + p.TWO_PI * rho1 * rho1);
}
`;

// Norm reduction (unchanged — u=0 inside r_c so those cells contribute 0)
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

// Normalize (u is free — no r_c masking)
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

// Extract 2D slice (unchanged)
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
  let base = gid.x * 7u + scfg.offset;
  slice[base]      = u_el[id3];
  slice[base + 1u] = w_el[id3];
  slice[base + 2u] = P_el[id3];
  if (scfg.offset == 0u) {
    slice[gid.x * 7u + 6u] = Kbuf[id3];
  }
}
`;

// --- GPU State ---
let device, gpuReady = false, gpuError = null;
let paramsBuf, initCfgBuf, sorCfgBuf;
let K_buf;
let u_buf = [[], []];
let w_buf = [[], []];
let P_buf = [[], []];
let normPartialBuf = [];
let sliceBuf, sliceReadBuf, sliceCfgBuf;
let cur = 0;

let initPL, fusedWU_PL, updateP_both_PL;
let reduceNormPL, normalizePL, extractSlicePL;
let initBG = [];
let fusedWU_BG = [[], []];      // [elec][cur] — group 0
let fusedWU_mask_BG = [[], []]; // [elec][cur] — group 1 (w mask)
let rcMask_buf = [];
let updateP_pass1_BG = [];
let updateP_pass2_BG = [];
let reduceNormBG = [[], []], normalizeBG = [[], []];
let extractSliceBG = [];

// Energy values
let E_T = 0, E_eK = 0, E_ee = 0, E_pot = 0, E_tot = 0;
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

    // Init config uniform
    initCfgBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | COPY });
    // SOR color configs: pre-create two buffers for red(0) and black(1)
    sorCfgBuf = [
      device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | COPY }),
      device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | COPY }),
    ];
    device.queue.writeBuffer(sorCfgBuf[0], 0, new Uint32Array([0, 0, 0, 0]));  // red
    device.queue.writeBuffer(sorCfgBuf[1], 0, new Uint32Array([1, 0, 0, 0]));  // black

    // Buffers
    K_buf = device.createBuffer({ size: bs, usage: STOR | COPY | COPY_SRC });
    const zeros = new Float32Array(S3);
    device.queue.writeBuffer(K_buf, 0, zeros);

    // Per-electron r_c masks: w=0 inside r_c of own nucleus
    rcMask_buf = [];
    for (let m = 0; m < NELEC; m++) {
      const maskData = new Float32Array(S3);
      maskData.fill(1.0);
      const nuc = nuclei[m];
      const rc = nuc.rc || 0;
      if (rc > 0) {
        const rc2 = rc * rc;
        const rcCells = Math.ceil(rc / h) + 1;
        for (let di = -rcCells; di <= rcCells; di++)
          for (let dj = -rcCells; dj <= rcCells; dj++)
            for (let dk = -rcCells; dk <= rcCells; dk++) {
              const gi = nuc.i + di, gj = nuc.j + dj, gk = nuc.k + dk;
              if (gi < 0 || gi >= S || gj < 0 || gj >= S || gk < 0 || gk >= S) continue;
              const r2 = (di * h) * (di * h) + (dj * h) * (dj * h) + (dk * h) * (dk * h);
              if (r2 < rc2) maskData[gi * S2 + gj * S + gk] = 0.0;
            }
        console.log(`Electron ${m}: w=0 inside r_c=${rc} au of nucleus at (${nuc.i},${nuc.j},${nuc.k})`);
      }
      rcMask_buf[m] = device.createBuffer({ size: bs, usage: STOR | COPY });
      device.queue.writeBuffer(rcMask_buf[m], 0, maskData);
    }

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
      device.queue.writeBuffer(u_buf[m][0], 0, zeros);
      device.queue.writeBuffer(u_buf[m][1], 0, zeros);
      device.queue.writeBuffer(w_buf[m][0], 0, zeros);
      device.queue.writeBuffer(w_buf[m][1], 0, zeros);
      device.queue.writeBuffer(P_buf[m][0], 0, zeros);
      device.queue.writeBuffer(P_buf[m][1], 0, zeros);

      normPartialBuf[m] = device.createBuffer({ size: N_REDUCE_WG * 4, usage: STOR | COPY });
    }

    // Slice buffer
    const sliceSize = S * S * 7 * 4;
    sliceBuf = device.createBuffer({ size: sliceSize, usage: STOR | COPY_SRC | COPY });
    sliceReadBuf = device.createBuffer({ size: sliceSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    sliceCfgBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | COPY });

    // Compile shaders
    async function compile(name, code) {
      console.log(`Compiling shader: ${name}`);
      const mod = device.createShaderModule({ code });
      const info = await mod.getCompilationInfo();
      for (const msg of info.messages) {
        console[msg.type === 'error' ? 'error' : 'warn'](`${name} ${msg.type}: ${msg.message} (line ${msg.lineNum}:${msg.linePos})`);
      }
      if (info.messages.some(m => m.type === 'error')) {
        throw new Error(`Shader ${name} compilation failed`);
      }
      console.log(`Shader ${name} compiled OK`);
      return mod;
    }
    const initMod = await compile('init', initWGSL);
    const fusedWU_Mod = await compile('fusedWU', fusedWU_WGSL);
    const updateP_both_Mod = await compile('updateP_both', updateP_both_WGSL);
    const reduceNormMod = await compile('reduceNorm', reduceNormWGSL);
    const normalizeMod = await compile('normalize', normalizeWGSL);
    const extractSliceMod = await compile('extractSlice', extractSliceWGSL);

    // Pipelines
    console.log("Creating pipelines...");
    initPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: initMod, entryPoint: 'main' } });
    console.log("  initPL OK");
    fusedWU_PL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: fusedWU_Mod, entryPoint: 'main' } });
    console.log("  fusedWU_PL OK");
    updateP_both_PL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: updateP_both_Mod, entryPoint: 'main' } });
    console.log("  updateP_both_PL OK");
    reduceNormPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: reduceNormMod, entryPoint: 'main' } });
    console.log("  reduceNormPL OK");
    normalizePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: normalizeMod, entryPoint: 'main' } });
    console.log("  normalizePL OK");
    extractSlicePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: extractSliceMod, entryPoint: 'main' } });
    console.log("  extractSlicePL OK");

    // Init bind groups and run GPU init for each electron
    for (let m = 0; m < NELEC; m++) {
      initBG[m] = device.createBindGroup({ layout: initPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: K_buf } },
        { binding: 2, resource: { buffer: u_buf[m][0] } },
        { binding: 3, resource: { buffer: w_buf[m][0] } },
        { binding: 4, resource: { buffer: P_buf[m][0] } },
        { binding: 5, resource: { buffer: initCfgBuf } },
      ]});

      const nuc = nuclei[m];
      const otherNuc = nuclei[1 - m];
      const nucRC = nuc.rc || 0;
      const otherRC = otherNuc.rc || 0;
      const cfg = new Float32Array([nuc.i, nuc.j, nuc.k, nuc.Z, otherNuc.i, otherNuc.j, otherNuc.k, otherNuc.Z]);
      const cfgU32 = new Uint32Array([0, m === 0 ? 0 : 1, N2, m]);
      const cfgRC = new Float32Array([nucRC * nucRC, otherRC * otherRC]);
      const cfgBuf = new ArrayBuffer(64);
      new Float32Array(cfgBuf, 0, 8).set(cfg);
      new Uint32Array(cfgBuf, 32, 4).set(cfgU32);
      new Float32Array(cfgBuf, 48, 2).set(cfgRC);
      device.queue.writeBuffer(initCfgBuf, 0, cfgBuf);
      console.log(`Electron ${m}: nuc at (${nuc.i},${nuc.j},${nuc.k}) Z=${nuc.Z} rc=${nucRC}`);

      const enc = device.createCommandEncoder();
      const cp = enc.beginComputePass();
      cp.setPipeline(initPL);
      cp.setBindGroup(0, initBG[m]);
      cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
      cp.end();
      device.queue.submit([enc.finish()]);
    }
    await device.queue.onSubmittedWorkDone();

    // --- Create simulation bind groups ---

    // Fused w+u: group 0 (data) + group 1 (per-electron r_c mask)
    for (let c = 0; c < 2; c++) {
      for (let m = 0; m < NELEC; m++) {
        const other = 1 - m;
        const P_read = m === 0 ? P_buf[0][0] : P_buf[1][1]; // el0: P from buf[0], el1: P from buf[1] (after pass1)

        fusedWU_BG[m][c] = device.createBindGroup({ layout: fusedWU_PL.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: paramsBuf } },
          { binding: 1, resource: { buffer: u_buf[m][c] } },
          { binding: 2, resource: { buffer: u_buf[other][c] } },
          { binding: 3, resource: { buffer: w_buf[m][c] } },
          { binding: 4, resource: { buffer: K_buf } },
          { binding: 5, resource: { buffer: P_read } },
          { binding: 6, resource: { buffer: w_buf[m][1-c] } },
          { binding: 7, resource: { buffer: u_buf[m][1-c] } },
        ]});

        // Group 1: per-electron mask (w=0 inside r_c)
        fusedWU_mask_BG[m][c] = device.createBindGroup({ layout: fusedWU_PL.getBindGroupLayout(1), entries: [
          { binding: 0, resource: { buffer: rcMask_buf[m] } },
        ]});
      }
    }

    // P update pass 1 (SOR Jacobi ping-pong)
    for (let c = 0; c < 2; c++) {
      updateP_pass1_BG[c] = device.createBindGroup({ layout: updateP_both_PL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: P_buf[0][0] } },
        { binding: 2, resource: { buffer: P_buf[1][0] } },
        { binding: 3, resource: { buffer: u_buf[1][c] } },
        { binding: 4, resource: { buffer: u_buf[0][1-c] } },
        { binding: 5, resource: { buffer: P_buf[0][1] } },
        { binding: 6, resource: { buffer: P_buf[1][1] } },
      ]});
    }

    // P update pass 2
    for (let c = 0; c < 2; c++) {
      updateP_pass2_BG[c] = device.createBindGroup({ layout: updateP_both_PL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: P_buf[0][1] } },
        { binding: 2, resource: { buffer: P_buf[1][1] } },
        { binding: 3, resource: { buffer: u_buf[1][1-c] } },
        { binding: 4, resource: { buffer: u_buf[0][1-c] } },
        { binding: 5, resource: { buffer: P_buf[0][0] } },
        { binding: 6, resource: { buffer: P_buf[1][0] } },
      ]});
    }

    // Norm reduction and normalize bind groups
    for (let m = 0; m < NELEC; m++) {
      for (let c = 0; c < 2; c++) {
        reduceNormBG[m][c] = device.createBindGroup({ layout: reduceNormPL.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: paramsBuf } },
          { binding: 1, resource: { buffer: u_buf[m][1-c] } },
          { binding: 2, resource: { buffer: normPartialBuf[m] } },
        ]});
        normalizeBG[m][c] = device.createBindGroup({ layout: normalizePL.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: paramsBuf } },
          { binding: 1, resource: { buffer: u_buf[m][1-c] } },
          { binding: 2, resource: { buffer: normPartialBuf[m] } },
        ]});
      }
    }

    // Extract slice bind groups
    for (let m = 0; m < NELEC; m++) {
      extractSliceBG[m] = [];
      for (let c = 0; c < 2; c++) {
        extractSliceBG[m][c] = device.createBindGroup({ layout: extractSlicePL.getBindGroupLayout(0), entries: [
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

    gpuReady = true;
    console.log("RealQM GPU initialized — fused w+u with r_c pseudopotential");
  } catch (e) {
    gpuError = e.message;
    console.error("GPU init failed:", e);
  }
}

// Reinitialize simulation with new nuclear distance D_new (in grid cells)
// Keeps all GPU pipelines/buffers, just re-zeros and re-runs init shader
async function reinitSim(D_new) {
  if (!gpuReady) return;

  // Update nuclei positions
  nuclei[0].i = N2 - Math.round(D_new / 2);
  nuclei[1].i = N2 + Math.round(D_new / 2);

  // Recompute V_KK
  const di_h = (nuclei[0].i - nuclei[1].i) * h;
  const dj_h = (nuclei[0].j - nuclei[1].j) * h;
  const dk_h = (nuclei[0].k - nuclei[1].k) * h;
  V_KK = nuclei[0].Z * nuclei[1].Z / Math.sqrt(di_h*di_h + dj_h*dj_h + dk_h*dk_h);

  const bs = S3 * 4;
  const zeros = new Float32Array(S3);

  // Zero K, u, w, P buffers
  device.queue.writeBuffer(K_buf, 0, zeros);
  for (let m = 0; m < NELEC; m++) {
    device.queue.writeBuffer(u_buf[m][0], 0, zeros);
    device.queue.writeBuffer(u_buf[m][1], 0, zeros);
    device.queue.writeBuffer(w_buf[m][0], 0, zeros);
    device.queue.writeBuffer(w_buf[m][1], 0, zeros);
    device.queue.writeBuffer(P_buf[m][0], 0, zeros);
    device.queue.writeBuffer(P_buf[m][1], 0, zeros);
  }

  // Update r_c masks
  for (let m = 0; m < NELEC; m++) {
    const maskData = new Float32Array(S3);
    maskData.fill(1.0);
    const nuc = nuclei[m];
    const rc = nuc.rc || 0;
    if (rc > 0) {
      const rc2 = rc * rc;
      const rcCells = Math.ceil(rc / h) + 1;
      for (let di = -rcCells; di <= rcCells; di++)
        for (let dj = -rcCells; dj <= rcCells; dj++)
          for (let dk = -rcCells; dk <= rcCells; dk++) {
            const gi = nuc.i + di, gj = nuc.j + dj, gk = nuc.k + dk;
            if (gi < 0 || gi >= S || gj < 0 || gj >= S || gk < 0 || gk >= S) continue;
            const r2 = (di * h) * (di * h) + (dj * h) * (dj * h) + (dk * h) * (dk * h);
            if (r2 < rc2) maskData[gi * S2 + gj * S + gk] = 0.0;
          }
    }
    device.queue.writeBuffer(rcMask_buf[m], 0, maskData);
  }

  // Re-run init shader for each electron
  for (let m = 0; m < NELEC; m++) {
    const nuc = nuclei[m];
    const otherNuc = nuclei[1 - m];
    const nucRC = nuc.rc || 0;
    const otherRC = otherNuc.rc || 0;
    const cfg = new Float32Array([nuc.i, nuc.j, nuc.k, nuc.Z, otherNuc.i, otherNuc.j, otherNuc.k, otherNuc.Z]);
    const cfgU32 = new Uint32Array([0, m === 0 ? 0 : 1, N2, m]);
    const cfgRC = new Float32Array([nucRC * nucRC, otherRC * otherRC]);
    const cfgBuf = new ArrayBuffer(64);
    new Float32Array(cfgBuf, 0, 8).set(cfg);
    new Uint32Array(cfgBuf, 32, 4).set(cfgU32);
    new Float32Array(cfgBuf, 48, 2).set(cfgRC);
    device.queue.writeBuffer(initCfgBuf, 0, cfgBuf);

    const enc = device.createCommandEncoder();
    const cp = enc.beginComputePass();
    cp.setPipeline(initPL);
    cp.setBindGroup(0, initBG[m]);
    cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
    cp.end();
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();

  cur = 0;
  stepCount = 0;
  E_T = 0; E_eK = 0; E_ee = 0; E_pot = 0; E_tot = 0;
  sliceData = null;
  energyReadBufs = null;
  console.log(`Reinit: D=${D_new} R=${(D_new * h).toFixed(2)} au, V_KK=${V_KK.toFixed(4)}`);
}

function doSteps(nSteps) {
  if (!gpuReady) return;
  const enc = device.createCommandEncoder();

  for (let s = 0; s < nSteps; s++) {
    const next = 1 - cur;

    // Step 1: Fused w+u for electron 0
    {
      const cp = enc.beginComputePass();
      cp.setPipeline(fusedWU_PL);
      cp.setBindGroup(0, fusedWU_BG[0][cur]);
      cp.setBindGroup(1, fusedWU_mask_BG[0][cur]);
      cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
      cp.end();
    }

    // Step 2: SOR Jacobi P update (pass 1: buf[0]→buf[1])
    {
      const cp = enc.beginComputePass();
      cp.setPipeline(updateP_both_PL);
      cp.setBindGroup(0, updateP_pass1_BG[cur]);
      cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
      cp.end();
    }

    // Step 3: Fused w+u for electron 1 (reads P1 from buf[1])
    {
      const cp = enc.beginComputePass();
      cp.setPipeline(fusedWU_PL);
      cp.setBindGroup(0, fusedWU_BG[1][cur]);
      cp.setBindGroup(1, fusedWU_mask_BG[1][cur]);
      cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
      cp.end();
    }

    // Step 4: SOR Jacobi P update (pass 2: buf[1]→buf[0])
    {
      const cp = enc.beginComputePass();
      cp.setPipeline(updateP_both_PL);
      cp.setBindGroup(0, updateP_pass2_BG[cur]);
      cp.dispatchWorkgroups(DISPATCH_X, DISPATCH_Y);
      cp.end();
    }

    // Step 5: Normalize both electrons
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

    cur = next;
    stepCount++;
  }

  device.queue.submit([enc.finish()]);
}

// Full 3D CPU readback for energy
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
          // w-weighted energy (matching h2_clean.js)
          const gx = uArr[m][id + S2] - v;
          const gy = uArr[m][id + S]  - v;
          const gz = uArr[m][id + 1]  - v;
          E_T += 0.5 * wv * (gx*gx + gy*gy + gz*gz) * h;
          E_eK += -K[id] * wv * rho * h3;
          E_ee += PArr[m][id] * wv * rho * h3;
          E_pot += (PArr[m][id] - K[id]) * wv * rho * h3;
        }
      }
    }
  }
  E_tot = E_T + E_eK + E_ee + V_KK;

  // Debug: check normalization
  for (let m = 0; m < NELEC; m++) {
    let norm = 0;
    for (let i = 1; i < NN; i++)
      for (let j = 1; j < NN; j++)
        for (let k = 1; k < NN; k++)
          norm += uArr[m][i*S2+j*S+k] ** 2 * h3;
    console.log("  norm[" + m + "] = " + norm.toFixed(4));
  }

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
  const enc = device.createCommandEncoder();
  const sliceBytes = S * S * 7 * 4;
  enc.copyBufferToBuffer(sliceBuf, 0, sliceReadBuf, 0, sliceBytes);
  device.queue.submit([enc.finish()]);
  await sliceReadBuf.mapAsync(GPUMapMode.READ);
  sliceData = new Float32Array(sliceReadBuf.getMappedRange().slice(0));
  sliceReadBuf.unmap();
}

// --- p5.js Integration ---
const DISPLAY_SIZE = 700;
const STEPS_PER_FRAME = NN <= 100 ? 10 : 2;
let initDone = false;
let readbackPending = false;
let lastLogStep = -1;

window.setup = function() {
  createCanvas(700, 730);
  initGPU().then(() => {
    initDone = true;
    console.log("GPU init complete, starting simulation");
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

  if (!readbackPending && (frameCount % 3 === 0 || stepCount >= MAX_STEPS)) {
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

  // Draw visualization
  noStroke();
  const imgW = 350, imgH = 350;
  const imgX0 = 0, imgY0 = 0;
  const sx = imgW / S, sy = imgH / S;

  const plotX0 = imgX0, plotW = imgW;
  const plotH_w = 100, plotH_u = 100, plotH_P = 100;
  const gap = 18;
  const baseW = imgY0 + imgH + gap + plotH_w;
  const baseU = baseW + gap + plotH_u;
  const baseP = baseU + gap + plotH_P;

  if (sliceData) {
    for (let i = 1; i < NN; i++) {
      for (let j = 1; j < NN; j++) {
        const idx = (j * S + i) * 7;
        const wu1 = sliceData[idx] * sliceData[idx + 1];
        const wu2 = sliceData[idx + 3] * sliceData[idx + 4];
        const a1 = Math.min(255, 500 * Math.abs(wu1));
        const a2 = Math.min(255, 500 * Math.abs(wu2));
        if (a1 > 2) {
          fill(255, 50, 50, a1);
          square(imgX0 + i * sx, imgY0 + j * sy, Math.max(sx, 1));
        }
        if (a2 > 2) {
          fill(50, 50, 255, a2);
          square(imgX0 + i * sx, imgY0 + j * sy, Math.max(sx, 1));
        }
      }
    }

    // Draw r_c circles around nuclei
    noFill();
    for (let n = 0; n < nuclei.length; n++) {
      const nuc = nuclei[n];
      const rc = nuc.rc || 0;
      if (rc > 0) {
        stroke(255, 200, 0, 150); strokeWeight(1.5);
        const rcPx = rc / h * sx;
        circle(imgX0 + nuc.i * sx, imgY0 + nuc.j * sy, rcPx * 2);
      }
    }

    fill(0); noStroke();
    for (const nuc of nuclei) {
      circle(imgX0 + nuc.i * sx, imgY0 + nuc.j * sy, 8);
    }

    stroke(255, 255, 0, 120); strokeWeight(1);
    line(imgX0, imgY0 + N2 * sy, imgX0 + imgW, imgY0 + N2 * sy);
    noStroke();

    for (let i = 0; i < S; i++) {
      const idx = (N2 * S + i) * 7;
      const u1v = sliceData[idx];
      const u2v = sliceData[idx + 3];
      const w1v = sliceData[idx + 1];
      const w2v = sliceData[idx + 4];
      const P1v = sliceData[idx + 2];
      const P2v = sliceData[idx + 5];
      const Kv = sliceData[idx + 6];
      const x = plotX0 + i * (plotW / S);

      fill(255, 50, 50); noStroke();
      ellipse(x, baseW - plotH_w * w1v, 3);
      fill(50, 50, 255);
      ellipse(x, baseW - plotH_w * w2v, 3);

      fill(255, 50, 50);
      ellipse(x, baseU - plotH_u * u1v * w1v, 3);
      fill(50, 50, 255);
      ellipse(x, baseU - plotH_u * u2v * w2v, 3);

      fill(255, 50, 50, 180);
      ellipse(x, baseP - plotH_P * P1v, 2);
      fill(50, 50, 255, 180);
      ellipse(x, baseP - plotH_P * P2v, 2);
      fill(0, 180, 0);
      ellipse(x, baseP - 30 * Kv, 3);
    }
  }

  let sliceMax1 = 0, sliceMax2 = 0;
  if (sliceData) {
    for (let i = 0; i < S; i++) {
      const idx = (N2 * S + i) * 7;
      sliceMax1 = Math.max(sliceMax1, Math.abs(sliceData[idx] * sliceData[idx + 1]));
      sliceMax2 = Math.max(sliceMax2, Math.abs(sliceData[idx + 3] * sliceData[idx + 4]));
    }
  }

  const rx = imgX0 + imgW + 15;
  fill(0); textSize(13);
  text("H\u2082 RealQM GPU", rx, 20);
  text(NN + "\u00B3 / " + screenAu + " au", rx, 38);
  text("R = " + (D_CELLS * h).toFixed(2) + " au", rx, 56);
  const rcStr = nuclei.some(n => n.rc > 0) ? "  r_c=" + nuclei[0].rc : "";
  text("step " + stepCount + rcStr, rx, 74);
  text("|wu1|=" + sliceMax1.toFixed(4) + " |wu2|=" + sliceMax2.toFixed(4), rx, 92);

  textSize(12);
  text("T = " + E_T.toFixed(4), rx, 110);
  text("V_eK = " + E_eK.toFixed(4), rx, 126);
  text("V_ee = " + E_ee.toFixed(4), rx, 142);
  text("V_KK = " + V_KK.toFixed(4), rx, 158);
  textSize(13);
  text("E = " + E_tot.toFixed(4), rx, 180);
  textSize(11);
  text("(ref -1.17 Ha)", rx, 196);

  fill(0); textSize(12);
  text("w (level sets)", plotX0 + 5, baseW - plotH_w - 4);
  fill(255, 50, 50); text("w\u2081", plotX0 + 110, baseW - plotH_w - 4);
  fill(50, 50, 255); text("w\u2082", plotX0 + 135, baseW - plotH_w - 4);

  fill(0); text("u (wavefunctions)", plotX0 + 5, baseU - plotH_u - 4);
  fill(255, 50, 50); text("u\u2081", plotX0 + 140, baseU - plotH_u - 4);
  fill(50, 50, 255); text("u\u2082", plotX0 + 160, baseU - plotH_u - 4);

  fill(0); text("P (Poisson), K (nuclear)", plotX0 + 5, baseP - plotH_P - 4);

  if (stepCount > 0 && stepCount % 100 === 0 && stepCount !== lastLogStep) {
    lastLogStep = stepCount;
    console.log("step=" + stepCount + " T=" + E_T.toFixed(4) + " V_eK=" + E_eK.toFixed(4) +
      " V_ee=" + E_ee.toFixed(4) + " V_KK=" + V_KK.toFixed(4) + " E=" + E_tot.toFixed(4));
  }
};
