// H2O (Water) Quantum Simulation — WebGPU Compute Shaders
// H(red,Z=1) O(green,Z=8→2) H(blue,Z=1) triangular molecule

const NN = 200;
const S = NN + 1;
const S2 = S * S;
const S3 = S * S * S;
const N2 = Math.round(NN / 2);
const D_bond = 40;   // 4 au between adjacent atoms (at 20 au screen)
const r_cut = window.USER_RC || [0.1, 0.6, 0.1];   // au, inner w cutoff per nucleus
let R_out = 2.0;   // au, outer w cutoff
const NELEC = 3;
const NRED = 6;  // 3 norms + T + V_eK + V_ee
const _uz = window.USER_Z || [1, 2, 1];
let Z = [..._uz];
let Ne = [..._uz];    // electron occupation = charge
const Z_orig = [..._uz];
const Ne_orig = [..._uz];

// Triangle in i-j plane: Green at apex, Red-Green-Blue angle
const bondAngle = (window.USER_ANGLE || 90) * Math.PI / 180;
function triPos(db) {
  // Green (index 1) at apex, Red (0) and Blue (2) as legs
  const halfA = bondAngle / 2;
  const gi = N2, gj = N2 - Math.round(db * Math.cos(halfA) / 3);
  const ri = N2 - Math.round(db * Math.sin(halfA)), rj = N2 + Math.round(2 * db * Math.cos(halfA) / 3);
  const bi = N2 + Math.round(db * Math.sin(halfA)), bj = rj;
  return [[ri, rj, N2], [gi, gj, N2], [bi, bj, N2]];
}
const rawPos = triPos(D_bond);
const comI = Math.round((rawPos[0][0] + rawPos[1][0] + rawPos[2][0]) / 3);
const comJ = Math.round((rawPos[0][1] + rawPos[1][1] + rawPos[2][1]) / 3);
let nucPos = rawPos.map(p => [p[0] + N2 - comI, p[1] + N2 - comJ, N2]);
const molNucPos = nucPos.map(p => [...p]);  // save molecule positions

let E_min = Infinity;
let screenAu = 20;
let hv = screenAu / NN, h2v = hv * hv, h3v = hv * hv * hv;
const dv = 0.12;
let dtv = dv * h2v, half_dv = 0.5 * dv;
const PX = 400 / NN;
const INTERIOR = (NN - 1) * (NN - 1) * (NN - 1);
const STEPS_PER_FRAME = 500;
const NORM_INTERVAL = 20;

// ===== WGSL SHADERS =====

const paramStructWGSL = `
struct P {
  NN: u32, S: u32, S2: u32, S3: u32,
  N2: u32, h0I: u32, h1I: u32, h2I: u32,
  h: f32, h2: f32, inv_h: f32, inv_h2: f32,
  dt: f32, half_d: f32, _pad0: f32, TWO_PI: f32,
  h3: f32, Z0: f32, Z1: f32, Z2: f32,
  h0J: u32, h1J: u32, h2J: u32, R_out: f32,
  rc0: f32, rc1: f32, rc2: f32, voronoi: f32
}`;

const updateWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> Ui: array<f32>;
@group(0) @binding(3) var<storage, read> Wi: array<f32>;
@group(0) @binding(4) var<storage, read> Pi: array<f32>;
@group(0) @binding(5) var<storage, read_write> Uo: array<f32>;
@group(0) @binding(6) var<storage, read_write> Wo: array<f32>;
@group(0) @binding(7) var<storage, read_write> Po: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }

  let m = gid.y;
  let o = m * p.S3;

  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  var cm: f32 = 0.0;
  cm -= Ui[0u * p.S3 + id] + Ui[1u * p.S3 + id] + Ui[2u * p.S3 + id];
  cm += Ui[o + id];
  cm = 0.5 * (cm + Ui[o + id]);

  let wc  = Wi[o + id];
  let wip = Wi[o + id + p.S2]; let wim = Wi[o + id - p.S2];
  let wjp = Wi[o + id + p.S];  let wjm = Wi[o + id - p.S];
  let wkp = Wi[o + id + 1u];   let wkm = Wi[o + id - 1u];

  let lw = (wip + wim + wjp + wjm + wkp + wkm - 6.0 * wc) * p.inv_h2;
  let gx = (wip - wim) * p.inv_h;
  let gy = (wjp - wjm) * p.inv_h;
  let gz = (wkp - wkm) * p.inv_h;
  var nw = wc + 0.25 * p.dt * abs(cm) * lw + 5.0 * p.dt * cm * sqrt(gx * gx + gy * gy + gz * gz);
  nw = clamp(nw, 0.0, 1.0);

  let dkk = f32(k) - f32(p.N2);

  let d0i = f32(i) - f32(p.h0I); let d0j = f32(j) - f32(p.h0J);
  let r0 = sqrt(d0i*d0i + d0j*d0j + dkk*dkk) * p.h;
  if (r0 < p.rc0) {
    let edge0 = p.rc0 - 3.0 * p.h;
    let t = clamp((r0 - edge0) / (p.rc0 - edge0), 0.0, 1.0);
    nw = min(nw, t * t * (3.0 - 2.0 * t));
  }

  let d1i = f32(i) - f32(p.h1I); let d1j = f32(j) - f32(p.h1J);
  let r1 = sqrt(d1i*d1i + d1j*d1j + dkk*dkk) * p.h;
  if (r1 < p.rc1) {
    let edge1 = p.rc1 - 3.0 * p.h;
    let t = clamp((r1 - edge1) / (p.rc1 - edge1), 0.0, 1.0);
    nw = min(nw, t * t * (3.0 - 2.0 * t));
  }

  let d2i = f32(i) - f32(p.h2I); let d2j = f32(j) - f32(p.h2J);
  let r2 = sqrt(d2i*d2i + d2j*d2j + dkk*dkk) * p.h;
  if (r2 < p.rc2) {
    let edge2 = p.rc2 - 3.0 * p.h;
    let t = clamp((r2 - edge2) / (p.rc2 - edge2), 0.0, 1.0);
    nw = min(nw, t * t * (3.0 - 2.0 * t));
  }

  // Voronoi restriction: w=0 if point is closer to another nucleus
  if (p.voronoi > 0.5) {
    var rm: f32;
    if (m == 0u) { rm = r0; } else if (m == 1u) { rm = r1; } else { rm = r2; }
    if ((m != 0u && r0 < rm) || (m != 1u && r1 < rm) || (m != 2u && r2 < rm)) { nw = 0.0; }
  }

  Wo[o + id] = nw;

  let uc  = Ui[o + id];
  let uip = Ui[o + id + p.S2]; let uim = Ui[o + id - p.S2];
  let ujp = Ui[o + id + p.S];  let ujm = Ui[o + id - p.S];
  let ukp = Ui[o + id + 1u];   let ukm = Ui[o + id - 1u];

  Uo[o + id] = uc
    + p.half_d * ((uip - uc) * (wip + nw) * 0.5 - (uc - uim) * (nw + wim) * 0.5)
    + p.half_d * ((ujp - uc) * (wjp + nw) * 0.5 - (uc - ujm) * (nw + wjm) * 0.5)
    + p.half_d * ((ukp - uc) * (wkp + nw) * 0.5 - (uc - ukm) * (nw + wkm) * 0.5)
    + p.dt * (K[id] - 2.0 * Pi[o + id]) * uc * wc;

  let Pc = Pi[o + id];
  let u0 = Ui[0u * p.S3 + id];
  let u1 = Ui[1u * p.S3 + id];
  let u2 = Ui[2u * p.S3 + id];
  var rho: f32 = p.Z0 * u0*u0 + p.Z1 * u1*u1 + p.Z2 * u2*u2;
  let self_u = Ui[o + id];
  rho -= self_u * self_u;

  Po[o + id] = Pc
    + p.dt * (Pi[o + id + p.S2] + Pi[o + id - p.S2]
            + Pi[o + id + p.S]  + Pi[o + id - p.S]
            + Pi[o + id + 1u]   + Pi[o + id - 1u]
            - 6.0 * Pc) * p.inv_h2
    + p.TWO_PI * p.dt * rho;
}
`;

const reduceWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> Pv: array<f32>;
@group(0) @binding(4) var<storage, read> K: array<f32>;
@group(0) @binding(5) var<storage, read_write> partials: array<f32>;

var<workgroup> sn: array<f32, 768>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(workgroup_id) wgid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;

  for (var x: u32 = 0u; x < 6u; x++) { sn[lid * 6u + x] = 0.0; }

  if (gid.x < tot) {
    let k = (gid.x % NM) + 1u;
    let j = ((gid.x / NM) % NM) + 1u;
    let i = (gid.x / (NM * NM)) + 1u;
    let id = i * p.S2 + j * p.S + k;

    var T: f32 = 0.0;
    var VeK: f32 = 0.0;
    var Vee: f32 = 0.0;
    for (var m: u32 = 0u; m < 3u; m++) {
      let o = m * p.S3;
      let v = U[o + id];
      var Zm: f32;
      if (m == 0u) { Zm = p.Z0; } else if (m == 1u) { Zm = p.Z1; } else { Zm = p.Z2; }
      sn[lid * 6u + m] = v * v * p.h3;
      if (W[o + id] > 0.1) {
        let a = U[o + id + p.S2] - v;
        let b = U[o + id + p.S] - v;
        let c = U[o + id + 1u] - v;
        T += Zm * 0.5 * (a * a + b * b + c * c) * p.h;
      }
      VeK -= Zm * K[id] * v * v * p.h3;
      Vee += Zm * Pv[o + id] * v * v * p.h3;
    }
    sn[lid * 6u + 3u] = T;
    sn[lid * 6u + 4u] = VeK;
    sn[lid * 6u + 5u] = Vee;
  }

  workgroupBarrier();

  for (var s: u32 = 64u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var x: u32 = 0u; x < 6u; x++) {
        sn[lid * 6u + x] += sn[(lid + s) * 6u + x];
      }
    }
    workgroupBarrier();
  }

  if (lid == 0u) {
    let base = wgid.x * 6u;
    for (var x: u32 = 0u; x < 6u; x++) {
      partials[base + x] = sn[x];
    }
  }
}
`;

const finalizeWGSL = `
struct NWG { count: u32 }
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> sums: array<f32>;
@group(0) @binding(2) var<uniform> nwg: NWG;

var<workgroup> wg: array<f32, 768>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lid: u32) {
  for (var x: u32 = 0u; x < 6u; x++) { wg[lid * 6u + x] = 0.0; }

  for (var i: u32 = lid; i < nwg.count; i += 128u) {
    for (var x: u32 = 0u; x < 6u; x++) {
      wg[lid * 6u + x] += partials[i * 6u + x];
    }
  }

  workgroupBarrier();

  for (var s: u32 = 64u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var x: u32 = 0u; x < 6u; x++) {
        wg[lid * 6u + x] += wg[(lid + s) * 6u + x];
      }
    }
    workgroupBarrier();
  }

  if (lid == 0u) {
    for (var x: u32 = 0u; x < 6u; x++) {
      sums[x] = wg[x];
    }
  }
}
`;

const normalizeWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> U: array<f32>;
@group(0) @binding(2) var<storage, read> sums: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (g.x >= tot) { return; }

  let k = (g.x % NM) + 1u;
  let j = ((g.x / NM) % NM) + 1u;
  let i = (g.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  for (var m: u32 = 0u; m < 3u; m++) {
    let n = sums[m];
    if (n > 0.0) { U[m * p.S3 + id] *= inverseSqrt(n); }
  }
}
`;

const extractWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> Pv: array<f32>;
@group(0) @binding(4) var<storage, read> K: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  let j = g.y;
  let SS = p.NN + 1u;
  if (i > p.NN || j > p.NN) { return; }

  for (var m: u32 = 0u; m < 3u; m++) {
    let idx = m * p.S3 + i * p.S2 + j * p.S + p.N2;
    out[m * SS * SS + i * SS + j] = select(0.0, U[idx], W[idx] > 0.0);
  }

  if (j == 0u) {
    let b = 3u * SS * SS;
    for (var m: u32 = 0u; m < 3u; m++) {
      out[b + m * SS + i] = W[m * p.S3 + i * p.S2 + (p.N2 + 8u) * p.S + p.N2];
      let uIdx = m * p.S3 + i * p.S2 + (p.N2 + 5u) * p.S + p.N2;
      out[b + 3u * SS + m * SS + i] = U[uIdx] * W[uIdx];
      out[b + 6u * SS + m * SS + i] = Pv[m * p.S3 + i * p.S2 + p.N2 * p.S + p.N2];
    }
    out[b + 9u * SS + i] = K[i * p.S2 + p.N2 * p.S + p.N2];
  }
}
`;

// ===== GPU STATE =====
let device, paramsBuf, K_buf, sumsBuf, sumsReadBuf, sliceBuf, sliceReadBuf, partialsBuf, numWGBuf;
let U_buf = [], W_buf = [], P_buf = [];
let updatePL, reducePL, finalizePL, normalizePL, extractPL;
let updateBG = [], reduceBG = [], finalizeBG, normalizeBG = [], extractBG = [];
let cur = 0, gpuReady = false, computing = false;
let tStep = 0, E = 0, lastMs = 0;
let E_T = 0, E_eK = 0, E_ee = 0, E_KK = 0;
let gpuError = null;

// Phase system: sweep over distances
const D_SWEEP = [2, 6];  // au
const D_SCREEN = [10, 20];  // au, per-phase screen size
let phase = 0, phaseSteps = 0;
const PHASE_STEPS = [20000, 10000];
let E_sweep = [];
let addNucRepulsion = true;
let curD_bond = D_bond;

const SLICE_SIZE = (3 * S * S + 10 * S) * 4;
const WG_UPDATE = Math.ceil(INTERIOR / 256);
const WG_REDUCE = Math.ceil(INTERIOR / 128);
const WG_NORM = Math.ceil(INTERIOR / 256);
const WG_EXTRACT = Math.ceil(S / 16);
const SUMS_BYTES = NRED * 4;

let sliceData = null;

function smoothCut(r, rc) {
  if (r >= rc) return 1;
  const edge = rc - 3 * hv;
  const t = Math.max(0, Math.min(1, (r - edge) / (rc - edge)));
  return t * t * (3 - 2 * t);
}

function uploadInitialData() {
  console.log("Init: k0=(" + nucPos[0] + ") k1=(" + nucPos[1] + ") k2=(" + nucPos[2] + ")");

  const Kd = new Float32Array(S3);
  const Ud = new Float32Array(NELEC * S3);
  const Wd = new Float32Array(NELEC * S3);
  const Pd = new Float32Array(NELEC * S3);
  const soft = 0.04 * h2v;

  for (let i = 0; i <= NN; i++) {
    const dx = [];
    for (let n = 0; n < 3; n++) dx[n] = (i - nucPos[n][0]) * hv;
    for (let j = 0; j <= NN; j++) {
      const dy = [];
      for (let n = 0; n < 3; n++) dy[n] = (j - nucPos[n][1]) * hv;
      for (let k = 0; k <= NN; k++) {
        const dz = [];
        for (let n = 0; n < 3; n++) dz[n] = (k - nucPos[n][2]) * hv;
        const id = i * S2 + j * S + k;

        const r = [], ir = [], u = [];
        for (let n = 0; n < 3; n++) {
          r[n] = Math.sqrt(dx[n]*dx[n] + dy[n]*dy[n] + dz[n]*dz[n] + soft);
          ir[n] = 1 / r[n];
          u[n] = Math.exp(-Z[n] * r[n]);
        }

        Kd[id] = Z[0]*ir[0] + Z[1]*ir[1] + Z[2]*ir[2];

        let best = -1;
        for (let n = 0; n < 3; n++) {
          if (Z[n] > 0 && (best < 0 || u[n] > u[best])) best = n;
        }
        if (best >= 0) {
          const w = r[best] <= R_out ? smoothCut(r[best], r_cut[best]) : 0;
          Ud[best*S3+id] = u[best];
          Wd[best*S3+id] = w;
        }

        const pAvg = (Z[0]*ir[0] + Z[1]*ir[1] + Z[2]*ir[2]) / 3;
        for (let m = 0; m < NELEC; m++) {
          Pd[m*S3+id] = pAvg;
        }
      }
    }
  }

  console.log("Uploading to GPU...");
  device.queue.writeBuffer(K_buf, 0, Kd);
  for (let i = 0; i < 2; i++) {
    device.queue.writeBuffer(U_buf[i], 0, Ud);
    device.queue.writeBuffer(W_buf[i], 0, Wd);
    device.queue.writeBuffer(P_buf[i], 0, Pd);
  }
  cur = 0;
}

function updateParamsBuf() {
  const pb = new ArrayBuffer(112);
  const pu = new Uint32Array(pb);
  const pf = new Float32Array(pb);
  pu[0] = NN; pu[1] = S; pu[2] = S2; pu[3] = S3;
  pu[4] = N2; pu[5] = nucPos[0][0]; pu[6] = nucPos[1][0]; pu[7] = nucPos[2][0];
  pf[8] = hv; pf[9] = h2v; pf[10] = 1 / hv; pf[11] = 1 / h2v;
  pf[12] = dtv; pf[13] = half_dv; pf[14] = 0; pf[15] = 2 * Math.PI;
  pf[16] = h3v; pf[17] = Z[0]; pf[18] = Z[1]; pf[19] = Z[2];
  pu[20] = nucPos[0][1]; pu[21] = nucPos[1][1]; pu[22] = nucPos[2][1]; pf[23] = R_out;
  pf[24] = r_cut[0]; pf[25] = r_cut[1]; pf[26] = r_cut[2]; pf[27] = 1.0;
  device.queue.writeBuffer(paramsBuf, 0, pb);
}

function startMolPhase(dist_au, phaseNum, screen) {
  if (screen) {
    screenAu = screen;
    hv = screenAu / NN; h2v = hv * hv; h3v = hv * hv * hv;
    dtv = dv * h2v; half_dv = 0.5 * dv;
  }
  const d_bond = Math.round(dist_au / hv);
  curD_bond = d_bond;
  const rp = triPos(d_bond);
  const ci = Math.round((rp[0][0] + rp[1][0] + rp[2][0]) / 3);
  const cj = Math.round((rp[0][1] + rp[1][1] + rp[2][1]) / 3);
  nucPos = rp.map(p => [p[0] + N2 - ci, p[1] + N2 - cj, N2]);
  Z = [...Z_orig]; Ne = [...Ne_orig];
  R_out = 2.0;
  addNucRepulsion = true;
  updateParamsBuf();
  uploadInitialData();
  tStep = 0; phaseSteps = 0; E_min = Infinity; cur = 0;
  phase = phaseNum;
  console.log("=== MOLECULE D=" + d_bond + " (" + (d_bond*hv).toFixed(2) + " au): phase " + phase + " ===");
}

function startAtomPhase(atomIdx) {
  // Single atom at center, only active field has Z>0
  nucPos = [[N2, N2, N2], [N2, N2, N2], [N2, N2, N2]];
  Z = [0, 0, 0];
  Ne = [0, 0, 0];
  Z[atomIdx] = Z_orig[atomIdx];
  Ne[atomIdx] = Ne_orig[atomIdx];
  R_out = 2.0;
  addNucRepulsion = false;
  updateParamsBuf();
  uploadInitialData();
  tStep = 0;
  phaseSteps = 0;
  E_min = Infinity;
  cur = 0;
  phase = atomIdx + 1;
  console.log("=== ATOM " + atomIdx + " (Z=" + Z_orig[atomIdx] + "): phase " + phase + " ===");
}

function setup() {
  createCanvas(400, 400);
  textSize(9);
  initGPU();
}

async function initGPU() {
  try {
    if (!navigator.gpu) {
      gpuError = "WebGPU not supported. Use Chrome 113+ or Safari 17+.";
      return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) { gpuError = "No GPU adapter found."; return; }

    try {
      const info = await adapter.requestAdapterInfo();
      console.log("GPU:", info.vendor, info.architecture, info.description);
    } catch (e) {}

    device = await adapter.requestDevice();
    console.log("WebGPU device ready");

    device.lost.then((info) => {
      gpuError = "GPU device lost: " + info.message;
      gpuReady = false;
    });

    const bs = S3 * 4, bN = NELEC * S3 * 4;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    K_buf = device.createBuffer({ size: bs, usage });
    for (let i = 0; i < 2; i++) {
      U_buf[i] = device.createBuffer({ size: bN, usage: usage | GPUBufferUsage.COPY_SRC });
      W_buf[i] = device.createBuffer({ size: bN, usage: usage | GPUBufferUsage.COPY_SRC });
      P_buf[i] = device.createBuffer({ size: bN, usage: usage | GPUBufferUsage.COPY_SRC });
    }

    const partialSize = WG_REDUCE * NRED * 4;
    partialsBuf = device.createBuffer({ size: partialSize, usage: GPUBufferUsage.STORAGE });
    sumsBuf = device.createBuffer({ size: SUMS_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    sumsReadBuf = device.createBuffer({ size: SUMS_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    sliceBuf = device.createBuffer({ size: SLICE_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    sliceReadBuf = device.createBuffer({ size: SLICE_SIZE, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    numWGBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(numWGBuf, 0, new Uint32Array([WG_REDUCE, 0, 0, 0]));

    const pb = new ArrayBuffer(112);
    const pu = new Uint32Array(pb);
    const pf = new Float32Array(pb);
    pu[0] = NN; pu[1] = S; pu[2] = S2; pu[3] = S3;
    pu[4] = N2; pu[5] = nucPos[0][0]; pu[6] = nucPos[1][0]; pu[7] = nucPos[2][0];
    pf[8] = hv; pf[9] = h2v; pf[10] = 1 / hv; pf[11] = 1 / h2v;
    pf[12] = dtv; pf[13] = half_dv; pf[14] = 0; pf[15] = 2 * Math.PI;
    pf[16] = h3v; pf[17] = Z[0]; pf[18] = Z[1]; pf[19] = Z[2];
    pu[20] = nucPos[0][1]; pu[21] = nucPos[1][1]; pu[22] = nucPos[2][1]; pf[23] = R_out;
    pf[24] = r_cut[0]; pf[25] = r_cut[1]; pf[26] = r_cut[2]; pf[27] = 1.0;
    paramsBuf = device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuf, 0, pb);

    uploadInitialData();

    async function compileShader(name, code) {
      const module = device.createShaderModule({ code });
      try {
        const getInfo = module.getCompilationInfo || module.compilationInfo;
        if (getInfo) {
          const info = await getInfo.call(module);
          for (const msg of info.messages) {
            if (msg.type === 'error') {
              throw new Error("Shader '" + name + "': " + msg.message + " (line " + msg.lineNum + ")");
            }
          }
        }
      } catch (e) {
        if (e.message.startsWith("Shader '")) throw e;
      }
      console.log("Shader '" + name + "' OK");
      return module;
    }

    const updateMod = await compileShader('update', updateWGSL);
    const reduceMod = await compileShader('reduce', reduceWGSL);
    const finalizeMod = await compileShader('finalize', finalizeWGSL);
    const normalizeMod = await compileShader('normalize', normalizeWGSL);
    const extractMod = await compileShader('extract', extractWGSL);

    updatePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: updateMod, entryPoint: 'main' } });
    reducePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: reduceMod, entryPoint: 'main' } });
    finalizePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: finalizeMod, entryPoint: 'main' } });
    normalizePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: normalizeMod, entryPoint: 'main' } });
    extractPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: extractMod, entryPoint: 'main' } });

    for (let c = 0; c < 2; c++) {
      const n = 1 - c;
      updateBG[c] = device.createBindGroup({ layout: updatePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: K_buf } },
        { binding: 2, resource: { buffer: U_buf[c] } },
        { binding: 3, resource: { buffer: W_buf[c] } },
        { binding: 4, resource: { buffer: P_buf[c] } },
        { binding: 5, resource: { buffer: U_buf[n] } },
        { binding: 6, resource: { buffer: W_buf[n] } },
        { binding: 7, resource: { buffer: P_buf[n] } },
      ]});
      reduceBG[c] = device.createBindGroup({ layout: reducePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: W_buf[c] } },
        { binding: 3, resource: { buffer: P_buf[c] } },
        { binding: 4, resource: { buffer: K_buf } },
        { binding: 5, resource: { buffer: partialsBuf } },
      ]});
      normalizeBG[c] = device.createBindGroup({ layout: normalizePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: sumsBuf } },
      ]});
      extractBG[c] = device.createBindGroup({ layout: extractPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: W_buf[c] } },
        { binding: 3, resource: { buffer: P_buf[c] } },
        { binding: 4, resource: { buffer: K_buf } },
        { binding: 5, resource: { buffer: sliceBuf } },
      ]});
    }

    finalizeBG = device.createBindGroup({ layout: finalizePL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: partialsBuf } },
      { binding: 1, resource: { buffer: sumsBuf } },
      { binding: 2, resource: { buffer: numWGBuf } },
    ]});

    console.log("Ready! dispatch(" + WG_UPDATE + "," + NELEC + ",1)");
    gpuReady = true;
    // Start first sweep distance
    startMolPhase(D_SWEEP[0], 0, D_SCREEN[0]);

  } catch (e) {
    gpuError = e.message || String(e);
    console.error("GPU init failed:", e);
  }
}

async function doSteps(n) {
  const t0 = performance.now();
  const enc = device.createCommandEncoder();

  for (let s = 0; s < n; s++) {
    const next = 1 - cur;

    let cp = enc.beginComputePass();
    cp.setPipeline(updatePL);
    cp.setBindGroup(0, updateBG[cur]);
    cp.dispatchWorkgroups(WG_UPDATE, NELEC, 1);
    cp.end();

    if ((s + 1) % NORM_INTERVAL === 0 || s === n - 1) {
      cp = enc.beginComputePass();
      cp.setPipeline(reducePL);
      cp.setBindGroup(0, reduceBG[next]);
      cp.dispatchWorkgroups(WG_REDUCE);
      cp.end();

      cp = enc.beginComputePass();
      cp.setPipeline(finalizePL);
      cp.setBindGroup(0, finalizeBG);
      cp.dispatchWorkgroups(1);
      cp.end();

      cp = enc.beginComputePass();
      cp.setPipeline(normalizePL);
      cp.setBindGroup(0, normalizeBG[next]);
      cp.dispatchWorkgroups(WG_NORM);
      cp.end();
    }

    cur = next;
  }

  let cp = enc.beginComputePass();
  cp.setPipeline(extractPL);
  cp.setBindGroup(0, extractBG[cur]);
  cp.dispatchWorkgroups(WG_EXTRACT, WG_EXTRACT);
  cp.end();

  enc.copyBufferToBuffer(sumsBuf, 0, sumsReadBuf, 0, SUMS_BYTES);
  enc.copyBufferToBuffer(sliceBuf, 0, sliceReadBuf, 0, SLICE_SIZE);
  device.queue.submit([enc.finish()]);

  await sumsReadBuf.mapAsync(GPUMapMode.READ);
  const sumsData = new Float32Array(sumsReadBuf.getMappedRange().slice(0));
  sumsReadBuf.unmap();
  E_T = sumsData[3];
  E_eK = sumsData[4];
  E_ee = sumsData[5];

  await sliceReadBuf.mapAsync(GPUMapMode.READ);
  sliceData = new Float32Array(sliceReadBuf.getMappedRange().slice(0));
  sliceReadBuf.unmap();

  tStep += n;
  lastMs = performance.now() - t0;

  E_KK = 0;
  if (addNucRepulsion) {
    const soft_nuc = 0.04 * h2v;
    for (let a = 0; a < 3; a++) {
      for (let b = a + 1; b < 3; b++) {
        const d = Math.sqrt(
          ((nucPos[a][0]-nucPos[b][0])*hv)**2 +
          ((nucPos[a][1]-nucPos[b][1])*hv)**2 +
          ((nucPos[a][2]-nucPos[b][2])*hv)**2 + soft_nuc);
        E_KK += Z[a]*Z[b]/d;
      }
    }
  }
  E = E_T + E_eK + E_ee + E_KK;

  if (!isFinite(E)) {
    gpuError = "Numerical instability at step " + tStep;
    return;
  }

  console.log("Step " + tStep + ": E=" + E.toFixed(6) + " (" + lastMs.toFixed(0) + "ms/" + n + "steps)");
}

function draw() {
  background(0);
  noStroke();

  if (gpuError) {
    fill(200, 0, 0);
    textSize(11);
    text("GPU Error:", 10, 180);
    textSize(9);
    const lines = gpuError.match(/.{1,55}/g) || [gpuError];
    for (let i = 0; i < Math.min(lines.length, 8); i++) {
      text(lines[i], 10, 198 + i * 14);
    }
    return;
  }

  if (!gpuReady) {
    fill(255);
    text("Initializing WebGPU...", 10, 200);
    return;
  }

  if (!computing && phase < D_SWEEP.length) {
    computing = true;
    doSteps(STEPS_PER_FRAME).then(() => {
      computing = false;
      phaseSteps += STEPS_PER_FRAME;
      if (phaseSteps >= 5000) {
        device.queue.writeBuffer(paramsBuf, 27 * 4, new Float32Array([0.0]));
      }
      if (isFinite(E) && E < E_min) E_min = E;

      if (phaseSteps >= PHASE_STEPS[phase]) {
        E_sweep[phase] = { d: D_SWEEP[phase], E: E, T: E_T, VeK: E_eK, Vee: E_ee, VKK: E_KK };
        console.log("=== D=" + D_SWEEP[phase] + " au DONE: E=" + E.toFixed(6) + " ===");
        if (phase + 1 < D_SWEEP.length) {
          const nextD = D_SWEEP[phase + 1];
          startMolPhase(nextD, phase + 1, D_SCREEN[phase + 1]);
        } else {
          phase = D_SWEEP.length;  // done
        }
      }
    }).catch((e) => {
      gpuError = e.message || String(e);
      console.error("GPU step failed:", e);
      computing = false;
    });
  }

  if (sliceData) {
    const SS = S;
    loadPixels();
    const d = pixelDensity();
    const W = 400 * d, H = 400 * d;
    for (let p = 0; p < W * H * 4; p += 4) {
      pixels[p] = 0; pixels[p+1] = 0; pixels[p+2] = 0; pixels[p+3] = 255;
    }
    for (let i = 1; i < NN; i++) {
      const px0 = Math.floor(PX * i * d);
      const px1 = Math.floor(PX * (i + 1) * d);
      for (let j = 1; j < NN; j++) {
        const py0 = Math.floor(PX * j * d);
        const py1 = Math.floor(PX * (j + 1) * d);
        const b = i * SS + j;
        // 3 electrons: red, green, blue
        const e0 = 500 * sliceData[0 * SS * SS + b];
        const e1 = 500 * sliceData[1 * SS * SS + b];
        const e2 = 500 * sliceData[2 * SS * SS + b];
        const ri = Math.min(255, Math.floor(e0));
        const gi = Math.min(255, Math.floor(e1));
        const bi = Math.min(255, Math.floor(e2));
        for (let py = py0; py < py1 && py < H; py++) {
          for (let px = px0; px < px1 && px < W; px++) {
            const idx = (py * W + px) * 4;
            pixels[idx] = ri;
            pixels[idx + 1] = gi;
            pixels[idx + 2] = bi;
            pixels[idx + 3] = 255;
          }
        }
      }
    }
    updatePixels();

    // Line plots
    const lb = 3 * SS * SS;
    const colors = [[255,0,0],[0,255,0],[0,100,255]];
    for (let i = 1; i < NN - 10; i++) {
      for (let m = 0; m < 3; m++) {
        fill(255); ellipse(PX * i, 300 - 100 * sliceData[lb + m * SS + i], 2);
        fill(colors[m][0], colors[m][1], colors[m][2]);
        ellipse(PX * i, 300 - 100 * sliceData[lb + 3 * SS + m * SS + i], 3);
        fill(0, 255, 255, 200); ellipse(PX * i, 300 - 30 * sliceData[lb + 6 * SS + m * SS + i], 2);
      }
      fill(0, 0, 255, 200); ellipse(PX * i, 300 - 30 * sliceData[lb + 9 * SS + i], 2);
    }
  }

  // Draw nuclear positions
  fill(255); stroke(255); strokeWeight(1);
  for (let n = 0; n < 3; n++) {
    if (Z[n] > 0) circle(nucPos[n][0] * PX, nucPos[n][1] * PX, 6);
  }
  // Screen boundary
  noFill(); stroke(100); strokeWeight(1);
  rect(0, 0, 400, 400);
  noStroke();

  fill(255);
  const pLabel = phase < D_SWEEP.length ? "D=" + D_SWEEP[phase] + " au (" + screenAu + " au)" : "DONE";
  text("H2O Water | " + pLabel + " | H(red)=" + Z_orig[0] + " O(green)=" + Z_orig[1] + " H(blue)=" + Z_orig[2] + " | " + NN + "^3", 5, 20);
  text("step " + tStep + " (" + phaseSteps + "/" + (PHASE_STEPS[phase] || "done") + ")  E=" + E.toFixed(6), 5, 35);
  if (lastMs > 0) text((lastMs / STEPS_PER_FRAME).toFixed(1) + "ms/step", 300, 35);

  fill(200);
  text("T=" + E_T.toFixed(4) + " V_eK=" + E_eK.toFixed(4) + " V_ee=" + E_ee.toFixed(4) + " V_KK=" + E_KK.toFixed(4), 5, 50);

  // Show completed sweep results
  let y = 65;
  for (let i = 0; i < E_sweep.length; i++) {
    const s = E_sweep[i];
    fill(255, 255, 0);
    text("D=" + s.d.toFixed(1) + "  E=" + s.E.toFixed(4) + "  T=" + s.T.toFixed(4) + "  VeK=" + s.VeK.toFixed(4) + "  Vee=" + s.Vee.toFixed(4) + "  VKK=" + s.VKK.toFixed(4), 5, y);
    y += 13;
  }
  if (E_sweep.length >= 2) {
    const Ebind = E_sweep[0].E - E_sweep[E_sweep.length - 1].E;
    fill(0, 255, 128);
    text("Binding energy: E(D=" + E_sweep[0].d.toFixed(1) + ") - E(D=" + E_sweep[E_sweep.length-1].d.toFixed(1) + ") = " + Ebind.toFixed(4) + " au = " + (Ebind * 27.211).toFixed(2) + " eV", 5, y);
  }
}
