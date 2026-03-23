// molecule3.js — Generalized N-atom Quantum Simulation — WebGPU Compute Shaders
// Supports arbitrary numbers of atoms via USER_ATOMS configuration
// Each atom: {i, j, k, Z, rc} where i,j,k are grid positions, Z is nuclear charge, rc is cutoff radius

// ===== USER CONFIGURATION (set before loading) =====
// USER_ATOMS: array of {i, j, k, Z, rc} — atom positions and charges
// USER_NN: grid size (default 200)
// USER_SCREEN: screening length R_out (default 2.0)
// USER_STEPS: steps per frame (default 500)
// USER_Z: array of nuclear charges (overrides USER_ATOMS Z if set)
// USER_RC: cutoff radius override (overrides USER_ATOMS rc if set)
// USER_Z_NUC: array of nuclear charges for Coulomb potential (for bare protons with Z=0 electron fields)
// USER_INIT_POS: array of {i, j, k} for electron initial positions (one per electron field)

const NN = (typeof USER_NN !== 'undefined') ? USER_NN : 200;
const S = NN + 1;
const S2 = S * S;
const S3 = S * S * S;
const N2 = Math.round(NN / 2);

// Parse atoms
if (typeof USER_ATOMS === 'undefined') {
  throw new Error("molecule3.js requires USER_ATOMS to be defined before loading");
}
const ATOMS = USER_ATOMS;
const NELEC = ATOMS.length;
const NRED = NELEC + 1;  // NELEC norms + 1 energy

// Nuclear charges for electron fields (occupation/normalization)
let Z = ATOMS.map(a => a.Z);
if (typeof USER_Z !== 'undefined') Z = USER_Z.slice();
const Z_orig = Z.slice();

// Nuclear charges for Coulomb potential (allows bare protons)
let Z_nuc = (typeof USER_Z_NUC !== 'undefined') ? USER_Z_NUC.slice() : Z.slice();
const Z_nuc_orig = Z_nuc.slice();

// Cutoff radii
const r_cut_arr = ATOMS.map(a => {
  if (typeof USER_RC !== 'undefined') return USER_RC;
  return a.rc !== undefined ? a.rc : 0.5;
});
const r_cut = r_cut_arr[0]; // uniform cutoff for param struct (used in w update edge calc)

let R_out = (typeof USER_SCREEN !== 'undefined') ? USER_SCREEN : 2.0;
const STEPS_PER_FRAME = (typeof USER_STEPS !== 'undefined') ? USER_STEPS : 500;

// Nuclear positions
let nucPos = ATOMS.map(a => [a.i, a.j, a.k !== undefined ? a.k : N2]);
const molNucPos = nucPos.map(p => [...p]);

// Electron init positions (default: same as nuclear positions)
let initPos = (typeof USER_INIT_POS !== 'undefined')
  ? USER_INIT_POS.map(p => [p.i, p.j, p.k !== undefined ? p.k : N2])
  : nucPos.map(p => [...p]);

let E_min = Infinity;
const hv = 10 / NN, h2v = hv * hv, h3v = hv * hv * hv;
const dv = 0.12;
const dtv = dv * h2v, half_dv = 0.5 * dv;
const PX = 400 / NN;
const INTERIOR = (NN - 1) * (NN - 1) * (NN - 1);
const NORM_INTERVAL = 20;

let E = 0, lastMs = 0, tStep = 0;
let addNucRepulsion = true;

// ===== WGSL SHADERS (generated dynamically) =====

// Param struct: fixed fields + atom positions in a storage buffer
const paramStructWGSL = `
struct P {
  NN: u32, S: u32, S2: u32, S3: u32,
  N2: u32, NELEC: u32, h: f32, h2: f32,
  inv_h: f32, inv_h2: f32, dt: f32, half_d: f32,
  r_cut: f32, FOUR_PI: f32, h3: f32, _pad: u32
}`;

// Atom struct for the atom buffer
const atomStructWGSL = `
struct Atom {
  posI: u32, posJ: u32, posK: u32, _pad0: u32,
  Z: f32, Z_nuc: f32, rc: f32, _pad1: f32
}`;

function generateUpdateWGSL() {
  return `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> Ui: array<f32>;
@group(0) @binding(3) var<storage, read> Wi: array<f32>;
@group(0) @binding(4) var<storage, read> Pi: array<f32>;
@group(0) @binding(5) var<storage, read_write> Uo: array<f32>;
@group(0) @binding(6) var<storage, read_write> Wo: array<f32>;
@group(0) @binding(7) var<storage, read_write> Po: array<f32>;
@group(0) @binding(8) var<storage, read> atoms: array<Atom>;

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

  // c = 0.5*(2*u_self - sum_all) formulation
  var cm: f32 = 0.0;
  for (var e: u32 = 0u; e < ${NELEC}u; e++) {
    cm -= Ui[e * p.S3 + id];
  }
  cm += Ui[o + id];
  cm = 0.5 * (cm + Ui[o + id]);

  // w update
  let wc  = Wi[o + id];
  let wip = Wi[o + id + p.S2]; let wim = Wi[o + id - p.S2];
  let wjp = Wi[o + id + p.S];  let wjm = Wi[o + id - p.S];
  let wkp = Wi[o + id + 1u];   let wkm = Wi[o + id - 1u];

  let lw = (wip + wim + wjp + wjm + wkp + wkm - 6.0 * wc) * p.inv_h2;
  let gx = (wip - wim) * p.inv_h;
  let gy = (wjp - wjm) * p.inv_h;
  let gz = (wkp - wkm) * p.inv_h;
  var nw = wc + 0.5 * p.dt * abs(cm) * lw + 5.0 * p.dt * cm * sqrt(gx * gx + gy * gy + gz * gz);
  nw = clamp(nw, 0.0, 1.0);

  // Smooth Hermite cutoff near each nucleus
  for (var a: u32 = 0u; a < ${NELEC}u; a++) {
    let rc_a = atoms[a].rc;
    let edge = rc_a - 3.0 * p.h;
    let dai = f32(i) - f32(atoms[a].posI);
    let daj = f32(j) - f32(atoms[a].posJ);
    let dak = f32(k) - f32(atoms[a].posK);
    let ra = sqrt(dai*dai + daj*daj + dak*dak) * p.h;
    if (ra < rc_a) {
      let t = clamp((ra - edge) / (rc_a - edge), 0.0, 1.0);
      nw = min(nw, t * t * (3.0 - 2.0 * t));
    }
  }
  Wo[o + id] = nw;

  // u update (fused diffusion)
  let uc  = Ui[o + id];
  let uip = Ui[o + id + p.S2]; let uim = Ui[o + id - p.S2];
  let ujp = Ui[o + id + p.S];  let ujm = Ui[o + id - p.S];
  let ukp = Ui[o + id + 1u];   let ukm = Ui[o + id - 1u];

  Uo[o + id] = uc
    + p.half_d * ((uip - uc) * (wip + nw) * 0.5 - (uc - uim) * (nw + wim) * 0.5)
    + p.half_d * ((ujp - uc) * (wjp + nw) * 0.5 - (uc - ujm) * (nw + wjm) * 0.5)
    + p.half_d * ((ukp - uc) * (wkp + nw) * 0.5 - (uc - ukm) * (nw + wkm) * 0.5)
    + p.dt * (K[id] - 2.0 * Pi[o + id]) * uc * wc;

  // Poisson update with 4*pi*rho
  let Pc = Pi[o + id];
  var rho: f32 = 0.0;
  for (var e2: u32 = 0u; e2 < ${NELEC}u; e2++) {
    let ue = Ui[e2 * p.S3 + id];
    rho += atoms[e2].Z * ue * ue;
  }
  let self_u = Ui[o + id];
  rho -= self_u * self_u;

  Po[o + id] = Pc
    + p.dt * (Pi[o + id + p.S2] + Pi[o + id - p.S2]
            + Pi[o + id + p.S]  + Pi[o + id - p.S]
            + Pi[o + id + 1u]   + Pi[o + id - 1u]
            - 6.0 * Pc) * p.inv_h2
    + p.FOUR_PI * p.dt * rho;
}
`;
}

function generateReduceWGSL() {
  return `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> Pv: array<f32>;
@group(0) @binding(4) var<storage, read> K: array<f32>;
@group(0) @binding(5) var<storage, read_write> partials: array<f32>;
@group(0) @binding(6) var<storage, read> atoms: array<Atom>;

var<workgroup> sn: array<f32, ${Math.max(1024, (NRED) * 256)}>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(workgroup_id) wgid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;

  for (var x: u32 = 0u; x < ${NRED}u; x++) { sn[lid * ${NRED}u + x] = 0.0; }

  if (gid.x < tot) {
    let k = (gid.x % NM) + 1u;
    let j = ((gid.x / NM) % NM) + 1u;
    let i = (gid.x / (NM * NM)) + 1u;
    let id = i * p.S2 + j * p.S + k;

    var en: f32 = 0.0;
    for (var m: u32 = 0u; m < ${NELEC}u; m++) {
      let o = m * p.S3;
      let v = U[o + id];
      let Zm = atoms[m].Z;
      sn[lid * ${NRED}u + m] = v * v * p.h3;
      if (W[o + id] >= 0.5) {
        let a = U[o + id + p.S2] - v;
        let b = U[o + id + p.S] - v;
        let c = U[o + id + 1u] - v;
        // w-weighted gradient for kinetic energy
        let wval = W[o + id];
        en += Zm * 0.5 * wval * (a * a + b * b + c * c) * p.h;
      }
      en += Zm * (Pv[o + id] - K[id]) * v * v * p.h3;
    }
    sn[lid * ${NRED}u + ${NELEC}u] = en;
  }

  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var x: u32 = 0u; x < ${NRED}u; x++) {
        sn[lid * ${NRED}u + x] += sn[(lid + s) * ${NRED}u + x];
      }
    }
    workgroupBarrier();
  }

  if (lid == 0u) {
    let base = wgid.x * ${NRED}u;
    for (var x: u32 = 0u; x < ${NRED}u; x++) {
      partials[base + x] = sn[x];
    }
  }
}
`;
}

function generateFinalizeWGSL() {
  return `
struct NWG { count: u32 }
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> sums: array<f32>;
@group(0) @binding(2) var<uniform> nwg: NWG;

var<workgroup> wg: array<f32, ${Math.max(1024, NRED * 256)}>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
  for (var x: u32 = 0u; x < ${NRED}u; x++) { wg[lid * ${NRED}u + x] = 0.0; }

  for (var i: u32 = lid; i < nwg.count; i += 256u) {
    for (var x: u32 = 0u; x < ${NRED}u; x++) {
      wg[lid * ${NRED}u + x] += partials[i * ${NRED}u + x];
    }
  }

  workgroupBarrier();

  for (var s: u32 = 128u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var x: u32 = 0u; x < ${NRED}u; x++) {
        wg[lid * ${NRED}u + x] += wg[(lid + s) * ${NRED}u + x];
      }
    }
    workgroupBarrier();
  }

  if (lid == 0u) {
    for (var x: u32 = 0u; x < ${NRED}u; x++) {
      sums[x] = wg[x];
    }
  }
}
`;
}

function generateNormalizeWGSL() {
  return `
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

  // Normalize each electron to 1 (not Z)
  for (var m: u32 = 0u; m < ${NELEC}u; m++) {
    let n = sums[m];
    if (n > 0.0) { U[m * p.S3 + id] *= inverseSqrt(n); }
  }
}
`;
}

function generateExtractWGSL() {
  return `
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

  // 2D slices for each electron
  for (var m: u32 = 0u; m < ${NELEC}u; m++) {
    let idx = m * p.S3 + i * p.S2 + j * p.S + p.N2;
    out[m * SS * SS + i * SS + j] = select(0.0, U[idx], W[idx] > 0.0);
  }

  // Line plots (w, u, P, K) along i-axis at fixed j,k
  if (j == 0u) {
    let b = ${NELEC}u * SS * SS;
    for (var m: u32 = 0u; m < ${NELEC}u; m++) {
      // w slice
      out[b + m * SS + i] = W[m * p.S3 + i * p.S2 + (p.N2 + 8u) * p.S + p.N2];
      // u slice
      let uIdx = m * p.S3 + i * p.S2 + (p.N2 + 5u) * p.S + p.N2;
      out[b + ${NELEC}u * SS + m * SS + i] = select(0.0, U[uIdx], W[uIdx] > 0.0);
      // P slice
      out[b + ${2 * NELEC}u * SS + m * SS + i] = Pv[m * p.S3 + i * p.S2 + p.N2 * p.S + p.N2];
    }
    // K slice
    out[b + ${3 * NELEC}u * SS + i] = K[i * p.S2 + p.N2 * p.S + p.N2];
  }
}
`;
}

// ===== GPU STATE =====
let device, paramsBuf, atomsBuf, K_buf, sumsBuf, sumsReadBuf, sliceBuf, sliceReadBuf, partialsBuf, numWGBuf;
let U_buf = [], W_buf = [], P_buf = [];
let updatePL, reducePL, finalizePL, normalizePL, extractPL;
let updateBG = [], reduceBG = [], finalizeBG, normalizeBG = [], extractBG = [];
let cur = 0, gpuReady = false, computing = false;
let gpuError = null;

// Phase system: 0=molecule, 1=far apart, 2=done
let phase = 0, phaseSteps = 0;
const PHASE_STEPS = 10000;
let E_mol = 0, E_mol2 = 0;

const SLICE_SIZE = (NELEC * S * S + (3 * NELEC + 1) * S) * 4;
const WG_UPDATE = Math.ceil(INTERIOR / 256);
const WG_REDUCE = Math.ceil(INTERIOR / 256);
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

function buildAtomBuffer() {
  // Each atom: 8 x f32 (posI, posJ, posK, _pad, Z, Z_nuc, rc, _pad)
  const data = new ArrayBuffer(NELEC * 32);
  const u32v = new Uint32Array(data);
  const f32v = new Float32Array(data);
  for (let a = 0; a < NELEC; a++) {
    const base = a * 8;
    u32v[base + 0] = nucPos[a][0];
    u32v[base + 1] = nucPos[a][1];
    u32v[base + 2] = nucPos[a][2];
    u32v[base + 3] = 0;
    f32v[base + 4] = Z[a];
    f32v[base + 5] = Z_nuc[a];
    f32v[base + 6] = r_cut_arr[a];
    f32v[base + 7] = 0;
  }
  return data;
}

function uploadInitialData() {
  console.log("Init: " + NELEC + " atoms");
  for (let n = 0; n < NELEC; n++) {
    console.log("  atom " + n + ": pos=(" + nucPos[n] + ") Z=" + Z[n] + " Z_nuc=" + Z_nuc[n]);
  }

  const Kd = new Float32Array(S3);
  const Ud = new Float32Array(NELEC * S3);
  const Wd = new Float32Array(NELEC * S3);
  const Pd = new Float32Array(NELEC * S3);
  const soft = 0.04 * h2v;

  for (let i = 0; i <= NN; i++) {
    const dx = [];
    for (let n = 0; n < NELEC; n++) dx[n] = (i - nucPos[n][0]) * hv;
    const dxI = [];
    for (let n = 0; n < NELEC; n++) dxI[n] = (i - initPos[n][0]) * hv;
    for (let j = 0; j <= NN; j++) {
      const dy = [];
      for (let n = 0; n < NELEC; n++) dy[n] = (j - nucPos[n][1]) * hv;
      const dyI = [];
      for (let n = 0; n < NELEC; n++) dyI[n] = (j - initPos[n][1]) * hv;
      for (let k = 0; k <= NN; k++) {
        const dz = [];
        for (let n = 0; n < NELEC; n++) dz[n] = (k - nucPos[n][2]) * hv;
        const dzI = [];
        for (let n = 0; n < NELEC; n++) dzI[n] = (k - initPos[n][2]) * hv;
        const id = i * S2 + j * S + k;

        // Nuclear Coulomb potential using Z_nuc
        let Kval = 0;
        for (let n = 0; n < NELEC; n++) {
          const rn = Math.sqrt(dx[n]*dx[n] + dy[n]*dy[n] + dz[n]*dz[n] + soft);
          Kval += Z_nuc[n] / rn;
        }
        Kd[id] = Kval;

        // Initial wavefunctions: place electron at init position
        const r = [], u = [];
        for (let n = 0; n < NELEC; n++) {
          r[n] = Math.sqrt(dxI[n]*dxI[n] + dyI[n]*dyI[n] + dzI[n]*dzI[n] + soft);
          const Zeff = Z[n] > 0 ? Z[n] : 1;
          u[n] = Math.exp(-Zeff * r[n]);
        }

        // Assign each grid point to nearest active electron
        let best = -1;
        for (let n = 0; n < NELEC; n++) {
          if (Z[n] > 0 && (best < 0 || u[n] > u[best])) best = n;
        }
        if (best >= 0) {
          const rBest = Math.sqrt(dx[best]*dx[best] + dy[best]*dy[best] + dz[best]*dz[best] + soft);
          const w = rBest <= R_out ? smoothCut(rBest, r_cut_arr[best]) : 0;
          Ud[best * S3 + id] = u[best];
          Wd[best * S3 + id] = w;
        }

        // Initial Poisson: average potential
        const pAvg = Kval / NELEC;
        for (let m = 0; m < NELEC; m++) {
          Pd[m * S3 + id] = pAvg;
        }
      }
    }
  }

  console.log("Uploading to GPU...");
  device.queue.writeBuffer(K_buf, 0, Kd);
  device.queue.writeBuffer(atomsBuf, 0, new Uint8Array(buildAtomBuffer()));
  for (let i = 0; i < 2; i++) {
    device.queue.writeBuffer(U_buf[i], 0, Ud);
    device.queue.writeBuffer(W_buf[i], 0, Wd);
    device.queue.writeBuffer(P_buf[i], 0, Pd);
  }
  cur = 0;
}

function updateParamsBuf() {
  // Param struct: 16 fields = 64 bytes
  const pb = new ArrayBuffer(64);
  const pu = new Uint32Array(pb);
  const pf = new Float32Array(pb);
  pu[0] = NN; pu[1] = S; pu[2] = S2; pu[3] = S3;
  pu[4] = N2; pu[5] = NELEC; pf[6] = hv; pf[7] = h2v;
  pf[8] = 1 / hv; pf[9] = 1 / h2v; pf[10] = dtv; pf[11] = half_dv;
  pf[12] = r_cut; pf[13] = 4 * Math.PI; pf[14] = h3v; pu[15] = 0;
  device.queue.writeBuffer(paramsBuf, 0, pb);
  // Also update atom buffer
  device.queue.writeBuffer(atomsBuf, 0, new Uint8Array(buildAtomBuffer()));
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
    console.log("WebGPU device ready, NELEC=" + NELEC);

    device.lost.then((info) => {
      gpuError = "GPU device lost: " + info.message;
      gpuReady = false;
    });

    const bs = S3 * 4, bN = NELEC * S3 * 4;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    K_buf = device.createBuffer({ size: bs, usage });
    atomsBuf = device.createBuffer({ size: NELEC * 32, usage });
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

    paramsBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    updateParamsBuf();
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

    const updateMod = await compileShader('update', generateUpdateWGSL());
    const reduceMod = await compileShader('reduce', generateReduceWGSL());
    const finalizeMod = await compileShader('finalize', generateFinalizeWGSL());
    const normalizeMod = await compileShader('normalize', generateNormalizeWGSL());
    const extractMod = await compileShader('extract', generateExtractWGSL());

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
        { binding: 8, resource: { buffer: atomsBuf } },
      ]});
      reduceBG[c] = device.createBindGroup({ layout: reducePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: W_buf[c] } },
        { binding: 3, resource: { buffer: P_buf[c] } },
        { binding: 4, resource: { buffer: K_buf } },
        { binding: 5, resource: { buffer: partialsBuf } },
        { binding: 6, resource: { buffer: atomsBuf } },
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
  E = sumsData[NELEC]; // energy is last element

  await sliceReadBuf.mapAsync(GPUMapMode.READ);
  sliceData = new Float32Array(sliceReadBuf.getMappedRange().slice(0));
  sliceReadBuf.unmap();

  tStep += n;
  lastMs = performance.now() - t0;

  if (addNucRepulsion) {
    const soft_nuc = 0.04 * h2v;
    for (let a = 0; a < NELEC; a++) {
      for (let b = a + 1; b < NELEC; b++) {
        const d = Math.sqrt(
          ((nucPos[a][0]-nucPos[b][0])*hv)**2 +
          ((nucPos[a][1]-nucPos[b][1])*hv)**2 +
          ((nucPos[a][2]-nucPos[b][2])*hv)**2 + soft_nuc);
        E += Z_nuc[a]*Z_nuc[b]/d;
      }
    }
  }

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

  if (!computing && phase < 2) {
    computing = true;
    doSteps(STEPS_PER_FRAME).then(() => {
      computing = false;
      phaseSteps += STEPS_PER_FRAME;
      if (isFinite(E) && E < E_min) E_min = E;

      if (phaseSteps >= PHASE_STEPS) {
        if (phase === 0) {
          E_mol = E;
          console.log("=== MOLECULE DONE: E_mol=" + E_mol.toFixed(6) + " ===");
          phase = 2; // done (no automatic second phase in generalized version)
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

    // Color palette for up to 8 electrons (r, g, b, yellow, cyan, magenta, orange, white)
    const elecColors = [
      [1, 0, 0], [0, 1, 0], [0, 0.4, 1],
      [1, 1, 0], [0, 1, 1], [1, 0, 1],
      [1, 0.5, 0], [1, 1, 1]
    ];

    for (let i = 1; i < NN; i++) {
      const px0 = Math.floor(PX * i * d);
      const px1 = Math.floor(PX * (i + 1) * d);
      for (let j = 1; j < NN; j++) {
        const py0 = Math.floor(PX * j * d);
        const py1 = Math.floor(PX * (j + 1) * d);
        const b = i * SS + j;
        let ri = 0, gi = 0, bi = 0;
        for (let m = 0; m < NELEC; m++) {
          const val = 500 * sliceData[m * SS * SS + b];
          const cc = elecColors[m % elecColors.length];
          ri += val * cc[0];
          gi += val * cc[1];
          bi += val * cc[2];
        }
        ri = Math.min(255, Math.floor(ri));
        gi = Math.min(255, Math.floor(gi));
        bi = Math.min(255, Math.floor(bi));
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
    const lb = NELEC * SS * SS;
    for (let i = 1; i < NN - 10; i++) {
      for (let m = 0; m < NELEC; m++) {
        const cc = elecColors[m % elecColors.length];
        // w
        fill(255); ellipse(PX * i, 300 - 100 * sliceData[lb + m * SS + i], 2);
        // u
        fill(cc[0]*255, cc[1]*255, cc[2]*255);
        ellipse(PX * i, 300 - 100 * sliceData[lb + NELEC * SS + m * SS + i], 3);
        // P
        fill(0, 255, 255, 200);
        ellipse(PX * i, 300 - 30 * sliceData[lb + 2 * NELEC * SS + m * SS + i], 2);
      }
      // K
      fill(0, 0, 255, 200);
      ellipse(PX * i, 300 - 30 * sliceData[lb + 3 * NELEC * SS + i], 2);
    }
  }

  // Draw nuclear positions
  fill(255); stroke(255); strokeWeight(1);
  for (let n = 0; n < NELEC; n++) {
    circle(nucPos[n][0] * PX, nucPos[n][1] * PX, 6);
  }
  // Screen boundary
  noFill(); stroke(100); strokeWeight(1);
  rect(0, 0, 400, 400);
  noStroke();

  // HUD
  fill(255);
  text(NELEC + " atoms | " + NN + "^3", 5, 20);
  text("step " + tStep + " (" + phaseSteps + "/" + PHASE_STEPS + ")  E=" + E.toFixed(6), 5, 35);
  if (lastMs > 0) {
    const msPerStep = (lastMs / STEPS_PER_FRAME).toFixed(2);
    text(msPerStep + " ms/step", 300, 20);
    text(lastMs.toFixed(0) + "ms/" + STEPS_PER_FRAME + "steps", 300, 35);
  }

  if (phase >= 1) {
    fill(255, 255, 0);
    text("E_mol = " + E_mol.toFixed(6), 5, 55);
  }
  if (phase >= 2) {
    fill(0, 255, 0);
    text("E_min = " + E_min.toFixed(6), 5, 70);
  }
}
