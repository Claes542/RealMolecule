// Molecule Quantum Simulation — WebGPU Compute Shaders + Nuclear Dynamics
// Up to 10 atoms placed interactively, 3D geometry
console.log("molecule.js loaded v3 — shear body force available via USER_SHEAR_RATE");
const NN = window.USER_NN || 200;
const S = NN + 1;
const S2 = S * S;
const S3 = S * S * S;
const N2 = Math.round(NN / 2);
const MAX_ATOMS = window.USER_MAX_ATOMS || 2048;
const _uz = window.USER_Z || [2, 3, 1, 0, 0];
while (_uz.length < MAX_ATOMS) _uz.push(0);
const NELEC = (function() {
  const zn = window.USER_Z_NUC || _uz;
  let c = 0;
  for (let i = 0; i < _uz.length; i++) if (_uz[i] > 0 || (zn[i] && zn[i] > 0)) c++;
  return c || 3;
})();
const N_ELECTRONS = _uz.reduce((s, z) => s + z, 0);  // total valence electrons
const NRED_E = 6;  // Energy reduce: T + V_eK + V_ee + dipole(x,y,z)
const r_cut = window.USER_RC || [0, 0, 0, 0, 0];
while (r_cut.length < MAX_ATOMS) r_cut.push(0);
// Z_nuc: nuclear charge for K potential (defaults to Z if not set)
// Allows bare protons (Z=0 no electron, Z_nuc=1 contributes to K)
const Z_nuc = window.USER_Z_NUC || _uz.map(z => z);
while (Z_nuc.length < MAX_ATOMS) Z_nuc.push(0);
let R_out = 0.5;   // au, unused legacy
let curvReg = (window.USER_CURV_REG !== undefined) ? window.USER_CURV_REG : 0.15;  // curvature regularization for free boundary
let Z = [..._uz];
let Ne = [..._uz];
const Z_orig = [..._uz];
const Ne_orig = [..._uz];

// Atom positions from interactive placement or defaults
const _atoms = window.USER_ATOMS || [
  { i: 80, j: 120, Z: 2, el: 'O' },
  { i: 100, j: 77, Z: 3, el: 'N' },
  { i: 120, j: 120, Z: 1, el: 'H' },
  { i: 100, j: 100, Z: 0, el: '' },
  { i: 100, j: 100, Z: 0, el: '' }
];
while (_atoms.length < MAX_ATOMS) _atoms.push({ i: N2, j: N2, Z: 0, el: '' });
const atomLabels = _atoms.map(a => a.el);
// Build unique nucleus list for V_KK (electrons on same nucleus share position)
// Z_eff = valence charge (core electrons screen full nuclear charge)
const _nucMap = new Map(); // key "i,j,k" -> { idx, Z_eff, elecIndices }
for (let e = 0; e < _atoms.length; e++) {
  const zEl = _atoms[e].Z;
  const zNucVal = Z_nuc[e] || 0;
  if (zEl === 0 && zNucVal === 0) continue;
  const a = _atoms[e];
  const k = a.k !== undefined ? a.k : N2;
  const key = a.i + "," + a.j + "," + k;
  // Z_eff for nuclear repulsion uses Z_nuc (screened by core electrons in r_c)
  if (_nucMap.has(key)) { _nucMap.get(key).Z_eff += zNucVal || zEl; _nucMap.get(key).elecIndices.push(e); }
  else _nucMap.set(key, { idx: _nucMap.size, Z_eff: zNucVal || zEl, elecIndices: [e] });
}
const uniqueNuclei = [..._nucMap.values()]; // [{idx, Z_eff, elecIndices}, ...]
// Map electron index -> unique nucleus index
const elecToNuc = new Array(_atoms.length).fill(-1);
for (const nuc of uniqueNuclei) for (const e of nuc.elecIndices) elecToNuc[e] = nuc.idx;
let nucPos = _atoms.map(a => [a.i, a.j, a.k !== undefined ? a.k : N2]);
const molNucPos = nucPos.map(p => [...p]);

let E_min = Infinity;
const PROTEIN_COUNT = window.USER_PROTEIN_COUNT || NELEC;  // atoms with force arrows
let screenAu = window.USER_SCREEN || 10;
let hGrid = screenAu / NN, h2v = hGrid * hGrid, h3v = hGrid * hGrid * hGrid;
const dv = window.USER_DV || (NELEC > 500 ? 0.01 : NELEC > 100 ? 0.03 : 0.12);  // smaller timestep for large systems
let dtv = dv * h2v, half_dv = 0.5 * dv;
const CANVAS_SIZE = window.USER_CANVAS || 700;
const PX = CANVAS_SIZE / NN;
const INTERIOR = (NN - 1) * (NN - 1) * (NN - 1);
const R_SING = 2 * hGrid;  // exclude 2 grid spacings from nucleus
const W_CUTOFF = window.USER_W_CUTOFF || 0;  // smooth ψ cutoff near other nuclei (au), 0 = off
// Detect if any bare nuclei exist (Z=0, Z_nuc>0) at compile time
const HAS_BARE_NUCLEI = (function() {
  const zn = window.USER_Z_NUC || _uz.map(z => z);
  for (let i = 0; i < _uz.length; i++) {
    if (_uz[i] === 0 && (zn[i] || 0) > 0) return true;
  }
  return false;
})();

// 2D dispatch to handle >65535 workgroups (300^3 grid needs ~104K)
const DISPATCH_X = 256;  // workgroups in x dimension (fixed)
const THREADS_X = DISPATCH_X * 256;  // 65536 total x-threads
function dispatchLinear(pass, totalCells) {
  const wg = Math.ceil(totalCells / 256);
  if (wg <= 65535) {
    pass.dispatchWorkgroups(wg);
  } else {
    pass.dispatchWorkgroups(DISPATCH_X, Math.ceil(wg / DISPATCH_X));
  }
}

const STEPS_PER_FRAME = window.USER_STEPS_PER_FRAME || (NELEC <= 5 ? 50 : NELEC <= 15 ? 50 : NELEC <= 30 ? 50 : NELEC <= 100 ? 5 : 2);
const W_STEPS_PER_FRAME = Math.max(1, STEPS_PER_FRAME);  // match U steps per frame
const BOUNDARY_INTERVAL = 20;
const NORM_INTERVAL = 20;
const POISSON_INTERVAL = window.USER_POISSON_INTERVAL || 50;
const USE_DIRECT_POTHER = (window.USER_DIRECT_POTHER !== undefined) ? window.USER_DIRECT_POTHER : (NELEC <= 5);  // direct per-electron Poisson solve (no SIC needed)
const SIC_INTERVAL = NELEC <= 15 ? 1 : NELEC <= 30 ? 5 : 999999;  // SIC in dynamics to remove self-interaction from wavefunction evolution
const SIC_JACOBI = NELEC <= 15 ? 50 : 10;

// === Nuclear dynamics state ===
const N_MOVE = window.USER_N_MOVE || 10;  // electronic steps between nuclear moves
const DT_NUC = window.USER_DT_NUC || (NELEC <= 5 ? 2.0 : NELEC > 200 ? 0.2 : 0.8);  // au
const NUC_SUBSTEPS = window.USER_NUC_SUBSTEPS || (NELEC <= 5 ? 2 : 1);
const DAMPING = window.USER_DAMPING || 0.98;       // light damping
const MAX_VEL = NELEC <= 5 ? 0.3 : NELEC > 200 ? 0.03 : 0.1;  // au/au_time
let forceScale = window.USER_FORCE_SCALE || 1.0;       // adjustable via slider/keys
// Langevin thermostat: gamma = friction, kT = target temperature in Hartree
// T(K) = kT * 315775.  Room temp 300K → kT = 0.00095 Ha.  Set USER_TEMPERATURE_K to enable.
const LANGEVIN_GAMMA = window.USER_LANGEVIN_GAMMA || 0.01;  // friction coefficient (au^-1)
let langevinKT = 0;  // 0 = thermostat off (pure damping)
if (window.USER_TEMPERATURE_K) {
  langevinKT = window.USER_TEMPERATURE_K / 315775.0;
  console.log("Langevin thermostat: T=" + window.USER_TEMPERATURE_K + " K, kT=" + langevinKT.toExponential(3) + " Ha, gamma=" + LANGEVIN_GAMMA);
}
// Expose for runtime temperature control (slider)
Object.defineProperty(window, 'langevinKT', {
  get() { return langevinKT; },
  set(v) { langevinKT = v; }
});
let boundarySpeed = 0.5;    // dt_w for free boundary evolution
let nucVel = Array.from({length: MAX_ATOMS}, () => [0, 0, 0]);
// Apply initial velocities if specified
if (window.USER_INIT_VEL) {
  for (let a = 0; a < window.USER_INIT_VEL.length && a < MAX_ATOMS; a++) {
    nucVel[a] = [...window.USER_INIT_VEL[a]];
  }
}
let nucForce = Array.from({length: MAX_ATOMS}, () => [0, 0, 0]);
let nucForceElec = Array.from({length: MAX_ATOMS}, () => [0, 0, 0]);
let nucForceNuc = Array.from({length: MAX_ATOMS}, () => [0, 0, 0]);
let nucForceTotal = Array.from({length: MAX_ATOMS}, () => [0, 0, 0]);
let nucStepCount = 0, dynamicsEnabled = window.USER_DYNAMICS || false;
function nucMass(z) { return ({1:1, 2:16, 3:14, 4:12}[z] || 1) * 1836; }

// Multigrid coarse grid
if (NN % 2 !== 0) throw new Error("NN must be even for multigrid");
const NC = Math.floor(NN / 2);
const SC = NC + 1, SC2 = SC * SC, SC3 = SC * SC * SC;
const INTERIOR_C = (NC - 1) * (NC - 1) * (NC - 1);

// Energy reduce: 6 values (T, V_eK, V_ee, dipX, dipY, dipZ) — REDUCE_WG=128 (6*128*4 = 3072 bytes shared mem)
const REDUCE_WG = 128;
const NORM_SCALE = 16777216.0; // 2^24 for fixed-point atomic norm accumulation
const NORM_INV_SCALE = 1.0 / NORM_SCALE;

// ===== WGSL SHADERS =====

// WGSL helper: compute linear cell index from 2D dispatch
const cellIdxWGSL = `
fn cellIdx(gid: vec3<u32>) -> u32 {
  return gid.x + gid.y * ${THREADS_X}u;
}`;

// Param struct: 16 common fields = 64 bytes. Atom data in separate storage buffer.
const PARAM_BYTES = 64;
const paramStructWGSL = `
struct P {
  NN: u32, S: u32, S2: u32, S3: u32,
  N2: u32, dt_w: f32, curvReg: f32, TWO_PI: f32,
  h: f32, h2: f32, inv_h: f32, inv_h2: f32,
  dt: f32, half_d: f32, h3: f32, sliceK: u32,
}`;

const ATOM_STRIDE = 8; // 8 f32s per atom (posI, posJ, posK, Z, rc, Z_nuc, initZeff, initRcut)
const ATOM_BUF_BYTES = MAX_ATOMS * ATOM_STRIDE * 4;
const atomStructWGSL = `
struct Atom {
  posI: u32, posJ: u32, posK: u32, Z: f32,
  rc: f32, Z_nuc: f32, initZeff: f32, initRcut: f32,
}`;

// U update — label-based domains, Neumann BC at domain boundaries and r_c surfaces
const updateU_WGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> Ui: array<f32>;
@group(0) @binding(3) var<storage, read> label: array<u32>;
@group(0) @binding(4) var<storage, read> Pi: array<f32>;
@group(0) @binding(5) var<storage, read_write> Uo: array<f32>;
@group(0) @binding(6) var<storage, read> atoms: array<Atom>;

fn distToAtom(ci: u32, cj: u32, ck: u32, n: u32) -> f32 {
  let dx = (f32(ci) - f32(atoms[n].posI)) * p.h;
  let dy = (f32(cj) - f32(atoms[n].posJ)) * p.h;
  let dz = (f32(ck) - f32(atoms[n].posK)) * p.h;
  return sqrt(dx*dx + dy*dy + dz*dz);
}

fn isInsideRc(ci: u32, cj: u32, ck: u32, lbl: u32) -> bool {
  let rc = atoms[lbl].rc;
  if (rc > 0.0 && distToAtom(ci, cj, ck, lbl) < rc) { return true; }
${HAS_BARE_NUCLEI ? `
  // Check bare nuclei (Z=0, Z_nuc>0) — only compiled when they exist
  for (var n: u32 = 0u; n < ${NELEC}u; n++) {
    if (atoms[n].Z <= 0.0 && atoms[n].Z_nuc > 0.0 && atoms[n].rc > 0.0) {
      if (distToAtom(ci, cj, ck, n) < atoms[n].rc) { return true; }
    }
  }
` : ''}
  return false;
}

fn isInsideAnalytical(ci: u32, cj: u32, ck: u32, lbl: u32) -> bool {
  if (atoms[lbl].rc > 0.0) { return false; }  // pseudopotential atoms: handled by isInsideRc
  return distToAtom(ci, cj, ck, lbl) < ${R_SING};
}

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }

  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let myL = label[id];

  // Inside r_c (pseudopotential): U = 0 (Dirichlet)
  if (isInsideRc(i, j, k, myL)) {
    Uo[id] = 0.0;
    return;
  }

  let uc = Ui[id];

  // Neumann BC at r_c boundaries and domain boundaries
  let l_ip = label[id + p.S2]; let excl_ip = isInsideRc(i+1u, j, k, l_ip);
  let l_im = label[id - p.S2]; let excl_im = isInsideRc(i-1u, j, k, l_im);
  let l_jp = label[id + p.S];  let excl_jp = isInsideRc(i, j+1u, k, l_jp);
  let l_jm = label[id - p.S];  let excl_jm = isInsideRc(i, j-1u, k, l_jm);
  let l_kp = label[id + 1u];   let excl_kp = isInsideRc(i, j, k+1u, l_kp);
  let l_km = label[id - 1u];   let excl_km = isInsideRc(i, j, k-1u, l_km);

  // Smooth W cutoff: Neumann BC near other electrons' nuclei
  // w(r) blends neighbor toward uc (flat) near other nucleus
  let w_c = ${W_CUTOFF.toFixed(6)};
  var w_ip: f32 = 1.0; var w_im: f32 = 1.0;
  var w_jp: f32 = 1.0; var w_jm: f32 = 1.0;
  var w_kp: f32 = 1.0; var w_km: f32 = 1.0;
  if (w_c > 0.0) {
    let hw = 2.0 * p.h;
    for (var n: u32 = 0u; n < ${NELEC}u; n++) {
      if (n == myL) { continue; }
      let ax = f32(atoms[n].posI); let ay = f32(atoms[n].posJ); let az = f32(atoms[n].posK);
      // w(r) for each neighbor position
      let d_ip = sqrt((f32(i+1u)-ax)*(f32(i+1u)-ax) + (f32(j)-ay)*(f32(j)-ay) + (f32(k)-az)*(f32(k)-az)) * p.h;
      let d_im = sqrt((f32(i-1u)-ax)*(f32(i-1u)-ax) + (f32(j)-ay)*(f32(j)-ay) + (f32(k)-az)*(f32(k)-az)) * p.h;
      let d_jp = sqrt((f32(i)-ax)*(f32(i)-ax) + (f32(j+1u)-ay)*(f32(j+1u)-ay) + (f32(k)-az)*(f32(k)-az)) * p.h;
      let d_jm = sqrt((f32(i)-ax)*(f32(i)-ax) + (f32(j-1u)-ay)*(f32(j-1u)-ay) + (f32(k)-az)*(f32(k)-az)) * p.h;
      let d_kp = sqrt((f32(i)-ax)*(f32(i)-ax) + (f32(j)-ay)*(f32(j)-ay) + (f32(k+1u)-az)*(f32(k+1u)-az)) * p.h;
      let d_km = sqrt((f32(i)-ax)*(f32(i)-ax) + (f32(j)-ay)*(f32(j)-ay) + (f32(k-1u)-az)*(f32(k-1u)-az)) * p.h;
      w_ip = min(w_ip, clamp((d_ip - w_c + hw) / (2.0 * hw), 0.0, 1.0));
      w_im = min(w_im, clamp((d_im - w_c + hw) / (2.0 * hw), 0.0, 1.0));
      w_jp = min(w_jp, clamp((d_jp - w_c + hw) / (2.0 * hw), 0.0, 1.0));
      w_jm = min(w_jm, clamp((d_jm - w_c + hw) / (2.0 * hw), 0.0, 1.0));
      w_kp = min(w_kp, clamp((d_kp - w_c + hw) / (2.0 * hw), 0.0, 1.0));
      w_km = min(w_km, clamp((d_km - w_c + hw) / (2.0 * hw), 0.0, 1.0));
    }
  }

  // Blend neighbors toward uc where w < 1 (smooth Neumann at cutoff)
  let u_raw_ip = select(uc, Ui[id + p.S2], l_ip == myL && !excl_ip);
  let u_raw_im = select(uc, Ui[id - p.S2], l_im == myL && !excl_im);
  let u_raw_jp = select(uc, Ui[id + p.S],  l_jp == myL && !excl_jp);
  let u_raw_jm = select(uc, Ui[id - p.S],  l_jm == myL && !excl_jm);
  let u_raw_kp = select(uc, Ui[id + 1u],   l_kp == myL && !excl_kp);
  let u_raw_km = select(uc, Ui[id - 1u],   l_km == myL && !excl_km);

  let u_ip = mix(uc, u_raw_ip, w_ip);
  let u_im = mix(uc, u_raw_im, w_im);
  let u_jp = mix(uc, u_raw_jp, w_jp);
  let u_jm = mix(uc, u_raw_jm, w_jm);
  let u_kp = mix(uc, u_raw_kp, w_kp);
  let u_km = mix(uc, u_raw_km, w_km);

  let lap = u_ip + u_im + u_jp + u_jm + u_kp + u_km - 6.0 * uc;

  // Full nuclear potential (all nuclei) minus other-electron repulsion (no self-repulsion)
  Uo[id] = uc + p.half_d * lap + p.dt * (K[id] - 2.0 * Pi[id]) * uc;
}
`;

// Chebyshev semi-iterative acceleration of ITP
// ψ_{n+1} = ω * (standard ITP step) + (1-ω) * ψ_{n-1}
// Same stencil as updateU, with momentum term
const chebyshevStepWGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> Ui: array<f32>;
@group(0) @binding(3) var<storage, read> label: array<u32>;
@group(0) @binding(4) var<storage, read> Pi: array<f32>;
@group(0) @binding(5) var<storage, read_write> Uo: array<f32>;
@group(0) @binding(6) var<storage, read> atoms: array<Atom>;
@group(0) @binding(7) var<storage, read> Uprev: array<f32>;

struct ChebP { omega: f32, _p0: f32, _p1: f32, _p2: f32 }
@group(0) @binding(8) var<uniform> cheb: ChebP;

fn distToAtom(ci: u32, cj: u32, ck: u32, n: u32) -> f32 {
  let dx = (f32(ci) - f32(atoms[n].posI)) * p.h;
  let dy = (f32(cj) - f32(atoms[n].posJ)) * p.h;
  let dz = (f32(ck) - f32(atoms[n].posK)) * p.h;
  return sqrt(dx*dx + dy*dy + dz*dz);
}

fn isInsideRc(ci: u32, cj: u32, ck: u32, lbl: u32) -> bool {
  let rc = atoms[lbl].rc;
  if (rc <= 0.0) { return false; }
  return distToAtom(ci, cj, ck, lbl) < rc;
}

fn isInsideAnalytical(ci: u32, cj: u32, ck: u32, lbl: u32) -> bool {
  return false;
}

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }

  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let myL = label[id];

  if (isInsideRc(i, j, k, myL)) {
    Uo[id] = 0.0;
    return;
  }

  let uc = Ui[id];

  let l_ip = label[id + p.S2]; let excl_ip = isInsideRc(i+1u, j, k, l_ip);
  let l_im = label[id - p.S2]; let excl_im = isInsideRc(i-1u, j, k, l_im);
  let l_jp = label[id + p.S];  let excl_jp = isInsideRc(i, j+1u, k, l_jp);
  let l_jm = label[id - p.S];  let excl_jm = isInsideRc(i, j-1u, k, l_jm);
  let l_kp = label[id + 1u];   let excl_kp = isInsideRc(i, j, k+1u, l_kp);
  let l_km = label[id - 1u];   let excl_km = isInsideRc(i, j, k-1u, l_km);

  let u_ip = select(uc, Ui[id + p.S2], l_ip == myL && !excl_ip);
  let u_im = select(uc, Ui[id - p.S2], l_im == myL && !excl_im);
  let u_jp = select(uc, Ui[id + p.S],  l_jp == myL && !excl_jp);
  let u_jm = select(uc, Ui[id - p.S],  l_jm == myL && !excl_jm);
  let u_kp = select(uc, Ui[id + 1u],   l_kp == myL && !excl_kp);
  let u_km = select(uc, Ui[id - 1u],   l_km == myL && !excl_km);

  let lap = u_ip + u_im + u_jp + u_jm + u_kp + u_km - 6.0 * uc;

  let itpStep = uc + p.half_d * lap + p.dt * (K[id] - 2.0 * Pi[id]) * uc;

  Uo[id] = cheb.omega * itpStep + (1.0 - cheb.omega) * Uprev[id];
}
`;

// Level set boundary evolution — accumulate density difference in W, flip when W < 0
// W > 0 means "I belong here", W < 0 means "neighbor density dominates, should flip"
const evolveBoundaryWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> labelIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> labelOut: array<u32>;
@group(0) @binding(3) var<storage, read> U: array<f32>;
@group(0) @binding(4) var<storage, read_write> W: array<f32>;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }

  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let myL = labelIn[id];
  let myRho = U[id] * U[id];

  let id_ip = id + p.S2; let id_im = id - p.S2;
  let id_jp = id + p.S;  let id_jm = id - p.S;
  let id_kp = id + 1u;   let id_km = id - 1u;

  let l_ip = labelIn[id_ip]; let l_im = labelIn[id_im];
  let l_jp = labelIn[id_jp]; let l_jm = labelIn[id_jm];
  let l_kp = labelIn[id_kp]; let l_km = labelIn[id_km];

  // Find best cross-boundary neighbor (highest density U²) + count same-domain
  var bestOtherL: u32 = myL;
  var bestOtherRho: f32 = 0.0;
  var myCnt: f32 = 0.0;

  if (l_ip == myL) { myCnt += 1.0; } else { let r = U[id_ip] * U[id_ip]; if (r > bestOtherRho) { bestOtherRho = r; bestOtherL = l_ip; } }
  if (l_im == myL) { myCnt += 1.0; } else { let r = U[id_im] * U[id_im]; if (r > bestOtherRho) { bestOtherRho = r; bestOtherL = l_im; } }
  if (l_jp == myL) { myCnt += 1.0; } else { let r = U[id_jp] * U[id_jp]; if (r > bestOtherRho) { bestOtherRho = r; bestOtherL = l_jp; } }
  if (l_jm == myL) { myCnt += 1.0; } else { let r = U[id_jm] * U[id_jm]; if (r > bestOtherRho) { bestOtherRho = r; bestOtherL = l_jm; } }
  if (l_kp == myL) { myCnt += 1.0; } else { let r = U[id_kp] * U[id_kp]; if (r > bestOtherRho) { bestOtherRho = r; bestOtherL = l_kp; } }
  if (l_km == myL) { myCnt += 1.0; } else { let r = U[id_km] * U[id_km]; if (r > bestOtherRho) { bestOtherRho = r; bestOtherL = l_km; } }

  var w = W[id];

  // Interior cells (all 6 neighbors same domain): keep W positive, no evolution
  if (myCnt >= 6.0) {
    w = max(w, 1.0);
    W[id] = w;
    labelOut[id] = myL;
    return;
  }

  // Check locked boundaries: don't evolve if both labels are in a locked group
${(function() {
    const groups = window.USER_LOCKED_GROUPS || [];
    if (groups.length === 0) return '';
    let code = '  var locked = false;\n';
    for (const g of groups) {
      const checks = g.map(a => `myL == ${a}u || bestOtherL == ${a}u`);
      // Both must be in the same group
      const myChecks = g.map(a => `myL == ${a}u`).join(' || ');
      const otherChecks = g.map(a => `bestOtherL == ${a}u`).join(' || ');
      code += `  if ((${myChecks}) && (${otherChecks})) { locked = true; }\n`;
    }
    code += '  if (locked) { W[id] = max(w, 0.5); labelOut[id] = myL; return; }\n';
    return code;
  })()}
  // Boundary cell: evolve W based on relative density difference
  // Normalized velocity in [-1,+1] so boundary speed is independent of system size
  let denom = myRho + bestOtherRho;
  let velocity = select((myRho - bestOtherRho) / denom, 0.0, denom < 1e-20);
  w += p.dt_w * velocity;

  // Curvature regularization: smooth jagged boundaries
  let curv = (myCnt - 3.0) / 3.0;  // [-1, +1], negative = surrounded by others
  w += p.curvReg * curv;

  var newL = myL;
  if (w < 0.0) {
    // Flip to best neighbor domain
    newL = bestOtherL;
    w = 0.1;  // reset W slightly positive in new domain
  }

  W[id] = w;
  labelOut[id] = newL;
}
`;

// Fix U at flipped boundary cells for density continuity: Z_old * u² = Z_new * u'²
const fixBoundaryU_WGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> labelOld: array<u32>;
@group(0) @binding(2) var<storage, read> labelNew: array<u32>;
@group(0) @binding(3) var<storage, read_write> U: array<f32>;
@group(0) @binding(4) var<storage, read> atoms: array<Atom>;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let oldL = labelOld[id];
  let newL = labelNew[id];
  // When cell flips domain, keep U continuous (density U² is the physical quantity)
  // Normalization step will adjust ∫U² = Z_eff for each domain
}
`;

// Compute density for a single domain (for self-potential calculation)
const computeRhoSelfWGSL = `
${paramStructWGSL}
struct DomIdx { idx: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read_write> rhoSelf: array<f32>;
@group(0) @binding(3) var<storage, read> label: array<u32>;
@group(0) @binding(4) var<uniform> dom: DomIdx;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;
  let u = U[id];
  rhoSelf[id] = select(0.0, u * u, label[id] == dom.idx);
}
`;

// Compute rho of all electrons EXCEPT the target label (for direct Pother solve)
const computeRhoOtherWGSL = `
${paramStructWGSL}
struct DomIdx { idx: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read_write> rhoOther: array<f32>;
@group(0) @binding(3) var<storage, read> label: array<u32>;
@group(0) @binding(4) var<uniform> dom: DomIdx;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;
  let u = U[id];
  rhoOther[id] = select(u * u, 0.0, label[id] == dom.idx);
}
`;

// GPU-side initialization of P_direct[m] = sum_{n≠m} 0.5/r (replaces CPU triple loop)
const initPdirectWGSL = `
${paramStructWGSL}
${atomStructWGSL}
struct DomIdx { idx: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> atoms: array<Atom>;
@group(0) @binding(2) var<storage, read_write> Pdirect: array<f32>;
@group(0) @binding(3) var<storage, read_write> Pother: array<f32>;
@group(0) @binding(4) var<uniform> dom: DomIdx;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;
  let xi = f32(i) * p.h;
  let yj = f32(j) * p.h;
  let zk = f32(k) * p.h;
  let m = dom.idx;
  var val: f32 = 0.0;
  for (var n: u32 = 0u; n < ${NELEC}u; n++) {
    let Za = atoms[n].Z;
    if (Za <= 0.0 || n == m) { continue; }
    let dx = xi - f32(atoms[n].posI) * p.h;
    let dy = yj - f32(atoms[n].posJ) * p.h;
    let dz = zk - f32(atoms[n].posK) * p.h;
    let r = sqrt(dx*dx + dy*dy + dz*dz + p.h2);
    val += 0.5 / r;
  }
  Pdirect[id] = val;
  Pother[id] += val;
}
`;

// Copy P_direct[m] to PotherBuf at cells where label == target
const copyPotherForLabelWGSL = `
${paramStructWGSL}
struct DomIdx { idx: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Psrc: array<f32>;
@group(0) @binding(2) var<storage, read_write> Pother: array<f32>;
@group(0) @binding(3) var<storage, read> label: array<u32>;
@group(0) @binding(4) var<uniform> dom: DomIdx;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;
  if (label[id] == dom.idx) {
    Pother[id] = Psrc[id];
  }
}
`;

// Subtract per-domain self-potential from Pother at points in that domain
// For domain with Z_eff electrons: subtract 1/Z_eff of self-potential
// (keep (Z_eff-1)/Z_eff for intra-domain electron-electron repulsion)
const subtractPselfWGSL = `
${paramStructWGSL}
struct DomIdx { idx: u32, Zeff_bits: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Pm: array<f32>;
@group(0) @binding(2) var<storage, read_write> Pother: array<f32>;
@group(0) @binding(3) var<storage, read> label: array<u32>;
@group(0) @binding(4) var<uniform> dom: DomIdx;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;
  if (label[id] == dom.idx) {
    let Zeff = bitcast<f32>(dom.Zeff_bits);
    let frac = select(1.0, 1.0 / Zeff, Zeff > 1.0);
    Pother[id] -= Pm[id] * frac;
  }
}
`;

// Jacobi smoother for single-field Poisson: Lap(P) = -2*pi*rho
const jacobiSmoothWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Pin: array<f32>;
@group(0) @binding(2) var<storage, read_write> Pout: array<f32>;
@group(0) @binding(3) var<storage, read> rhoTotal: array<f32>;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let Pc = Pin[id];
  let sum_nbr = Pin[id + p.S2] + Pin[id - p.S2]
              + Pin[id + p.S]  + Pin[id - p.S]
              + Pin[id + 1u]   + Pin[id - 1u];
  let rhs = p.h2 * p.TWO_PI * rhoTotal[id];
  Pout[id] = 0.3333 * Pc + (sum_nbr + rhs) / 9.0;
}
`;

// === Multigrid V-cycle shaders ===

// Compute rho_total from single density field + labels
const computeRhoWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read_write> rhoTotal: array<f32>;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;
  let u = U[id];
  rhoTotal[id] = u * u;
}
`;

// Compute residual: r = -2*pi*rho - lap(P)
const computeResidualWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Pin: array<f32>;
@group(0) @binding(2) var<storage, read_write> Rout: array<f32>;
@group(0) @binding(3) var<storage, read> rhoTotal: array<f32>;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let Pc = Pin[id];
  let lap = (Pin[id + p.S2] + Pin[id - p.S2]
           + Pin[id + p.S]  + Pin[id - p.S]
           + Pin[id + 1u]   + Pin[id - 1u]
           - 6.0 * Pc) * p.inv_h2;
  Rout[id] = -p.TWO_PI * rhoTotal[id] - lap;
}
`;

// Full-weighting 3D restriction (27-point stencil) — single field
const restrictWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> fine: array<f32>;
@group(0) @binding(2) var<storage, read_write> coarse: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NCM = ${NC - 1}u;
  let tot = NCM * NCM * NCM;
  if (gid.x >= tot) { return; }
  let K2 = (gid.x % NCM) + 1u;
  let J = ((gid.x / NCM) % NCM) + 1u;
  let I = (gid.x / (NCM * NCM)) + 1u;
  let fi = 2u * I; let fj = 2u * J; let fk = 2u * K2;

  var val: f32 = 8.0 * fine[fi*p.S2 + fj*p.S + fk];
  val += 4.0 * (fine[(fi+1u)*p.S2 + fj*p.S + fk] + fine[(fi-1u)*p.S2 + fj*p.S + fk]
              + fine[fi*p.S2 + (fj+1u)*p.S + fk] + fine[fi*p.S2 + (fj-1u)*p.S + fk]
              + fine[fi*p.S2 + fj*p.S + (fk+1u)] + fine[fi*p.S2 + fj*p.S + (fk-1u)]);
  val += 2.0 * (fine[(fi+1u)*p.S2 + (fj+1u)*p.S + fk] + fine[(fi+1u)*p.S2 + (fj-1u)*p.S + fk]
              + fine[(fi-1u)*p.S2 + (fj+1u)*p.S + fk] + fine[(fi-1u)*p.S2 + (fj-1u)*p.S + fk]
              + fine[(fi+1u)*p.S2 + fj*p.S + (fk+1u)] + fine[(fi+1u)*p.S2 + fj*p.S + (fk-1u)]
              + fine[(fi-1u)*p.S2 + fj*p.S + (fk+1u)] + fine[(fi-1u)*p.S2 + fj*p.S + (fk-1u)]
              + fine[fi*p.S2 + (fj+1u)*p.S + (fk+1u)] + fine[fi*p.S2 + (fj+1u)*p.S + (fk-1u)]
              + fine[fi*p.S2 + (fj-1u)*p.S + (fk+1u)] + fine[fi*p.S2 + (fj-1u)*p.S + (fk-1u)]);
  val += fine[(fi+1u)*p.S2 + (fj+1u)*p.S + (fk+1u)] + fine[(fi+1u)*p.S2 + (fj+1u)*p.S + (fk-1u)]
       + fine[(fi+1u)*p.S2 + (fj-1u)*p.S + (fk+1u)] + fine[(fi+1u)*p.S2 + (fj-1u)*p.S + (fk-1u)]
       + fine[(fi-1u)*p.S2 + (fj+1u)*p.S + (fk+1u)] + fine[(fi-1u)*p.S2 + (fj+1u)*p.S + (fk-1u)]
       + fine[(fi-1u)*p.S2 + (fj-1u)*p.S + (fk+1u)] + fine[(fi-1u)*p.S2 + (fj-1u)*p.S + (fk-1u)];

  coarse[I * ${SC2}u + J * ${SC}u + K2] = val * 0.015625;
}
`;

// Weighted Jacobi on coarse grid — single field
const coarseSmoothWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Ein: array<f32>;
@group(0) @binding(2) var<storage, read_write> Eout: array<f32>;
@group(0) @binding(3) var<storage, read> coarseRhs: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NCM = ${NC - 1}u;
  let tot = NCM * NCM * NCM;
  if (gid.x >= tot) { return; }
  let K2 = (gid.x % NCM) + 1u;
  let J = ((gid.x / NCM) % NCM) + 1u;
  let I = (gid.x / (NCM * NCM)) + 1u;
  let cid = I * ${SC2}u + J * ${SC}u + K2;

  let ec = Ein[cid];
  let sum_nbr = Ein[cid + ${SC2}u] + Ein[cid - ${SC2}u]
              + Ein[cid + ${SC}u]  + Ein[cid - ${SC}u]
              + Ein[cid + 1u]      + Ein[cid - 1u];
  let hc2 = 4.0 * p.h2;
  let f = coarseRhs[cid];
  Eout[cid] = 0.3333 * ec + (sum_nbr + hc2 * f) / 9.0;
}
`;

// Trilinear prolongation + additive correction to P — single field
const prolongCorrectWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Ec: array<f32>;
@group(0) @binding(2) var<storage, read_write> Pf: array<f32>;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;

  let ci = i / 2u; let cj = j / 2u; let ck = k / 2u;
  let ci1 = min(ci + (i & 1u), ${NC}u);
  let cj1 = min(cj + (j & 1u), ${NC}u);
  let ck1 = min(ck + (k & 1u), ${NC}u);
  let wi = select(1.0, 0.5, (i & 1u) == 1u);
  let wj = select(1.0, 0.5, (j & 1u) == 1u);
  let wk = select(1.0, 0.5, (k & 1u) == 1u);
  let wi1 = 1.0 - wi; let wj1 = 1.0 - wj; let wk1 = 1.0 - wk;

  var corr: f32 = 0.0;
  corr += wi  * wj  * wk  * Ec[ci  * ${SC2}u + cj  * ${SC}u + ck];
  corr += wi  * wj  * wk1 * Ec[ci  * ${SC2}u + cj  * ${SC}u + ck1];
  corr += wi  * wj1 * wk  * Ec[ci  * ${SC2}u + cj1 * ${SC}u + ck];
  corr += wi  * wj1 * wk1 * Ec[ci  * ${SC2}u + cj1 * ${SC}u + ck1];
  corr += wi1 * wj  * wk  * Ec[ci1 * ${SC2}u + cj  * ${SC}u + ck];
  corr += wi1 * wj  * wk1 * Ec[ci1 * ${SC2}u + cj  * ${SC}u + ck1];
  corr += wi1 * wj1 * wk  * Ec[ci1 * ${SC2}u + cj1 * ${SC}u + ck];
  corr += wi1 * wj1 * wk1 * Ec[ci1 * ${SC2}u + cj1 * ${SC}u + ck1];

  let fid = i * p.S2 + j * p.S + k;
  Pf[fid] += 0.5 * corr;
}
`;

// === Atomic norm accumulation (scales to 1000+ atoms) ===
const accumNormsWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> label: array<u32>;
@group(0) @binding(3) var<storage, read_write> normAtomic: array<atomic<u32>>;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }
  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;
  let v = U[id];
  let contrib = u32(v * v * p.h3 * ${NORM_SCALE});
  atomicAdd(&normAtomic[label[id]], contrib);
}
`;

// Decode atomic u32 norms to f32
const decodeNormsWGSL = `
@group(0) @binding(0) var<storage, read> normAtomic: array<u32>;
@group(0) @binding(1) var<storage, read_write> normFloat: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ${MAX_ATOMS}u) { return; }
  normFloat[gid.x] = f32(normAtomic[gid.x]) * ${NORM_INV_SCALE};
}
`;

// Energy-only reduce: just T, V_eK, V_ee (3 values, REDUCE_WG=128)
const MAX_REDUCE_WG = 256;
const reduceEnergyWGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> Pv: array<f32>;
@group(0) @binding(3) var<storage, read> K: array<f32>;
@group(0) @binding(4) var<storage, read_write> partials: array<f32>;
@group(0) @binding(5) var<storage, read> label: array<u32>;
@group(0) @binding(6) var<storage, read> atoms: array<Atom>;

fn isInsideExcl(ci: u32, cj: u32, ck: u32, lbl: u32) -> bool {
  let rc = max(atoms[lbl].rc, ${R_SING});
  let dx = (f32(ci) - f32(atoms[lbl].posI)) * p.h;
  let dy = (f32(cj) - f32(atoms[lbl].posJ)) * p.h;
  let dz = (f32(ck) - f32(atoms[lbl].posK)) * p.h;
  return sqrt(dx*dx + dy*dy + dz*dz) < rc;
}

var<workgroup> sn: array<f32, ${NRED_E * REDUCE_WG}>;

@compute @workgroup_size(${REDUCE_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(workgroup_id) wgid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let stride = ${MAX_REDUCE_WG}u * ${REDUCE_WG}u;
  let NR = ${NRED_E}u;

  for (var q: u32 = 0u; q < NR; q++) { sn[lid * NR + q] = 0.0; }

  var cell = gid.x;
  loop {
    if (cell >= tot) { break; }
    let k = (cell % NM) + 1u;
    let j = ((cell / NM) % NM) + 1u;
    let i = (cell / (NM * NM)) + 1u;
    let id = i * p.S2 + j * p.S + k;

    let v = U[id];
    let rho = v * v;
    let myL = label[id];

    // Skip points inside r_c exclusion sphere (pseudopotential atoms only)
    if (isInsideExcl(i, j, k, myL)) { cell = cell + stride; continue; }

    // Gradients: zero if neighbor is inside any exclusion sphere (Neumann BC)
    let sameL_ip = label[id + p.S2] == myL && !isInsideExcl(i+1u, j, k, myL);
    let sameL_jp = label[id + p.S]  == myL && !isInsideExcl(i, j+1u, k, myL);
    let sameL_kp = label[id + 1u]   == myL && !isInsideExcl(i, j, k+1u, myL);
    let a = select(0.0, U[id + p.S2] - v, sameL_ip);
    let b = select(0.0, U[id + p.S]  - v, sameL_jp);
    let c = select(0.0, U[id + 1u]   - v, sameL_kp);
    sn[lid * NR]        += 0.5 * (a * a + b * b + c * c) * p.h;
    sn[lid * NR + 1u]   += -K[id] * rho * p.h3;
    sn[lid * NR + 2u]   += Pv[id] * rho * p.h3;

    // Electronic dipole: -∫ ρ(r)·r dV  (negative sign applied on CPU)
    let xi = f32(i) * p.h;
    let yj = f32(j) * p.h;
    let zk = f32(k) * p.h;
    sn[lid * NR + 3u]   += rho * xi * p.h3;
    sn[lid * NR + 4u]   += rho * yj * p.h3;
    sn[lid * NR + 5u]   += rho * zk * p.h3;

    cell = cell + stride;
  }

  workgroupBarrier();

  for (var s: u32 = ${REDUCE_WG >> 1}u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var q: u32 = 0u; q < NR; q++) {
        sn[lid * NR + q] += sn[(lid + s) * NR + q];
      }
    }
    workgroupBarrier();
  }

  if (lid == 0u) {
    let base = wgid.x * NR;
    for (var q: u32 = 0u; q < NR; q++) {
      partials[base + q] = sn[q];
    }
  }
}
`;

const finalizeEnergyWGSL = `
struct NWG { count: u32 }
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> sums: array<f32>;
@group(0) @binding(2) var<uniform> nwg: NWG;

var<workgroup> wg: array<f32, ${NRED_E * REDUCE_WG}>;

@compute @workgroup_size(${REDUCE_WG})
fn main(@builtin(local_invocation_index) lid: u32) {
  let NR = ${NRED_E}u;
  for (var q: u32 = 0u; q < NR; q++) { wg[lid * NR + q] = 0.0; }

  for (var i: u32 = lid; i < nwg.count; i += ${REDUCE_WG}u) {
    for (var q: u32 = 0u; q < NR; q++) {
      wg[lid * NR + q] += partials[i * NR + q];
    }
  }

  workgroupBarrier();

  for (var s: u32 = ${REDUCE_WG >> 1}u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var q: u32 = 0u; q < NR; q++) {
        wg[lid * NR + q] += wg[(lid + s) * NR + q];
      }
    }
    workgroupBarrier();
  }

  if (lid == 0u) {
    for (var q: u32 = 0u; q < NR; q++) {
      sums[q] = wg[q];
    }
  }
}
`;

const normalizeWGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> U: array<f32>;
@group(0) @binding(2) var<storage, read> normFloat: array<f32>;
@group(0) @binding(3) var<storage, read> label: array<u32>;
@group(0) @binding(4) var<storage, read> atoms: array<Atom>;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(g);
  if (cell >= tot) { return; }

  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let lbl = label[id];
  let n = normFloat[lbl];
  let Zeff = atoms[lbl].Z;
  // Normalize so that ∫U² dV = Z_eff
  if (n > 0.0 && Zeff > 0.0) { U[id] *= sqrt(Zeff / n); }
}
`;

// Compact extract: 4 slices (density, elementZ, boundary, K line) instead of NELEC slices
const extractWGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> label: array<u32>;
@group(0) @binding(3) var<storage, read> K: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> atoms: array<Atom>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  let j = g.y;
  let SS = p.NN + 1u;
  if (i > p.NN || j > p.NN) { return; }

  let sk = p.sliceK;

  // K line + w(r) cutoff line along j=sliceK axis
  if (j == 0u) {
    let lineIdx = i * p.S2 + sk * p.S + sk;
    out[3u * SS * SS + i] = K[lineIdx];
    // Compute w(r) cutoff analytically
    let w_c = ${W_CUTOFF.toFixed(6)};
    var wVal: f32 = 1.0;
    if (w_c > 0.0) {
      let hw = 2.0 * p.h;
      let lbl = label[lineIdx];
      for (var n: u32 = 0u; n < ${NELEC}u; n++) {
        if (n == lbl) { continue; }
        let dx = (f32(i) - f32(atoms[n].posI)) * p.h;
        let dy = (f32(sk) - f32(atoms[n].posJ)) * p.h;
        let dz = (f32(sk) - f32(atoms[n].posK)) * p.h;
        let d = sqrt(dx*dx + dy*dy + dz*dz);
        let s = clamp((d - w_c + hw) / (2.0 * hw), 0.0, 1.0);
        wVal = min(wVal, s * s * (3.0 - 2.0 * s));
      }
    }
    out[3u * SS * SS + SS + i] = wVal;
  }

  if (i < 1u || i >= p.NN || j < 1u || j >= p.NN) {
    out[i * SS + j] = 0.0;
    out[SS * SS + i * SS + j] = 0.0;
    out[2u * SS * SS + i * SS + j] = 0.0;
    return;
  }

  let idx = i * p.S2 + j * p.S + sk;
  let u = U[idx];
  let lbl = label[idx];
  let Zlbl = atoms[lbl].Z;
  out[i * SS + j] = u * u;
  // Pack both Z and domain label: Z + label/1000 (label recoverable as fract * 1000)
  out[SS * SS + i * SS + j] = f32(lbl) + Zlbl * 1000.0;

  // Boundary detection: 1.0 = active (density present), 2.0 = broken (density ~zero both sides)
  let dens = u * u;
  var bnd = 0.0;
  if (i > 1u && i < p.NN - 1u) {
    let nbIdx = idx + p.S2;
    if (lbl != label[nbIdx]) {
      let nbDens = U[nbIdx] * U[nbIdx];
      if (dens > 1e-6 || nbDens > 1e-6) {
        bnd = 1.0;  // active boundary — bond exists
      } else {
        bnd = 2.0;  // broken boundary — no density on either side
      }
    }
  }
  if (bnd < 1.5 && j > 1u && j < p.NN - 1u) {
    let nbIdx = idx + p.S;
    if (lbl != label[nbIdx]) {
      let nbDens = U[nbIdx] * U[nbIdx];
      if (dens > 1e-6 || nbDens > 1e-6) {
        bnd = max(bnd, 1.0);
      } else {
        bnd = 2.0;
      }
    }
  }
  out[2u * SS * SS + i * SS + j] = bnd;
}
`;

// === Nuclear dynamics shaders ===

// Two force methods: HF integral (small systems) and gradient-of-P (large systems)
const USE_GRADP_FORCE = window.USER_GRADP_FORCE !== undefined ? window.USER_GRADP_FORCE : true;  // default on
const FORCE_RADIUS = USE_GRADP_FORCE ? Math.max(5, Math.round(3.0 / hGrid)) : 0;  // 3 au local sphere

const gradPtotal_WGSL = USE_GRADP_FORCE ? `
// Averaged gradient of total electron potential P in sphere around nucleus
// F_A = 2 * Z_A * <∇P>_R — uses converged P from multigrid V-cycle
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Pt: array<f32>;
@group(0) @binding(2) var<storage, read_write> forceSums: array<f32>;
@group(0) @binding(3) var<storage, read> atoms: array<Atom>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let atom = gid.x;
  if (atom >= ${NELEC}u) { return; }

  let ZA = select(atoms[atom].Z, atoms[atom].Z_nuc, atoms[atom].Z <= 0.0);
  if (ZA <= 0.0) {
    forceSums[atom * 3u] = 0.0;
    forceSums[atom * 3u + 1u] = 0.0;
    forceSums[atom * 3u + 2u] = 0.0;
    return;
  }

  let ci = i32(atoms[atom].posI);
  let cj = i32(atoms[atom].posJ);
  let ck = i32(atoms[atom].posK);
  let R = ${FORCE_RADIUS}i;
  let R2f = f32(R * R);
  let inv2h = 0.5 * p.inv_h;
  let S = i32(p.S);

  var sumFi: f32 = 0.0;
  var sumFj: f32 = 0.0;
  var sumFk: f32 = 0.0;
  var count: f32 = 0.0;

  for (var di: i32 = -R; di <= R; di++) {
    let ii = ci + di;
    if (ii < 1 || ii >= S - 1) { continue; }
    for (var dj: i32 = -R; dj <= R; dj++) {
      let jj = cj + dj;
      if (jj < 1 || jj >= S - 1) { continue; }
      for (var dk: i32 = -R; dk <= R; dk++) {
        let kk = ck + dk;
        if (kk < 1 || kk >= S - 1) { continue; }
        let r2 = f32(di*di + dj*dj + dk*dk);
        if (r2 > R2f) { continue; }

        let ui = u32(ii); let uj = u32(jj); let uk = u32(kk);
        let gI = (Pt[(ui+1u)*p.S2 + uj*p.S + uk] - Pt[(ui-1u)*p.S2 + uj*p.S + uk]) * inv2h;
        let gJ = (Pt[ui*p.S2 + (uj+1u)*p.S + uk] - Pt[ui*p.S2 + (uj-1u)*p.S + uk]) * inv2h;
        let gK = (Pt[ui*p.S2 + uj*p.S + (uk+1u)] - Pt[ui*p.S2 + uj*p.S + (uk-1u)]) * inv2h;
        sumFi += gI;
        sumFj += gJ;
        sumFk += gK;
        count += 1.0;
      }
    }
  }

  if (count > 0.0) {
    sumFi /= count;
    sumFj /= count;
    sumFk /= count;
  }

  forceSums[atom * 3u]      = 2.0 * ZA * sumFi;
  forceSums[atom * 3u + 1u] = 2.0 * ZA * sumFj;
  forceSums[atom * 3u + 2u] = 2.0 * ZA * sumFk;
}
` :
// Direct Coulomb force (HF integral) for small systems
`
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read_write> forceSums: array<f32>;
@group(0) @binding(3) var<storage, read> atoms: array<Atom>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let atom = gid.x;
  if (atom >= ${NELEC}u) { return; }

  let ZA = select(atoms[atom].Z, atoms[atom].Z_nuc, atoms[atom].Z <= 0.0);
  if (ZA <= 0.0) {
    forceSums[atom * 3u] = 0.0;
    forceSums[atom * 3u + 1u] = 0.0;
    forceSums[atom * 3u + 2u] = 0.0;
    return;
  }

  let Ri = f32(atoms[atom].posI) * p.h;
  let Rj = f32(atoms[atom].posJ) * p.h;
  let Rk = f32(atoms[atom].posK) * p.h;
  let h3 = p.h * p.h * p.h;
  let soft = p.h2;

  var fi: f32 = 0.0;
  var fj: f32 = 0.0;
  var fk: f32 = 0.0;

  let NM = i32(p.NN) - 1;
  for (var i: i32 = 1; i <= NM; i++) {
    let xi = f32(i) * p.h;
    let di = xi - Ri;
    for (var j: i32 = 1; j <= NM; j++) {
      let yj = f32(j) * p.h;
      let dj = yj - Rj;
      for (var k: i32 = 1; k <= NM; k++) {
        let zk = f32(k) * p.h;
        let dk = zk - Rk;
        let r2 = di*di + dj*dj + dk*dk + soft;
        let r = sqrt(r2);
        let inv_r3 = 1.0 / (r * r2);
        let id = u32(i) * p.S2 + u32(j) * p.S + u32(k);
        let rho = U[id] * U[id];
        let w = rho * inv_r3 * h3;
        fi += di * w;
        fj += dj * w;
        fk += dk * w;
      }
    }
  }

  forceSums[atom * 3u]      = ZA * fi;
  forceSums[atom * 3u + 1u] = ZA * fj;
  forceSums[atom * 3u + 2u] = ZA * fk;
}
`;

// Recompute nuclear potential K on GPU after nuclei move — adapted for atomBuf (loop)
// Analytical inner-sphere corrections for H (ψ = exp(-r)/√π, r < R_SING)
// T_inner = 2·[1/4 - exp(-2a)(a²/2 + a/2 + 1/4)]
// V_inner = -4·[1/4 - exp(-2a)(a/2 + 1/4)]
// No analytical inner-sphere corrections
const recomputeK_WGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> K: array<f32>;
@group(0) @binding(2) var<storage, read> atoms: array<Atom>;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = cellIdx(gid);
  if (id >= p.S3) { return; }

  let k = id % p.S;
  let j = (id / p.S) % p.S;
  let i = id / p.S2;

  var Kval: f32 = 0.0;
  let soft_k = p.h2;  // Coulomb softening: r=sqrt(r²+h²) matching original
  for (var n: u32 = 0u; n < ${NELEC}u; n++) {
    let Za = atoms[n].Z;
    let Zn = select(Za, atoms[n].Z_nuc, atoms[n].Z_nuc > 0.0);
    if (Za <= 0.0 && Zn <= 0.0) { continue; }
    let di = (f32(i) - f32(atoms[n].posI)) * p.h;
    let dj = (f32(j) - f32(atoms[n].posJ)) * p.h;
    let dk = (f32(k) - f32(atoms[n].posK)) * p.h;
    let r2 = di*di + dj*dj + dk*dk;
    // Bare atoms (rc=0): clamp at R_SING. Pseudopotential (rc>0): hard cutoff at rc.
    let r_eff = select(max(sqrt(r2), ${R_SING}), max(sqrt(r2), atoms[n].rc), atoms[n].rc > 0.0);
    Kval += Zn / r_eff;
  }
  K[id] = Kval;
}
`;
const WG_RECOMPUTE_K = Math.ceil(S3 / 256);

// GPU init: split into two shaders to handle large atom counts
// Phase 1: accumulate K, find dominating atom by highest exp(-Z*r)
// Phase 2: set U from bestU values
const INIT_BATCH = 50;  // atoms per dispatch batch
const gpuInitAccumWGSL = `
${paramStructWGSL}
${atomStructWGSL}
struct Range { start: u32, count: u32, _p0: u32, _p1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> atoms: array<Atom>;
@group(0) @binding(2) var<storage, read_write> K: array<f32>;
@group(0) @binding(3) var<storage, read_write> bestU: array<f32>;
@group(0) @binding(4) var<storage, read_write> label: array<u32>;
@group(0) @binding(5) var<uniform> range: Range;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = cellIdx(gid);
  if (id >= p.S3) { return; }
  let kk = id % p.S;
  let j = (id / p.S) % p.S;
  let i = id / p.S2;
  let xi = f32(i) * p.h;
  let yj = f32(j) * p.h;
  let zk = f32(kk) * p.h;

  var Kval: f32 = K[id];
  var bU: f32 = bestU[id];
  var bestN: u32 = label[id];
  let soft_k = p.h2;  // Coulomb softening for bare atoms, matching original
  let end = range.start + range.count;

  for (var n: u32 = range.start; n < end; n++) {
    let Za = atoms[n].Z;
    let Zn = select(Za, atoms[n].Z_nuc, atoms[n].Z_nuc > 0.0); // nuclear charge (defaults to Z)
    if (Za <= 0.0 && Zn <= 0.0) { continue; }
    let dx = xi - f32(atoms[n].posI) * p.h;
    let dy = yj - f32(atoms[n].posJ) * p.h;
    let dz = zk - f32(atoms[n].posK) * p.h;
    let r2 = dx*dx + dy*dy + dz*dz;
    // Bare atoms (rc=0): clamp at R_SING. Pseudopotential (rc>0): hard cutoff at rc.
    let r = select(max(sqrt(r2), ${R_SING}), max(sqrt(r2 + 0.04 * p.h2), atoms[n].rc), atoms[n].rc > 0.0);
    Kval += Zn / r;
    // Normalized trial: ∫U²dV = Z_eff analytically (U = Zeff²/√π · exp(-Zeff·r))
    // Domains assigned by highest normalized density
    // Skip wavefunction init for bare protons (Za=0): no electron domain
    let Ze = atoms[n].initZeff;
    let rReal = sqrt(r2);
    let rCutInit = atoms[n].initRcut;
    let uTrial = select(Ze * Ze * ${(1/Math.sqrt(Math.PI)).toFixed(10)} * exp(-Ze * r), 0.0, rReal > rCutInit || Za <= 0.0);
    if (uTrial > bU) { bU = uTrial; bestN = n; }
  }

  K[id] = Kval;
  bestU[id] = bU;
  label[id] = bestN;
}
`;

// Phase 2: set U directly from bestU (= exp(-Z*r) of winning atom)
const gpuInitFinalWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> bestU: array<f32>;
@group(0) @binding(2) var<storage, read_write> U: array<f32>;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = cellIdx(gid);
  if (id >= p.S3) { return; }
  U[id] = bestU[id];
}
`;

// ===== LOBPCG SHADERS =====

// Apply Hamiltonian: HU = -½∇²U/h² + V_eff·U, where V_eff = -K + 2P
// Same domain boundary/exclusion logic as updateU, but outputs H·U instead of time-stepping
const applyH_WGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> Ui: array<f32>;
@group(0) @binding(3) var<storage, read> label: array<u32>;
@group(0) @binding(4) var<storage, read> Pi: array<f32>;
@group(0) @binding(5) var<storage, read_write> HU: array<f32>;
@group(0) @binding(6) var<storage, read> atoms: array<Atom>;

fn distToAtom(ci: u32, cj: u32, ck: u32, n: u32) -> f32 {
  let dx = (f32(ci) - f32(atoms[n].posI)) * p.h;
  let dy = (f32(cj) - f32(atoms[n].posJ)) * p.h;
  let dz = (f32(ck) - f32(atoms[n].posK)) * p.h;
  return sqrt(dx*dx + dy*dy + dz*dz);
}

fn isInsideRc(ci: u32, cj: u32, ck: u32, lbl: u32) -> bool {
  let rc = atoms[lbl].rc;
  if (rc <= 0.0) { return false; }
  return distToAtom(ci, cj, ck, lbl) < rc;
}

fn isInsideAnalytical(ci: u32, cj: u32, ck: u32, lbl: u32) -> bool {
  return false;  // disabled: bare atoms use Coulomb softening
}

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }

  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let myL = label[id];

  // Inside r_c exclusion: HU = 0
  if (isInsideRc(i, j, k, myL)) {
    HU[id] = 0.0;
    return;
  }

  let uc = Ui[id];

  // Neumann BC at domain boundaries and r_c surfaces
  let l_ip = label[id + p.S2]; let excl_ip = isInsideRc(i+1u, j, k, l_ip);
  let l_im = label[id - p.S2]; let excl_im = isInsideRc(i-1u, j, k, l_im);
  let l_jp = label[id + p.S];  let excl_jp = isInsideRc(i, j+1u, k, l_jp);
  let l_jm = label[id - p.S];  let excl_jm = isInsideRc(i, j-1u, k, l_jm);
  let l_kp = label[id + 1u];   let excl_kp = isInsideRc(i, j, k+1u, l_kp);
  let l_km = label[id - 1u];   let excl_km = isInsideRc(i, j, k-1u, l_km);

  let u_ip = select(uc, Ui[id + p.S2], l_ip == myL && !excl_ip);
  let u_im = select(uc, Ui[id - p.S2], l_im == myL && !excl_im);
  let u_jp = select(uc, Ui[id + p.S],  l_jp == myL && !excl_jp);
  let u_jm = select(uc, Ui[id - p.S],  l_jm == myL && !excl_jm);
  let u_kp = select(uc, Ui[id + 1u],   l_kp == myL && !excl_kp);
  let u_km = select(uc, Ui[id - 1u],   l_km == myL && !excl_km);

  let lap = u_ip + u_im + u_jp + u_jm + u_kp + u_km - 6.0 * uc;

  // H·U = -½∇²U + V_eff·U where V_eff = -K + 2P
  HU[id] = -0.5 * lap * p.inv_h2 + (-K[id] + 2.0 * Pi[id]) * uc;
}
`;

// LOBPCG per-domain inner products: batched reduction
// Computes 6 products per domain for the Ritz problem:
//   <X|X>, <X|HX>, <W|W>, <W|HW>, <X|W>, <X|HW>  (2-vector Ritz)
// Additional 6 for 3-vector Ritz when P is available:
//   <P|P>, <P|HP>, <X|P>, <X|HP>, <W|P>, <W|HP>
const NRED_LOBPCG = 12;  // 12 inner products per domain
const MAX_LOBPCG_WG = Math.min(256, Math.ceil(INTERIOR / REDUCE_WG));
const lobpcgInnerWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> HX: array<f32>;
@group(0) @binding(3) var<storage, read> W: array<f32>;
@group(0) @binding(4) var<storage, read> HW: array<f32>;
@group(0) @binding(5) var<storage, read> Pv: array<f32>;
@group(0) @binding(6) var<storage, read> HP: array<f32>;
@group(0) @binding(7) var<storage, read_write> partials: array<f32>;
@group(0) @binding(8) var<storage, read> label: array<u32>;

struct DomIdx { idx: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(9) var<uniform> dom: DomIdx;

var<workgroup> sn: array<f32, ${NRED_LOBPCG * REDUCE_WG}>;

@compute @workgroup_size(${REDUCE_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(workgroup_id) wgid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let stride = ${MAX_LOBPCG_WG}u * ${REDUCE_WG}u;
  let NR = ${NRED_LOBPCG}u;

  for (var q: u32 = 0u; q < NR; q++) { sn[lid * NR + q] = 0.0; }

  var cell = gid.x;
  loop {
    if (cell >= tot) { break; }
    let k = (cell % NM) + 1u;
    let j = ((cell / NM) % NM) + 1u;
    let i = (cell / (NM * NM)) + 1u;
    let id = i * p.S2 + j * p.S + k;

    if (label[id] == dom.idx) {
      let x = X[id]; let hx = HX[id]; let w = W[id]; let hw = HW[id];
      let pv = Pv[id]; let hp = HP[id];
      let dV = p.h3;
      sn[lid * NR + 0u]  += x * x * dV;     // <X|X>
      sn[lid * NR + 1u]  += x * hx * dV;    // <X|HX>
      sn[lid * NR + 2u]  += w * w * dV;     // <W|W>
      sn[lid * NR + 3u]  += w * hw * dV;    // <W|HW>
      sn[lid * NR + 4u]  += x * w * dV;     // <X|W>
      sn[lid * NR + 5u]  += x * hw * dV;    // <X|HW>
      sn[lid * NR + 6u]  += pv * pv * dV;   // <P|P>
      sn[lid * NR + 7u]  += pv * hp * dV;   // <P|HP>
      sn[lid * NR + 8u]  += x * pv * dV;    // <X|P>
      sn[lid * NR + 9u]  += x * hp * dV;    // <X|HP>  (= <HX|P> by symmetry)
      sn[lid * NR + 10u] += w * pv * dV;    // <W|P>
      sn[lid * NR + 11u] += w * hp * dV;    // <W|HP>
    }

    cell = cell + stride;
  }

  workgroupBarrier();

  for (var s: u32 = ${REDUCE_WG >> 1}u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var q: u32 = 0u; q < NR; q++) {
        sn[lid * NR + q] += sn[(lid + s) * NR + q];
      }
    }
    workgroupBarrier();
  }

  if (lid == 0u) {
    let base = wgid.x * NR;
    for (var q: u32 = 0u; q < NR; q++) {
      partials[base + q] = sn[q];
    }
  }
}
`;

const lobpcgFinalizeWGSL = `
struct NWG { count: u32 }
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> sums: array<f32>;
@group(0) @binding(2) var<uniform> nwg: NWG;

var<workgroup> wg: array<f32, ${NRED_LOBPCG * REDUCE_WG}>;

@compute @workgroup_size(${REDUCE_WG})
fn main(@builtin(local_invocation_index) lid: u32) {
  let NR = ${NRED_LOBPCG}u;
  for (var q: u32 = 0u; q < NR; q++) { wg[lid * NR + q] = 0.0; }

  for (var i: u32 = lid; i < nwg.count; i += ${REDUCE_WG}u) {
    for (var q: u32 = 0u; q < NR; q++) {
      wg[lid * NR + q] += partials[i * NR + q];
    }
  }

  workgroupBarrier();

  for (var s: u32 = ${REDUCE_WG >> 1}u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var q: u32 = 0u; q < NR; q++) {
        wg[lid * NR + q] += wg[(lid + s) * NR + q];
      }
    }
    workgroupBarrier();
  }

  if (lid == 0u) {
    for (var q: u32 = 0u; q < NR; q++) {
      sums[q] = wg[q];
    }
  }
}
`;

// Compute residual W = HX - lambda*X and precondition: W /= (diag(H) + shift)
// lambda is passed via uniform, diag(H) ≈ 3/h² + |V|
const lobpcgResidualWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> HX: array<f32>;
@group(0) @binding(3) var<storage, read_write> W: array<f32>;
@group(0) @binding(4) var<storage, read> label: array<u32>;
@group(0) @binding(5) var<storage, read> K: array<f32>;
@group(0) @binding(6) var<storage, read> Pot: array<f32>;

struct Lambda { val: f32, domIdx: u32, _p0: u32, _p1: u32 }
@group(0) @binding(7) var<uniform> lam: Lambda;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }

  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  if (label[id] != lam.domIdx) {
    W[id] = 0.0;
    return;
  }

  let r = HX[id] - lam.val * X[id];
  // Diagonal preconditioner: diag(H) = 3/h² + |V_eff|
  let vEff = abs(-K[id] + 2.0 * Pot[id]);
  let diagH = 3.0 * p.inv_h2 + vEff + 0.1;  // +0.1 shift for stability
  W[id] = r / diagH;
}
`;

// LOBPCG update: X_new = c0*X + c1*W + c2*P, P_new = c1*W + c2*P
// Coefficients from Ritz solve, passed via uniform buffer
const lobpcgUpdateWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> Pv: array<f32>;
@group(0) @binding(4) var<storage, read> label: array<u32>;

struct Coeff { c0: f32, c1: f32, c2: f32, domIdx: u32 }
@group(0) @binding(5) var<uniform> coeff: Coeff;

${cellIdxWGSL}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let cell = cellIdx(gid);
  if (cell >= tot) { return; }

  let k = (cell % NM) + 1u;
  let j = ((cell / NM) % NM) + 1u;
  let i = (cell / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  if (label[id] != coeff.domIdx) { return; }

  let x = X[id]; let w = W[id]; let pv = Pv[id];
  X[id] = coeff.c0 * x + coeff.c1 * w + coeff.c2 * pv;
  Pv[id] = coeff.c1 * w + coeff.c2 * pv;
}
`;

// ===== GPU STATE =====
let device, paramsBuf, atomBuf, K_buf, sumsBuf, sumsReadBuf, sliceBuf, sliceReadBuf, partialsBuf, numWGBuf;
let normAtomicBuf, normFloatBuf, initOffsetBuf, bestR2Buf, initRangeBuf;
let U_buf = [], P_buf = [], labelBuf, label2Buf, W_buf;
let rhoTotalBuf, residualBuf, Pc_buf = [], coarseRhsBuf;
let PotherBuf, PselfScratchBuf, sicBuf, sicResidualBuf, domainBufs = [];
let updatePL, evolveBoundaryPL, fixBoundaryUPL, jacobiSmoothPL;
let reduceEnergyPL, finalizeEnergyPL, accumNormsPL, decodeNormsPL, normalizePL, extractPL;
let gpuInitAccumPL, gpuInitFinalPL;
let computeRhoPL, computeResidualPL, restrictPL, coarseSmoothPL, prolongCorrectPL;
let computeRhoSelfPL, subtractPselfPL;
let computeRhoOtherPL, copyPotherForLabelPL, initPdirectPL;
let P_directBuf = [], P_directScratchBuf;
let computeRhoOtherBG = [], jacobiDirectBG = [], copyPotherForLabelBG = [], initPdirectBG = [];
let updateBG = [], evolveBoundaryBG = [], fixBoundaryUBG = [], jacobiFineBG = [];
let reduceEnergyBG = [], finalizeEnergyBG, accumNormsBG = [], decodeNormsBG, normalizeBG = [], extractBG = [];
let gpuInitAccumBG, gpuInitFinalBG;
let computeRhoBG = [], residualBG = [], prolongCorrectBG;
let restrictBG, coarseSmoothBG = [];
let computeRhoSelfBG = [], jacobiSelfBG = [], subtractPselfBG = [];
let sicResidualBG, sicRestrictBG, sicProlongBG;
// Nuclear dynamics GPU state
let forceSumsBuf, forceSumsReadBuf;
let gradPtotalPL, recomputeK_PL;
let gradPtotalBG = [], recomputeK_BG;
// LOBPCG state
let useLOBPCG = false;  // toggle: false = imaginary time, true = LOBPCG
const LOBPCG_ITERS = 10;  // LOBPCG iterations per SCF step
let applyH_PL, lobpcgResidualPL, lobpcgInnerPL, lobpcgFinalizePL, lobpcgUpdatePL;
let applyH_BG_X, applyH_BG_W, applyH_BG_P;
let lobpcgResidualBG, lobpcgUpdateBG;
let lobpcgInnerBG = [], lobpcgFinalizeBG;
let HX_buf, W_lobpcg_buf, HW_buf, P_lobpcg_buf, HP_buf;
let lobpcgPartialsBuf, lobpcgSumsBuf, lobpcgSumsReadBuf;
let lambdaBuf, coeffBuf, lobpcgNumWGBuf;
// Chebyshev state
let useCheb = false;
let chebOmega = 1.0;  // Chebyshev ω recurrence state
let chebyshevStepPL, chebyshevStepBG = [];
let U_prev_buf, chebParamsBuf;

let cur = 0, gpuReady = false, computing = false, initProgress = 0, computePromise = null;
let tStep = 0, E = 0, lastMs = 0;
let E_T = 0, E_eK = 0, E_ee = 0, E_KK = 0, dipole_au = 0, dipole_D = 0, E_bind = 0;

// Isolated atom energies via 1D variational: U(r) = A·exp(-α·r), r > r_c
// Minimizes E(α) = T + V_eK = ½Zα² - Z·(Z/I₂)·I₁
function isolatedAtomEnergy(Zn, rc) {
  if (Zn <= 0) return 0;
  if (Zn === 1 && rc === 0) return -0.5;  // exact hydrogen
  let bestE = 0;
  const aMax = Zn * 4;
  for (let a = 0.1; a <= aMax; a += 0.005) {
    const c = 2 * a;
    const erc = Math.exp(-c * rc);
    // I₂ = 4π ∫(rc..∞) r² exp(-cr) dr
    const I2 = 4 * Math.PI * erc * (rc * rc / c + 2 * rc / (c * c) + 2 / (c * c * c));
    // I₁ = 4π ∫(rc..∞) r exp(-cr) dr  (for ∫U²/r dV)
    const I1 = 4 * Math.PI * erc * (rc / c + 1 / (c * c));
    if (I2 <= 0) continue;
    const T = 0.5 * Zn * a * a;
    const V = -Zn * (Zn / I2) * I1;
    const E = T + V;
    if (E < bestE) bestE = E;
  }
  return bestE;
}

// Precompute isolated atom energies for all unique (Z, r_c) pairs
const _atomRefE = {};
for (let n = 0; n < NELEC; n++) {
  if (_uz[n] <= 0) continue;
  const key = _uz[n] + '_' + r_cut[n].toFixed(3);
  if (!(key in _atomRefE)) {
    _atomRefE[key] = isolatedAtomEnergy(_uz[n], r_cut[n]);
  }
}
let E_atoms_sum = 0;
for (let n = 0; n < NELEC; n++) {
  if (_uz[n] <= 0) continue;
  const key = _uz[n] + '_' + r_cut[n].toFixed(3);
  E_atoms_sum += _atomRefE[key];
}
console.log("Isolated atom energies:", _atomRefE, "Sum:", E_atoms_sum.toFixed(6));
let gpuError = null;

// Single phase run
let phase = 0, phaseSteps = 0, frameCount = 0;
const TOTAL_STEPS = window.USER_STEPS || 20000;
let addNucRepulsion = window.NO_NUC_REPULSION ? false : true;
let vcycleEnabled = true;
let vcycleCount = 0;

const SLICE_SIZE = (3 * S * S + 2 * S) * 4;  // 3 image slices (density, Z, boundary) + K line + W line
const WG_UPDATE = Math.ceil(INTERIOR / 256);
const WG_REDUCE = Math.min(MAX_REDUCE_WG, Math.ceil(INTERIOR / REDUCE_WG));
const WG_NORM = Math.ceil(INTERIOR / 256);
const WG_EXTRACT = Math.ceil(S / 16);
const WG_COARSE = Math.ceil(INTERIOR_C / 256);
const SUMS_BYTES = NRED_E * 4;  // 24 bytes: T, V_eK, V_ee, dipX, dipY, dipZ

let sliceData = null;
let sliceK = N2;  // z-slice index for visualization (scrollable with arrow keys)

function fillParamsBuf(pb) {
  const pu = new Uint32Array(pb);
  const pf = new Float32Array(pb);
  pu[0] = NN; pu[1] = S; pu[2] = S2; pu[3] = S3;
  pu[4] = N2; pf[5] = boundarySpeed; pf[6] = curvReg; pf[7] = 2 * Math.PI;
  pf[8] = hGrid; pf[9] = h2v; pf[10] = 1 / hGrid; pf[11] = 1 / h2v;
  pf[12] = dtv; pf[13] = half_dv; pf[14] = h3v; pu[15] = sliceK;
}

function fillAtomBuf() {
  const ab = new ArrayBuffer(ATOM_BUF_BYTES);
  const au = new Uint32Array(ab);
  const af = new Float32Array(ab);
  for (let n = 0; n < MAX_ATOMS; n++) {
    const off = n * ATOM_STRIDE;
    au[off] = nucPos[n][0];
    au[off + 1] = nucPos[n][1];
    au[off + 2] = nucPos[n][2];
    af[off + 3] = Z[n];
    af[off + 4] = r_cut[n];
    af[off + 5] = Z_nuc[n]; // nuclear charge (may differ from Z for bare protons)
    // Per-atom init: initZeff controls exp(-Zeff*r), initRcut limits domain radius
    const perZeff = window.USER_INIT_ZEFF;
    const perRcut = window.USER_INIT_RCUT;
    af[off + 6] = perZeff ? perZeff[n] || Z[n] : (window.INIT_ZEFF || Z[n]);
    af[off + 7] = perRcut ? perRcut[n] || 1e6 : (window.INIT_RCUT || 1e6);
  }
  device.queue.writeBuffer(atomBuf, 0, ab);
}

async function uploadInitialData() {
  console.log("GPU Init: " + NELEC + " atoms, NN=" + NN + " S3=" + S3);

  // If USER_INIT_POS is set, temporarily move atoms to electron positions for wavefunction init
  const initPos = window.USER_INIT_POS;
  const savedPos = [];
  if (initPos) {
    for (let n = 0; n < NELEC; n++) {
      savedPos.push([nucPos[n][0], nucPos[n][1], nucPos[n][2]]);
      if (initPos[n]) {
        nucPos[n][0] = initPos[n][0];
        nucPos[n][1] = initPos[n][1];
        // keep k (z) the same
        console.log("Atom " + n + ": init wavefunction at (" + initPos[n][0] + "," + initPos[n][1] + ") nucleus at (" + savedPos[n][0] + "," + savedPos[n][1] + ")");
      }
    }
  }

  fillAtomBuf(); // upload with electron positions for U init

  const t0 = performance.now();
  const WG_INIT = Math.ceil(S3 / 256);

  // Clear K, labels, and fill bestR2 with large values
  const clrEnc = device.createCommandEncoder();
  clrEnc.clearBuffer(K_buf);
  clrEnc.clearBuffer(labelBuf);
  clrEnc.clearBuffer(U_buf[0]);
  clrEnc.clearBuffer(P_buf[0]);
  device.queue.submit([clrEnc.finish()]);

  // Fill bestU with 0 from CPU (no density initially)
  const bU = new Float32Array(S3);
  bU.fill(0.0);
  device.queue.writeBuffer(bestR2Buf, 0, bU);

  // Batched init: process INIT_BATCH atoms per dispatch
  const nBatches = Math.ceil(NELEC / INIT_BATCH);
  console.log("GPU init: " + nBatches + " batches of " + INIT_BATCH + " atoms, " + WG_INIT + " workgroups");

  for (let b = 0; b < nBatches; b++) {
    const start = b * INIT_BATCH;
    const count = Math.min(INIT_BATCH, NELEC - start);
    device.queue.writeBuffer(initRangeBuf, 0, new Uint32Array([start, count, 0, 0]));
    const enc = device.createCommandEncoder();
    const ip = enc.beginComputePass();
    ip.setPipeline(gpuInitAccumPL);
    ip.setBindGroup(0, gpuInitAccumBG);
    dispatchLinear(ip, S3);
    ip.end();
    device.queue.submit([enc.finish()]);
  }

  // Restore nucleus positions and recompute K potential
  if (initPos && savedPos.length > 0) {
    for (let n = 0; n < NELEC; n++) {
      nucPos[n][0] = savedPos[n][0];
      nucPos[n][1] = savedPos[n][1];
      nucPos[n][2] = savedPos[n][2];
    }
    fillAtomBuf(); // re-upload with true nucleus positions
    // Recompute K with correct nucleus positions
    {
      const kEnc = device.createCommandEncoder();
      kEnc.clearBuffer(K_buf);
      device.queue.submit([kEnc.finish()]);
      const kEnc2 = device.createCommandEncoder();
      const kp = kEnc2.beginComputePass();
      kp.setPipeline(recomputeK_PL);
      kp.setBindGroup(0, recomputeK_BG);
      dispatchLinear(kp, S3);
      kp.end();
      device.queue.submit([kEnc2.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    console.log("Restored nucleus positions — K recomputed at proton locations");
  }

  // Final pass: set U and P from bestR2 (labels already set by accum)
  {
    device.pushErrorScope('validation');
    const enc = device.createCommandEncoder();
    const ip = enc.beginComputePass();
    ip.setPipeline(gpuInitFinalPL);
    ip.setBindGroup(0, gpuInitFinalBG);
    dispatchLinear(ip, S3);
    ip.end();
    device.queue.submit([enc.finish()]);
    const initErr = await device.popErrorScope();
    if (initErr) console.error("GPU init error:", initErr.message);
  }

  await device.queue.onSubmittedWorkDone();
  console.log("GPU init dispatched in " + ((performance.now() - t0)).toFixed(0) + "ms");

  // Debug: read back a few U values to verify init worked
  {
    const dbgBuf = device.createBuffer({ size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const dbgEnc = device.createCommandEncoder();
    dbgEnc.copyBufferToBuffer(U_buf[0], N2 * S2 * 4, dbgBuf, 0, 256);
    device.queue.submit([dbgEnc.finish()]);
    await dbgBuf.mapAsync(GPUMapMode.READ);
    const dbgData = new Float32Array(dbgBuf.getMappedRange().slice(0));
    dbgBuf.unmap(); dbgBuf.destroy();
    let dbgMax = 0, dbgNonZero = 0;
    for (let i = 0; i < dbgData.length; i++) {
      if (dbgData[i] !== 0) dbgNonZero++;
      if (Math.abs(dbgData[i]) > dbgMax) dbgMax = Math.abs(dbgData[i]);
    }
    console.log("Init U debug: max=" + dbgMax.toExponential(3) + " nonZero=" + dbgNonZero + "/64 sample: " +
      dbgData[0].toExponential(3) + " " + dbgData[32].toExponential(3));
    window._initDebug = "initU: max=" + dbgMax.toExponential(2) + " nz=" + dbgNonZero;
  }

  // Shell-based initialization: override Voronoi with radial shells
  // SHELL_INIT = [ { domains: [0,1], rmax: R, zeff: Z }, { domains: [2], rmin: R, zeff: Z }, ... ]
  if (window.SHELL_INIT) {
    console.log("Shell init: overriding Voronoi with radial shells");
    const uData = new Float32Array(S3);
    const lData = new Uint32Array(S3);
    // Center = average of all atom positions
    let cx = 0, cy = 0, cz = 0;
    for (let a = 0; a < NELEC; a++) {
      cx += nucPos[a][0]; cy += nucPos[a][1]; cz += nucPos[a][2];
    }
    cx /= NELEC; cy /= NELEC; cz /= NELEC;

    for (let i = 0; i < S; i++) {
      for (let j = 0; j < S; j++) {
        for (let k = 0; k < S; k++) {
          const id = i * S2 + j * S + k;
          const r = Math.sqrt(((i - cx) * hGrid) ** 2 + ((j - cy) * hGrid) ** 2 + ((k - cz) * hGrid) ** 2);

          // Find which shell this point belongs to
          let assigned = false;
          for (const shell of window.SHELL_INIT) {
            const rmin = shell.rmin || 0;
            const rmax = shell.rmax || Infinity;
            if (r >= rmin && r < rmax) {
              // Assign to one of the shell's domains based on angular partition
              const doms = shell.domains;
              let domIdx;
              if (doms.length === 1) {
                domIdx = doms[0];
              } else {
                // Split hemisphere: use angle from x-axis for 2, thirds for 3, etc.
                const angle = Math.atan2((j - cy), (i - cx));
                const sector = Math.floor((angle + Math.PI) / (2 * Math.PI) * doms.length);
                domIdx = doms[Math.min(sector, doms.length - 1)];
              }
              lData[id] = domIdx;
              const zeff = shell.zeff || 1;
              uData[id] = zeff * zeff / Math.sqrt(Math.PI) * Math.exp(-zeff * r);
              assigned = true;
              break;
            }
          }
          if (!assigned) {
            lData[id] = 0;
            uData[id] = 0;
          }
        }
      }
    }
    // Apply custom psi modification if provided
    if (window.CUSTOM_PSI_INIT) {
      window.CUSTOM_PSI_INIT(S, hGrid, uData, lData);
    }
    device.queue.writeBuffer(U_buf[0], 0, uData);
    device.queue.writeBuffer(labelBuf, 0, lData);
    console.log("Shell init complete");
  }

  // Custom label initialization: override Voronoi domain assignment
  if (window.CUSTOM_LABEL_INIT) {
    console.log("Custom label init: overriding Voronoi domains");
    const lData = window.CUSTOM_LABEL_INIT(S, hGrid, nucPos, NELEC);
    device.queue.writeBuffer(labelBuf, 0, lData);
    console.log("Custom label init complete");
  }

  // Copy to double-buffered slots
  const cpEnc = device.createCommandEncoder();
  cpEnc.copyBufferToBuffer(U_buf[0], 0, U_buf[1], 0, S3 * 4);
  cpEnc.copyBufferToBuffer(P_buf[0], 0, P_buf[1], 0, S3 * 4);
  cpEnc.copyBufferToBuffer(labelBuf, 0, label2Buf, 0, S3 * 4);
  cpEnc.copyBufferToBuffer(P_buf[0], 0, PotherBuf, 0, S3 * 4);
  device.queue.submit([cpEnc.finish()]);

  // Fill W with 1.0
  const Wd = new Float32Array(S3); Wd.fill(1.0);
  device.queue.writeBuffer(W_buf, 0, Wd);

  await device.queue.onSubmittedWorkDone();
  console.log("GPU init complete in " + ((performance.now() - t0)).toFixed(0) + "ms");
  cur = 0;
}

function updateParamsBuf() {
  const pb = new ArrayBuffer(PARAM_BYTES);
  fillParamsBuf(pb);
  device.queue.writeBuffer(paramsBuf, 0, pb);
  fillAtomBuf();
}

async function startMolPhase() {
  nucPos = molNucPos.map(p => [...p]);
  Z = [...Z_orig]; Ne = [...Ne_orig];
  R_out = 1.0;
  addNucRepulsion = !window.NO_NUC_REPULSION;
  updateParamsBuf();
  await uploadInitialData();
  tStep = 0;
  phaseSteps = 0;
  frameCount = 0;
  E_min = Infinity;
  cur = 0;
  phase = 0;
  console.log("=== Molecule: " + atomLabels.join("-") + " ===");
}

// External API: update atom positions and restart
window.restartWithPositions = async function(newPositions) {
  // newPositions: array of [i, j, k] per atom
  for (let n = 0; n < newPositions.length && n < NELEC; n++) {
    nucPos[n] = [...newPositions[n]];
    molNucPos[n] = [...newPositions[n]];
  }
  R_out = 1.0;
  addNucRepulsion = !window.NO_NUC_REPULSION;
  updateParamsBuf();
  await uploadInitialData();
  tStep = 0;
  phaseSteps = 0;
  frameCount = 0;
  E_min = Infinity;
  cur = 0;
  phase = 0;
  computing = false;
  gpuError = null;
  window._prevE = undefined;
  window._convCount = 0;
};

// External API: activate first `count` atoms, reinitialize simulation
let _activating = false, _pendingCount = null;
window.activateAtoms = async function(count) {
  if (!gpuReady) { console.warn("activateAtoms: GPU not ready"); return; }
  count = Math.max(1, Math.min(count, NELEC));
  if (_activating) { _pendingCount = count; return; }  // queue latest request
  _activating = true;
  // Wait for any in-flight compute to fully complete (including mapAsync)
  phase = 1;  // prevent new doSteps from starting
  if (computePromise) { await computePromise; }
  computing = false;
  await device.queue.onSubmittedWorkDone();
  for (let n = 0; n < MAX_ATOMS; n++) {
    Z[n] = (n < count) ? Z_orig[n] : 0;
    Ne[n] = (n < count) ? Ne_orig[n] : 0;
  }
  nucVel = Array.from({length: MAX_ATOMS}, () => [0, 0, 0]);
  nucForce = Array.from({length: MAX_ATOMS}, () => [0, 0, 0]);
  nucPos = molNucPos.map(p => [...p]);
  updateParamsBuf();
  await uploadInitialData();
  tStep = 0;
  phaseSteps = 0;
  frameCount = 0;
  E_min = Infinity;
  cur = 0;
  phase = 0;
  computing = false;
  gpuError = null;
  nucStepCount = 0;
  // Preserve dynamicsEnabled — don't reset it so user's D-key toggle persists
  // Recalculate isolated atom energy sum for active atoms
  E_atoms_sum = 0;
  for (let n = 0; n < count; n++) {
    if (Z_orig[n] <= 0) continue;
    const key = Z_orig[n] + '_' + r_cut[n].toFixed(3);
    E_atoms_sum += _atomRefE[key];
  }
  window._activeCount = count;
  _activating = false;
  // If another request came in while we were busy, process it
  if (_pendingCount !== null) {
    const next = _pendingCount;
    _pendingCount = null;
    window.activateAtoms(next);
  }
};
window.isGpuReady = function() { return gpuReady; };

function setup() {
  console.log("setup: NELEC=" + NELEC + " NRED_E=" + NRED_E + " REDUCE_WG=" + REDUCE_WG);
  createCanvas(CANVAS_SIZE, CANVAS_SIZE);
  textSize(9);

  // Control sliders (top-right)
  var ctrl = document.createElement('div');
  ctrl.style.cssText = 'position:fixed;top:10px;right:10px;z-index:100;background:rgba(0,0,0,0.7);padding:6px 10px;border-radius:6px;color:#fff;font:11px monospace';
  ctrl.innerHTML =
    'Force: <input id="forceSlider" type="range" min="0" max="500" value="100" style="width:120px;vertical-align:middle"><span id="forceScaleVal"> 1.0x</span><br>' +
    'Boundary: <input id="bndSlider" type="range" min="0" max="200" value="50" style="width:120px;vertical-align:middle"><span id="bndVal"> 0.50</span><br>' +
    'Curvature: <input id="curvSlider" type="range" min="0" max="100" value="15" style="width:120px;vertical-align:middle"><span id="curvVal"> 0.15</span>';
  document.body.appendChild(ctrl);
  document.getElementById('forceSlider').oninput = function() {
    forceScale = this.value / 100;
    document.getElementById('forceScaleVal').textContent = ' ' + forceScale.toFixed(1) + 'x';
  };
  document.getElementById('bndSlider').oninput = function() {
    boundarySpeed = this.value / 100;
    document.getElementById('bndVal').textContent = ' ' + boundarySpeed.toFixed(2);
    updateParamsBuf();
  };
  document.getElementById('curvSlider').oninput = function() {
    curvReg = this.value / 100;
    document.getElementById('curvVal').textContent = ' ' + curvReg.toFixed(2);
    updateParamsBuf();
  };

  initGPU();
}

async function initGPU() {
  try {
    if (!navigator.gpu) {
      gpuError = "WebGPU not supported. Use Chrome 113+ or Safari 17+.";
      return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      gpuError = "No WebGPU adapter found. Check GPU drivers or try Chrome 113+.";
      return;
    }

    try {
      const info = await adapter.requestAdapterInfo();
    } catch (e) {}

    const maxBuf = S3 * 4;
    if (maxBuf > adapter.limits.maxStorageBufferBindingSize || maxBuf > adapter.limits.maxBufferSize) {
      gpuError = "Grid " + NN + "^3 too large for GPU (" + (maxBuf/1e6).toFixed(0) + "MB needed). Use a smaller grid.";
      return;
    }
    device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: maxBuf,
        maxBufferSize: maxBuf
      }
    });

    device.lost.then((info) => {
      gpuError = "GPU device lost: " + info.message;
      gpuReady = false;
    });

    const bs = S3 * 4;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    K_buf = device.createBuffer({ size: bs, usage: usage | GPUBufferUsage.COPY_SRC });
    atomBuf = device.createBuffer({ size: ATOM_BUF_BYTES, usage });
    for (let i = 0; i < 2; i++) {
      U_buf[i] = device.createBuffer({ size: bs, usage: usage | GPUBufferUsage.COPY_SRC });
      P_buf[i] = device.createBuffer({ size: bs, usage: usage | GPUBufferUsage.COPY_SRC });
    }
    labelBuf = device.createBuffer({ size: bs, usage: usage | GPUBufferUsage.COPY_SRC });
    label2Buf = device.createBuffer({ size: bs, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    W_buf = device.createBuffer({ size: bs, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    PotherBuf = device.createBuffer({ size: bs, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    PselfScratchBuf = device.createBuffer({ size: bs, usage });
    sicBuf = device.createBuffer({ size: bs, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    sicResidualBuf = device.createBuffer({ size: bs, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    // Domain index uniform buffers (needed for SIC and direct Pother)
    for (let m = 0; m < NELEC; m++) {
      if (!domainBufs[m]) {
        domainBufs[m] = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const _zf = new Float32Array([Z[m]]);
        const _zu = new Uint32Array(_zf.buffer);
        device.queue.writeBuffer(domainBufs[m], 0, new Uint32Array([m, _zu[0], 0, 0]));
      }
    }
    if (USE_DIRECT_POTHER) {
      // Per-electron persistent P buffers for direct Pother solve
      P_directScratchBuf = device.createBuffer({ size: bs, usage });
      for (let m = 0; m < NELEC; m++) {
        P_directBuf[m] = device.createBuffer({ size: bs, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      }
    }
    // Multigrid buffers
    const cBufSize = SC3 * 4;
    rhoTotalBuf = device.createBuffer({ size: bs, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    residualBuf = device.createBuffer({ size: bs, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    for (let i = 0; i < 2; i++) {
      Pc_buf[i] = device.createBuffer({ size: cBufSize, usage: usage });
    }
    coarseRhsBuf = device.createBuffer({ size: cBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    console.log("Multigrid: fine=" + NN + " coarse=" + NC);

    const partialSize = WG_REDUCE * NRED_E * 4;
    partialsBuf = device.createBuffer({ size: partialSize, usage: GPUBufferUsage.STORAGE });
    normAtomicBuf = device.createBuffer({ size: MAX_ATOMS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    normFloatBuf = device.createBuffer({ size: MAX_ATOMS * 4, usage: GPUBufferUsage.STORAGE });
    sumsBuf = device.createBuffer({ size: SUMS_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    sumsReadBuf = device.createBuffer({ size: SUMS_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    sliceBuf = device.createBuffer({ size: SLICE_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    sliceReadBuf = device.createBuffer({ size: SLICE_SIZE, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    numWGBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    initOffsetBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    bestR2Buf = device.createBuffer({ size: S3 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    initRangeBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(numWGBuf, 0, new Uint32Array([WG_REDUCE, 0, 0, 0]));

    // Force buffers for nuclear dynamics
    const forceSumsSize = NELEC * 3 * 4;
    forceSumsBuf = device.createBuffer({ size: forceSumsSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    forceSumsReadBuf = device.createBuffer({ size: forceSumsSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    // LOBPCG buffers
    HX_buf = device.createBuffer({ size: bs, usage });
    W_lobpcg_buf = device.createBuffer({ size: bs, usage });
    HW_buf = device.createBuffer({ size: bs, usage });
    P_lobpcg_buf = device.createBuffer({ size: bs, usage: usage | GPUBufferUsage.COPY_DST });
    HP_buf = device.createBuffer({ size: bs, usage });
    const lobpcgPartialSize = MAX_LOBPCG_WG * NRED_LOBPCG * 4;
    lobpcgPartialsBuf = device.createBuffer({ size: lobpcgPartialSize, usage: GPUBufferUsage.STORAGE });
    const lobpcgSumsSize = NRED_LOBPCG * 4;
    lobpcgSumsBuf = device.createBuffer({ size: lobpcgSumsSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    lobpcgSumsReadBuf = device.createBuffer({ size: lobpcgSumsSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    lambdaBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    coeffBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    lobpcgNumWGBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(lobpcgNumWGBuf, 0, new Uint32Array([MAX_LOBPCG_WG, 0, 0, 0]));

    // Chebyshev buffers
    U_prev_buf = device.createBuffer({ size: bs, usage: usage | GPUBufferUsage.COPY_DST });
    chebParamsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const pb = new ArrayBuffer(PARAM_BYTES);
    fillParamsBuf(pb);
    paramsBuf = device.createBuffer({ size: PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuf, 0, pb);

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
      return module;
    }

    const updateMod = await compileShader('updateU', updateU_WGSL);
    const evolveBoundaryMod = await compileShader('evolveBoundary', evolveBoundaryWGSL);
    const jacobiSmoothMod = await compileShader('jacobiSmooth', jacobiSmoothWGSL);
    const computeRhoMod = await compileShader('computeRho', computeRhoWGSL);
    const computeResidualMod = await compileShader('computeResidual', computeResidualWGSL);
    const restrictMod = await compileShader('restrict', restrictWGSL);
    const coarseSmoothMod = await compileShader('coarseSmooth', coarseSmoothWGSL);
    const prolongCorrectMod = await compileShader('prolongCorrect', prolongCorrectWGSL);
    const reduceEnergyMod = await compileShader('reduceEnergy', reduceEnergyWGSL);
    const finalizeEnergyMod = await compileShader('finalizeEnergy', finalizeEnergyWGSL);
    const accumNormsMod = await compileShader('accumNorms', accumNormsWGSL);
    const decodeNormsMod = await compileShader('decodeNorms', decodeNormsWGSL);
    const normalizeMod = await compileShader('normalize', normalizeWGSL);
    const extractMod = await compileShader('extract', extractWGSL);
    const computeRhoSelfMod = await compileShader('computeRhoSelf', computeRhoSelfWGSL);
    const subtractPselfMod = await compileShader('subtractPself', subtractPselfWGSL);
    // Nuclear dynamics + GPU init shaders
    const gradPtotalMod = await compileShader('gradPtotal', gradPtotal_WGSL);
    const recomputeK_Mod = await compileShader('recomputeK', recomputeK_WGSL);
    const gpuInitAccumMod = await compileShader('gpuInitAccum', gpuInitAccumWGSL);
    const gpuInitFinalMod = await compileShader('gpuInitFinal', gpuInitFinalWGSL);
    // LOBPCG shaders
    const applyH_Mod = await compileShader('applyH', applyH_WGSL);
    const lobpcgResidualMod = await compileShader('lobpcgResidual', lobpcgResidualWGSL);
    const lobpcgInnerMod = await compileShader('lobpcgInner', lobpcgInnerWGSL);
    const lobpcgFinalizeMod = await compileShader('lobpcgFinalize', lobpcgFinalizeWGSL);
    const lobpcgUpdateMod = await compileShader('lobpcgUpdate', lobpcgUpdateWGSL);
    const chebyshevStepMod = await compileShader('chebyshevStep', chebyshevStepWGSL);

    updatePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: updateMod, entryPoint: 'main' } });
    evolveBoundaryPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: evolveBoundaryMod, entryPoint: 'main' } });
    // fixBoundaryU no longer needed (U stays continuous, normalization handles ∫U²=Z_eff)
    // const fixBoundaryUMod = await compileShader('fixBoundaryU', fixBoundaryU_WGSL);
    // fixBoundaryUPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: fixBoundaryUMod, entryPoint: 'main' } });
    computeRhoSelfPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: computeRhoSelfMod, entryPoint: 'main' } });
    subtractPselfPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: subtractPselfMod, entryPoint: 'main' } });
    if (USE_DIRECT_POTHER) {
      const computeRhoOtherMod = await compileShader('computeRhoOther', computeRhoOtherWGSL);
      const copyPotherForLabelMod = await compileShader('copyPotherForLabel', copyPotherForLabelWGSL);
      const initPdirectMod = await compileShader('initPdirect', initPdirectWGSL);
      computeRhoOtherPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: computeRhoOtherMod, entryPoint: 'main' } });
      copyPotherForLabelPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: copyPotherForLabelMod, entryPoint: 'main' } });
      initPdirectPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: initPdirectMod, entryPoint: 'main' } });
    }
    jacobiSmoothPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: jacobiSmoothMod, entryPoint: 'main' } });
    computeRhoPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: computeRhoMod, entryPoint: 'main' } });
    computeResidualPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: computeResidualMod, entryPoint: 'main' } });
    restrictPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: restrictMod, entryPoint: 'main' } });
    coarseSmoothPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: coarseSmoothMod, entryPoint: 'main' } });
    prolongCorrectPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: prolongCorrectMod, entryPoint: 'main' } });
    reduceEnergyPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: reduceEnergyMod, entryPoint: 'main' } });
    finalizeEnergyPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: finalizeEnergyMod, entryPoint: 'main' } });
    accumNormsPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: accumNormsMod, entryPoint: 'main' } });
    decodeNormsPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: decodeNormsMod, entryPoint: 'main' } });
    normalizePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: normalizeMod, entryPoint: 'main' } });
    extractPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: extractMod, entryPoint: 'main' } });
    gpuInitAccumPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: gpuInitAccumMod, entryPoint: 'main' } });
    gpuInitFinalPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: gpuInitFinalMod, entryPoint: 'main' } });
    // Nuclear dynamics pipelines
    gradPtotalPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: gradPtotalMod, entryPoint: 'main' } });
    recomputeK_PL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: recomputeK_Mod, entryPoint: 'main' } });
    // LOBPCG pipelines
    applyH_PL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: applyH_Mod, entryPoint: 'main' } });
    lobpcgResidualPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: lobpcgResidualMod, entryPoint: 'main' } });
    lobpcgInnerPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: lobpcgInnerMod, entryPoint: 'main' } });
    lobpcgFinalizePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: lobpcgFinalizeMod, entryPoint: 'main' } });
    lobpcgUpdatePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: lobpcgUpdateMod, entryPoint: 'main' } });
    chebyshevStepPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: chebyshevStepMod, entryPoint: 'main' } });

    for (let c = 0; c < 2; c++) {
      const n = 1 - c;
      // U update: label-based Neumann BC
      updateBG[c] = device.createBindGroup({ layout: updatePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: K_buf } },
        { binding: 2, resource: { buffer: U_buf[c] } },
        { binding: 3, resource: { buffer: labelBuf } },
        { binding: 4, resource: { buffer: PotherBuf } },
        { binding: 5, resource: { buffer: U_buf[n] } },
        { binding: 6, resource: { buffer: atomBuf } },
      ]});
      // Chebyshev step: same stencil as updateU + Uprev + chebParams
      chebyshevStepBG[c] = device.createBindGroup({ layout: chebyshevStepPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: K_buf } },
        { binding: 2, resource: { buffer: U_buf[c] } },
        { binding: 3, resource: { buffer: labelBuf } },
        { binding: 4, resource: { buffer: PotherBuf } },
        { binding: 5, resource: { buffer: U_buf[n] } },
        { binding: 6, resource: { buffer: atomBuf } },
        { binding: 7, resource: { buffer: U_prev_buf } },
        { binding: 8, resource: { buffer: chebParamsBuf } },
      ]});
      reduceEnergyBG[c] = device.createBindGroup({ layout: reduceEnergyPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: PotherBuf } },
        { binding: 3, resource: { buffer: K_buf } },
        { binding: 4, resource: { buffer: partialsBuf } },
        { binding: 5, resource: { buffer: labelBuf } },
        { binding: 6, resource: { buffer: atomBuf } },
      ]});
      accumNormsBG[c] = device.createBindGroup({ layout: accumNormsPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: labelBuf } },
        { binding: 3, resource: { buffer: normAtomicBuf } },
      ]});
      normalizeBG[c] = device.createBindGroup({ layout: normalizePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: normFloatBuf } },
        { binding: 3, resource: { buffer: labelBuf } },
        { binding: 4, resource: { buffer: atomBuf } },
      ]});
      extractBG[c] = device.createBindGroup({ layout: extractPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: labelBuf } },
        { binding: 3, resource: { buffer: K_buf } },
        { binding: 4, resource: { buffer: sliceBuf } },
        { binding: 5, resource: { buffer: atomBuf } },
      ]});
      // Multigrid bind groups (per cur for U dependency)
      computeRhoBG[c] = device.createBindGroup({ layout: computeRhoPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: rhoTotalBuf } },
      ]});
    }
    // Boundary evolution: labelBuf -> label2Buf, indexed by U cur
    for (let u = 0; u < 2; u++) {
      evolveBoundaryBG[u] = device.createBindGroup({ layout: evolveBoundaryPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: labelBuf } },
        { binding: 2, resource: { buffer: label2Buf } },
        { binding: 3, resource: { buffer: U_buf[u] } },
        { binding: 4, resource: { buffer: W_buf } },
      ]});
      // fixBoundaryU no longer needed — U stays continuous, normalization handles ∫U²=Z_eff
    }
    // Residual (cur-independent now — single field P)
    residualBG[0] = device.createBindGroup({ layout: computeResidualPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: P_buf[0] } },
      { binding: 2, resource: { buffer: residualBuf } },
      { binding: 3, resource: { buffer: rhoTotalBuf } },
    ]});
    // Jacobi fine-grid smoother: 2 directions (no U dependency)
    jacobiFineBG[0] = device.createBindGroup({ layout: jacobiSmoothPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: P_buf[0] } },
      { binding: 2, resource: { buffer: P_buf[1] } },
      { binding: 3, resource: { buffer: rhoTotalBuf } },
    ]});
    jacobiFineBG[1] = device.createBindGroup({ layout: jacobiSmoothPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: P_buf[1] } },
      { binding: 2, resource: { buffer: P_buf[0] } },
      { binding: 3, resource: { buffer: rhoTotalBuf } },
    ]});
    // Prolongation always corrects P_buf[0]
    prolongCorrectBG = device.createBindGroup({ layout: prolongCorrectPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: Pc_buf[0] } },
      { binding: 2, resource: { buffer: P_buf[0] } },
    ]});
    // Restriction and coarse solve (cur-independent)
    restrictBG = device.createBindGroup({ layout: restrictPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: residualBuf } },
      { binding: 2, resource: { buffer: coarseRhsBuf } },
    ]});
    for (let dir = 0; dir < 2; dir++) {
      coarseSmoothBG[dir] = device.createBindGroup({ layout: coarseSmoothPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: Pc_buf[dir] } },
        { binding: 2, resource: { buffer: Pc_buf[1 - dir] } },
        { binding: 3, resource: { buffer: coarseRhsBuf } },
      ]});
    }

    finalizeEnergyBG = device.createBindGroup({ layout: finalizeEnergyPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: partialsBuf } },
      { binding: 1, resource: { buffer: sumsBuf } },
      { binding: 2, resource: { buffer: numWGBuf } },
    ]});
    decodeNormsBG = device.createBindGroup({ layout: decodeNormsPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: normAtomicBuf } },
      { binding: 1, resource: { buffer: normFloatBuf } },
    ]});
    gpuInitAccumBG = device.createBindGroup({ layout: gpuInitAccumPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: atomBuf } },
      { binding: 2, resource: { buffer: K_buf } },
      { binding: 3, resource: { buffer: bestR2Buf } },
      { binding: 4, resource: { buffer: labelBuf } },
      { binding: 5, resource: { buffer: initRangeBuf } },
    ]});
    gpuInitFinalBG = device.createBindGroup({ layout: gpuInitFinalPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: bestR2Buf } },
      { binding: 2, resource: { buffer: U_buf[0] } },
    ]});

    // Now run GPU init (must be after gpuInitBG is created)
    await uploadInitialData();

    // Per-domain self-potential bind groups (only if SIC enabled)
    if (SIC_INTERVAL < 999999) {
      for (let m = 0; m < NELEC; m++) {
        computeRhoSelfBG[m] = [];
        for (let c = 0; c < 2; c++) {
          computeRhoSelfBG[m][c] = device.createBindGroup({ layout: computeRhoSelfPL.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: U_buf[c] } },
            { binding: 2, resource: { buffer: residualBuf } },
            { binding: 3, resource: { buffer: labelBuf } },
            { binding: 4, resource: { buffer: domainBufs[m] } },
          ]});
        }
        subtractPselfBG[m] = device.createBindGroup({ layout: subtractPselfPL.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: paramsBuf } },
          { binding: 1, resource: { buffer: sicBuf } },
          { binding: 2, resource: { buffer: PotherBuf } },
          { binding: 3, resource: { buffer: labelBuf } },
          { binding: 4, resource: { buffer: domainBufs[m] } },
        ]});
      }
    }
    // Jacobi for self-potential: sicBuf <-> PselfScratchBuf (NOT P_buf[1] — that's V-cycle scratch)
    jacobiSelfBG[0] = device.createBindGroup({ layout: jacobiSmoothPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: sicBuf } },
      { binding: 2, resource: { buffer: PselfScratchBuf } },
      { binding: 3, resource: { buffer: residualBuf } },
    ]});
    jacobiSelfBG[1] = device.createBindGroup({ layout: jacobiSmoothPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: PselfScratchBuf } },
      { binding: 2, resource: { buffer: sicBuf } },
      { binding: 3, resource: { buffer: residualBuf } },
    ]});
    // SIC V-cycle bind groups: residual, restrict, prolong for self-potential solve
    sicResidualBG = device.createBindGroup({ layout: computeResidualPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: sicBuf } },
      { binding: 2, resource: { buffer: sicResidualBuf } },
      { binding: 3, resource: { buffer: residualBuf } },  // rhoSelf stored here by computeRhoSelf
    ]});
    sicRestrictBG = device.createBindGroup({ layout: restrictPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: sicResidualBuf } },
      { binding: 2, resource: { buffer: coarseRhsBuf } },
    ]});
    sicProlongBG = device.createBindGroup({ layout: prolongCorrectPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: Pc_buf[0] } },
      { binding: 2, resource: { buffer: sicBuf } },
    ]});

    // Direct Pother bind groups (per-electron Poisson, no SIC)
    if (USE_DIRECT_POTHER) {
      for (let m = 0; m < NELEC; m++) {
        // computeRhoOther: same layout as computeRhoSelf (params, U, rhoOut, label, domIdx)
        computeRhoOtherBG[m] = [];
        for (let c = 0; c < 2; c++) {
          computeRhoOtherBG[m][c] = device.createBindGroup({ layout: computeRhoOtherPL.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: U_buf[c] } },
            { binding: 2, resource: { buffer: rhoTotalBuf } },  // reuse as rhoOther output
            { binding: 3, resource: { buffer: labelBuf } },
            { binding: 4, resource: { buffer: domainBufs[m] } },
          ]});
        }
        // Jacobi: P_direct[m] <-> P_directScratch, sourced from rhoTotalBuf
        jacobiDirectBG[m] = [
          device.createBindGroup({ layout: jacobiSmoothPL.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: P_directBuf[m] } },
            { binding: 2, resource: { buffer: P_directScratchBuf } },
            { binding: 3, resource: { buffer: rhoTotalBuf } },
          ]}),
          device.createBindGroup({ layout: jacobiSmoothPL.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: P_directScratchBuf } },
            { binding: 2, resource: { buffer: P_directBuf[m] } },
            { binding: 3, resource: { buffer: rhoTotalBuf } },
          ]}),
        ];
        // copyPotherForLabel: same layout as subtractPself (params, Psrc, Pother, label, domIdx)
        copyPotherForLabelBG[m] = device.createBindGroup({ layout: copyPotherForLabelPL.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: paramsBuf } },
          { binding: 1, resource: { buffer: P_directBuf[m] } },
          { binding: 2, resource: { buffer: PotherBuf } },
          { binding: 3, resource: { buffer: labelBuf } },
          { binding: 4, resource: { buffer: domainBufs[m] } },
        ]});
        // initPdirect: compute P_direct[m] = sum_{n!=m} 0.5/r on GPU
        initPdirectBG[m] = device.createBindGroup({ layout: initPdirectPL.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: paramsBuf } },
          { binding: 1, resource: { buffer: atomBuf } },
          { binding: 2, resource: { buffer: P_directBuf[m] } },
          { binding: 3, resource: { buffer: PotherBuf } },
          { binding: 4, resource: { buffer: domainBufs[m] } },
        ]});
      }
    }

    // Initialize P_direct buffers with 0.5/r on GPU (replaces CPU triple loop that blocked main thread)
    if (USE_DIRECT_POTHER && N_ELECTRONS > 1) {
      const clrEnc2 = device.createCommandEncoder();
      clrEnc2.clearBuffer(PotherBuf);
      device.queue.submit([clrEnc2.finish()]);
      for (let m = 0; m < NELEC; m++) {
        if (Z[m] === 0) continue;
        const initEnc = device.createCommandEncoder();
        const cp = initEnc.beginComputePass();
        cp.setPipeline(initPdirectPL);
        cp.setBindGroup(0, initPdirectBG[m]);
        dispatchLinear(cp, INTERIOR);
        cp.end();
        device.queue.submit([initEnc.finish()]);
      }
      await device.queue.onSubmittedWorkDone();
      console.log("Direct Pother: GPU-initialized P_direct + PotherBuf with 0.5/r, NELEC=" + NELEC);
    }

    // Nuclear dynamics bind groups
    // gradPtotal: direct Coulomb force from U² density on nuclei
    for (let c = 0; c < 2; c++) {
      // Bind to correct potential buffer:
      // NELEC ≤ 5: PotherBuf (direct per-electron Poisson)
      // NELEC > 5: P_buf[0] (total Poisson from multigrid)
      // HF uses U_buf; gradP uses P_buf (multigrid) or PotherBuf (direct)
      const forceBuf = USE_GRADP_FORCE ? (USE_DIRECT_POTHER ? PotherBuf : P_buf[0]) : U_buf[c];
      gradPtotalBG[c] = device.createBindGroup({ layout: gradPtotalPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: forceBuf } },
        { binding: 2, resource: { buffer: forceSumsBuf } },
        { binding: 3, resource: { buffer: atomBuf } },
      ]});
    }
    // recomputeK
    recomputeK_BG = device.createBindGroup({ layout: recomputeK_PL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: K_buf } },
      { binding: 2, resource: { buffer: atomBuf } },
    ]});

    // LOBPCG bind groups — applyH has same layout as updateU (K, U_in, label, P, HU_out, atoms)
    // We need 3 variants: applyH to X (U_buf[cur]), to W, and to P
    function makeApplyH_BG(inBuf, outBuf) {
      return device.createBindGroup({ layout: applyH_PL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: K_buf } },
        { binding: 2, resource: { buffer: inBuf } },
        { binding: 3, resource: { buffer: labelBuf } },
        { binding: 4, resource: { buffer: PotherBuf } },
        { binding: 5, resource: { buffer: outBuf } },
        { binding: 6, resource: { buffer: atomBuf } },
      ]});
    }
    // These will be recreated per cur flip — for now use U_buf[0]
    applyH_BG_X = [makeApplyH_BG(U_buf[0], HX_buf), makeApplyH_BG(U_buf[1], HX_buf)];
    applyH_BG_W = makeApplyH_BG(W_lobpcg_buf, HW_buf);
    applyH_BG_P = makeApplyH_BG(P_lobpcg_buf, HP_buf);

    // Ensure domainBufs exist for LOBPCG (need one per domain)
    for (let m = 0; m < NELEC; m++) {
      if (!domainBufs[m]) {
        domainBufs[m] = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const _zf = new Float32Array([Z[m]]);
        const _zu = new Uint32Array(_zf.buffer);
        device.queue.writeBuffer(domainBufs[m], 0, new Uint32Array([m, _zu[0], 0, 0]));
      }
    }

    // Inner product bind groups: [cur][domain] — cur=0 uses U_buf[0], cur=1 uses U_buf[1]
    {
      const bg0 = [], bg1 = [];
      for (let m = 0; m < NELEC; m++) {
        bg0[m] = device.createBindGroup({ layout: lobpcgInnerPL.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: paramsBuf } },
          { binding: 1, resource: { buffer: U_buf[0] } },
          { binding: 2, resource: { buffer: HX_buf } },
          { binding: 3, resource: { buffer: W_lobpcg_buf } },
          { binding: 4, resource: { buffer: HW_buf } },
          { binding: 5, resource: { buffer: P_lobpcg_buf } },
          { binding: 6, resource: { buffer: HP_buf } },
          { binding: 7, resource: { buffer: lobpcgPartialsBuf } },
          { binding: 8, resource: { buffer: labelBuf } },
          { binding: 9, resource: { buffer: domainBufs[m] } },
        ]});
        bg1[m] = device.createBindGroup({ layout: lobpcgInnerPL.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: paramsBuf } },
          { binding: 1, resource: { buffer: U_buf[1] } },
          { binding: 2, resource: { buffer: HX_buf } },
          { binding: 3, resource: { buffer: W_lobpcg_buf } },
          { binding: 4, resource: { buffer: HW_buf } },
          { binding: 5, resource: { buffer: P_lobpcg_buf } },
          { binding: 6, resource: { buffer: HP_buf } },
          { binding: 7, resource: { buffer: lobpcgPartialsBuf } },
          { binding: 8, resource: { buffer: labelBuf } },
          { binding: 9, resource: { buffer: domainBufs[m] } },
        ]});
      }
      lobpcgInnerBG = [bg0, bg1];
    }

    lobpcgFinalizeBG = device.createBindGroup({ layout: lobpcgFinalizePL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: lobpcgPartialsBuf } },
      { binding: 1, resource: { buffer: lobpcgSumsBuf } },
      { binding: 2, resource: { buffer: lobpcgNumWGBuf } },
    ]});

    // Residual bind groups (one per U_buf)
    lobpcgResidualBG = [];
    for (let c = 0; c < 2; c++) {
      lobpcgResidualBG[c] = device.createBindGroup({ layout: lobpcgResidualPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: HX_buf } },
        { binding: 3, resource: { buffer: W_lobpcg_buf } },
        { binding: 4, resource: { buffer: labelBuf } },
        { binding: 5, resource: { buffer: K_buf } },
        { binding: 6, resource: { buffer: PotherBuf } },
        { binding: 7, resource: { buffer: lambdaBuf } },
      ]});
    }

    // Update bind groups (one per U_buf since X is read_write)
    lobpcgUpdateBG = [];
    for (let c = 0; c < 2; c++) {
      lobpcgUpdateBG[c] = device.createBindGroup({ layout: lobpcgUpdatePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: W_lobpcg_buf } },
        { binding: 3, resource: { buffer: P_lobpcg_buf } },
        { binding: 4, resource: { buffer: labelBuf } },
        { binding: 5, resource: { buffer: coeffBuf } },
      ]});
    }

    console.log("Ready! dispatch(" + WG_UPDATE + ") " + NELEC + " domains + multigrid V-cycle + SIC + dynamics + LOBPCG");
    await startMolPhase();
    gpuReady = true;

  } catch (e) {
    gpuError = e.message || String(e);
    console.error("GPU init failed:", e);
  }
}

async function doSteps(n) {
  const t0 = performance.now();
  device.pushErrorScope('validation');
  device.pushErrorScope('out-of-memory');
  const enc = device.createCommandEncoder();
  let needForceReadback = false;

  for (let s = 0; s < n; s++) {
    const next = 1 - cur;
    // --- Poisson solve (skip for single-electron systems or NO_VEE) ---
    let vp;
    if (N_ELECTRONS > 1 && !window.NO_VEE && USE_DIRECT_POTHER) {
    // --- Direct per-electron Pother: solve ∇²P_m = -2π·ρ_other_m for each electron ---
    const JACOBI_DIRECT = 10;
    for (let m = 0; m < NELEC; m++) {
      if (Z[m] === 0) continue;
      // Compute rhoOther (density of all electrons except m) into rhoTotalBuf
      vp = enc.beginComputePass();
      vp.setPipeline(computeRhoOtherPL);
      vp.setBindGroup(0, computeRhoOtherBG[m][cur]);
      dispatchLinear(vp, INTERIOR);
      vp.end();
      // Jacobi iterations on P_direct[m] (persistent, accumulates across frames)
      for (let js = 0; js < JACOBI_DIRECT; js++) {
        vp = enc.beginComputePass();
        vp.setPipeline(jacobiSmoothPL);
        vp.setBindGroup(0, jacobiDirectBG[m][js % 2]);
        dispatchLinear(vp, INTERIOR);
        vp.end();
      }
      // Copy P_direct[m] to PotherBuf where label == m
      vp = enc.beginComputePass();
      vp.setPipeline(copyPotherForLabelPL);
      vp.setBindGroup(0, copyPotherForLabelBG[m]);
      dispatchLinear(vp, INTERIOR);
      vp.end();
    }

    } else if (N_ELECTRONS > 1 && !window.NO_VEE) {
    // --- Total Poisson + SIC path (for larger systems) ---
    // Compute rho_total from U[cur] + labels
    vp = enc.beginComputePass();
    vp.setPipeline(computeRhoPL);
    vp.setBindGroup(0, computeRhoBG[cur]);
    dispatchLinear(vp, INTERIOR);
    vp.end();
    // Jacobi smooth P every step
    const JACOBI_PER_STEP = window.USER_JACOBI_PER_STEP || 2;
    for (let js = 0; js < JACOBI_PER_STEP; js++) {
      vp = enc.beginComputePass();
      vp.setPipeline(jacobiSmoothPL);
      vp.setBindGroup(0, jacobiFineBG[js % 2]);
      dispatchLinear(vp, INTERIOR);
      vp.end();
    }

    // --- V-cycle coarse correction every POISSON_INTERVAL steps ---
    if (vcycleEnabled && s > 0 && s % POISSON_INTERVAL === 0) {
      vcycleCount++;
      // Residual from P[0]
      vp = enc.beginComputePass();
      vp.setPipeline(computeResidualPL);
      vp.setBindGroup(0, residualBG[0]);
      dispatchLinear(vp, INTERIOR);
      vp.end();
      // Restrict to coarse
      vp = enc.beginComputePass();
      vp.setPipeline(restrictPL);
      vp.setBindGroup(0, restrictBG);
      vp.dispatchWorkgroups(WG_COARSE);
      vp.end();
      // Zero coarse error + 10 coarse Jacobi sweeps
      enc.clearBuffer(Pc_buf[0]);
      for (let cs = 0; cs < 10; cs++) {
        vp = enc.beginComputePass();
        vp.setPipeline(coarseSmoothPL);
        vp.setBindGroup(0, coarseSmoothBG[cs % 2]);
        vp.dispatchWorkgroups(WG_COARSE);
        vp.end();
      }
      // Prolongate correction with damping
      vp = enc.beginComputePass();
      vp.setPipeline(prolongCorrectPL);
      vp.setBindGroup(0, prolongCorrectBG);
      dispatchLinear(vp, INTERIOR);
      vp.end();
    }

    } // end Poisson block

    // --- U update ---
    let cp;
    if (useCheb) {
      // Chebyshev semi-iterative acceleration of ITP
      // ψ_{n+1} = ω_n * (ITP step of ψ_n) + (1-ω_n) * ψ_{n-1}
      // Spectral radius of iteration matrix G = I - dt·H:
      //   eigenvalues of G: 1 - dt·λ_H, where λ_H ∈ [λ_min, λ_max]
      //   dt = dv*h², λ_max(H) ≈ 6/h², so dt·λ_max = 6*dv ≈ 0.72
      //   dt·λ_min ≈ dt*E ≈ small negative
      // Spectral radius ρ of G: max(|1-dt·λ_min|, |1-dt·λ_max|)
      // ρ_G ≈ 1 - dt·λ_min(H) for the "slow" eigenvalue (ground state)
      // Chebyshev ω recurrence accelerates convergence of the slow component.
      const dtLmax = 6.0 * dv;  // dt·λ_max ≈ 0.72
      const dtLmin = dtv * (isFinite(E) ? Math.max(-100, E) : -10);
      // Eigenvalues of G: gMax = 1 - dtLmin, gMin = 1 - dtLmax
      const gMax = 1 - dtLmin;  // close to 1 (slow convergence direction)
      const gMin = 1 - dtLmax;  // close to 0.28
      // Spectral radius ratio for Chebyshev: ρ² = ((gMax-gMin)/(gMax+gMin))²
      const rhoSq = ((gMax - gMin) / (gMax + gMin)) ** 2;

      // ω recurrence: ω_0 = 1, ω_1 = 1/(1-ρ²/2), ω_{n+1} = 1/(1-ρ²·ω_n/4)
      // Persists across frames — only resets when Chebyshev is toggled on
      let omega;
      if (chebOmega <= 1.0) {
        // First step after enable: no momentum
        omega = 1.0;
        chebOmega = 1.001;  // signal that step 0 is done
      } else if (chebOmega < 1.01) {
        // Second step: start recurrence
        omega = 1.0 / (1.0 - rhoSq / 2.0);
        chebOmega = omega;
      } else {
        // Steady state: continue recurrence (converges quickly to ω_∞)
        omega = 1.0 / (1.0 - rhoSq * chebOmega / 4.0);
        chebOmega = omega;
      }
      // Clamp for safety
      omega = Math.min(1.95, Math.max(1.0, omega));

      device.queue.writeBuffer(chebParamsBuf, 0, new Float32Array([omega, 0, 0, 0]));

      // Chebyshev step reads U[cur] and U_prev, writes U[next]
      // U_prev holds ψ_{n-1} from 2 steps ago (or garbage for step 0, but omega=1 so ignored)
      cp = enc.beginComputePass();
      cp.setPipeline(chebyshevStepPL);
      cp.setBindGroup(0, chebyshevStepBG[cur]);
      dispatchLinear(cp, INTERIOR);
      cp.end();

      // AFTER compute: save current ψ_n = U[cur] as ψ_{n-1} for next step
      enc.copyBufferToBuffer(U_buf[cur], 0, U_prev_buf, 0, S3 * 4);
    } else {
      // Original imaginary time propagation
      cp = enc.beginComputePass();
      cp.setPipeline(updatePL);
      cp.setBindGroup(0, updateBG[cur]);
      dispatchLinear(cp, INTERIOR);
      cp.end();
    }

    if ((s + 1) % NORM_INTERVAL === 0 || s === n - 1) {
      // Atomic norm accumulation: clear → accumulate → decode → normalize
      enc.clearBuffer(normAtomicBuf);
      cp = enc.beginComputePass();
      cp.setPipeline(accumNormsPL);
      cp.setBindGroup(0, accumNormsBG[next]);
      dispatchLinear(cp, INTERIOR);
      cp.end();

      cp = enc.beginComputePass();
      cp.setPipeline(decodeNormsPL);
      cp.setBindGroup(0, decodeNormsBG);
      cp.dispatchWorkgroups(Math.ceil(NELEC / 64));
      cp.end();

      cp = enc.beginComputePass();
      cp.setPipeline(normalizePL);
      cp.setBindGroup(0, normalizeBG[next]);
      dispatchLinear(cp, INTERIOR);
      cp.end();

    }

    cur = next;

    // Nuclear force computation at N_MOVE intervals — gradient of P directly
    if ((tStep + s + 1) % N_MOVE === 0) {
      cp = enc.beginComputePass();
      cp.setPipeline(gradPtotalPL);
      cp.setBindGroup(0, gradPtotalBG[cur]);
      cp.dispatchWorkgroups(NELEC);
      cp.end();
      needForceReadback = true;
    }

  }

  // --- Compute Pother = P_total - P_self (remove self-repulsion) ---
  if (N_ELECTRONS <= 1 || window.NO_VEE) {
    enc.clearBuffer(PotherBuf);  // no V_ee
  } else if (USE_DIRECT_POTHER) {
    // PotherBuf already built per-step in the direct Poisson loop above
  } else {
  enc.copyBufferToBuffer(P_buf[0], 0, PotherBuf, 0, S3 * 4);
  // Only run SIC periodically to avoid GPU timeout with many atoms
  if (frameCount > 0 && frameCount % SIC_INTERVAL === 0) {
    for (let m = 0; m < NELEC; m++) {
      if (Z[m] === 0) continue;
      let sp = enc.beginComputePass();
      sp.setPipeline(computeRhoSelfPL);
      sp.setBindGroup(0, computeRhoSelfBG[m][cur]);
      dispatchLinear(sp, INTERIOR);
      sp.end();
      enc.clearBuffer(sicBuf);
      // SIC V-cycle: pre-smooth, restrict, coarse solve, prolongate, post-smooth
      for (let js = 0; js < 4; js++) {
        sp = enc.beginComputePass();
        sp.setPipeline(jacobiSmoothPL);
        sp.setBindGroup(0, jacobiSelfBG[js % 2]);
        dispatchLinear(sp, INTERIOR);
        sp.end();
      }
      // Compute residual of SIC Poisson
      sp = enc.beginComputePass();
      sp.setPipeline(computeResidualPL);
      sp.setBindGroup(0, sicResidualBG);
      dispatchLinear(sp, INTERIOR);
      sp.end();
      // Restrict residual to coarse grid
      sp = enc.beginComputePass();
      sp.setPipeline(restrictPL);
      sp.setBindGroup(0, sicRestrictBG);
      sp.dispatchWorkgroups(WG_COARSE);
      sp.end();
      // Coarse solve (10 Jacobi sweeps)
      enc.clearBuffer(Pc_buf[0]);
      for (let cs = 0; cs < 10; cs++) {
        sp = enc.beginComputePass();
        sp.setPipeline(coarseSmoothPL);
        sp.setBindGroup(0, coarseSmoothBG[cs % 2]);
        sp.dispatchWorkgroups(WG_COARSE);
        sp.end();
      }
      // Prolongate correction back to sicBuf
      sp = enc.beginComputePass();
      sp.setPipeline(prolongCorrectPL);
      sp.setBindGroup(0, sicProlongBG);
      dispatchLinear(sp, INTERIOR);
      sp.end();
      // Post-smooth
      for (let js = 0; js < 4; js++) {
        sp = enc.beginComputePass();
        sp.setPipeline(jacobiSmoothPL);
        sp.setBindGroup(0, jacobiSelfBG[js % 2]);
        dispatchLinear(sp, INTERIOR);
        sp.end();
      }
      sp = enc.beginComputePass();
      sp.setPipeline(subtractPselfPL);
      sp.setBindGroup(0, subtractPselfBG[m]);
      dispatchLinear(sp, INTERIOR);
      sp.end();
    }
  }
  } // end Pother block

  // --- Energy reduce (after SIC so V_ee uses current frame's PotherBuf) ---
  {
    let cp = enc.beginComputePass();
    cp.setPipeline(reduceEnergyPL);
    cp.setBindGroup(0, reduceEnergyBG[cur]);
    cp.dispatchWorkgroups(WG_REDUCE);
    cp.end();

    cp = enc.beginComputePass();
    cp.setPipeline(finalizeEnergyPL);
    cp.setBindGroup(0, finalizeEnergyBG);
    cp.dispatchWorkgroups(1);
    cp.end();
  }

  // --- Evolve level set W + flip labels where W < 0 ---
  frameCount++;
  if (window.FREEZE_BOUNDARY) { /* skip boundary evolution */ }
  else for (let s = 0; s < W_STEPS_PER_FRAME; s++) {
    let bp = enc.beginComputePass();
    bp.setPipeline(evolveBoundaryPL);
    bp.setBindGroup(0, evolveBoundaryBG[cur]);
    dispatchLinear(bp, INTERIOR);
    bp.end();
    // U stays continuous at domain flips — normalization handles ∫U²=Z_eff
    enc.copyBufferToBuffer(label2Buf, 0, labelBuf, 0, S3 * 4);
  }

  const ep = enc.beginComputePass();
  ep.setPipeline(extractPL);
  ep.setBindGroup(0, extractBG[cur]);
  ep.dispatchWorkgroups(WG_EXTRACT, WG_EXTRACT);
  ep.end();

  enc.copyBufferToBuffer(sumsBuf, 0, sumsReadBuf, 0, SUMS_BYTES);
  enc.copyBufferToBuffer(sliceBuf, 0, sliceReadBuf, 0, SLICE_SIZE);
  if (needForceReadback) {
    enc.copyBufferToBuffer(forceSumsBuf, 0, forceSumsReadBuf, 0, NELEC * 3 * 4);
  }
  device.queue.submit([enc.finish()]);

  const oomErr = await device.popErrorScope();
  const valErr = await device.popErrorScope();
  if (oomErr) { console.error("GPU OOM:", oomErr.message); window._gpuValErr = "OOM: " + oomErr.message; }
  if (valErr) { console.error("GPU Validation:", valErr.message); window._gpuValErr = "VAL: " + valErr.message; }

  await sumsReadBuf.mapAsync(GPUMapMode.READ);
  const sumsData = new Float32Array(sumsReadBuf.getMappedRange().slice(0));
  sumsReadBuf.unmap();
  E_T = sumsData[0];
  E_eK = sumsData[1];
  E_ee = sumsData[2];  // SIC already removed self-interaction from PotherBuf
  // Per-H correction: exclusion sphere around each nucleus clips T and V_eK
  // This is a per-nucleus artifact independent of bonding
  {
    const H_RAW = { 100: [0.5, -1.0], 200: [0.503, -1.028], 300: [0.5, -1.0] };
    const raw = H_RAW[NN] || H_RAW[200];
    let nH = 0;
    for (let a = 0; a < NELEC; a++) { if (Z[a] === 1 && r_cut[a] === 0) nH++; }
    E_T  += nH * (0.5 - raw[0]);
    E_eK += nH * (-1.0 - raw[1]);
  }

  // Dipole moment: μ = Σ Z_a·R_a - ∫ ρ(r)·r dV
  // sumsData[3..5] = electronic part ∫ ρ·r dV (positive, negate for electron charge)
  {
    let dip_x = 0, dip_y = 0, dip_z = 0;
    // Nuclear contribution: Σ Z_a · R_a  (in grid coords * h = au)
    for (let a = 0; a < NELEC; a++) {
      if (Z[a] === 0) continue;
      dip_x += Z[a] * nucPos[a][0] * hGrid;
      dip_y += Z[a] * nucPos[a][1] * hGrid;
      dip_z += Z[a] * nucPos[a][2] * hGrid;
    }
    // Electronic contribution (negative charge)
    dip_x -= sumsData[3];
    dip_y -= sumsData[4];
    dip_z -= sumsData[5];
    dipole_au = Math.sqrt(dip_x * dip_x + dip_y * dip_y + dip_z * dip_z);
    dipole_D = dipole_au * 2.5417;  // au to Debye
  }

  await sliceReadBuf.mapAsync(GPUMapMode.READ);
  sliceData = new Float32Array(sliceReadBuf.getMappedRange().slice(0));
  sliceReadBuf.unmap();

  tStep += n;
  lastMs = performance.now() - t0;

  E_KK = 0;
  if (addNucRepulsion) {
    const soft_nuc = 0.04 * h2v;
    for (let a = 0; a < uniqueNuclei.length; a++) {
      const ea = uniqueNuclei[a].elecIndices[0]; // representative electron index for position
      for (let b = a + 1; b < uniqueNuclei.length; b++) {
        const eb = uniqueNuclei[b].elecIndices[0];
        const d = Math.sqrt(
          ((nucPos[ea][0]-nucPos[eb][0])*hGrid)**2 +
          ((nucPos[ea][1]-nucPos[eb][1])*hGrid)**2 +
          ((nucPos[ea][2]-nucPos[eb][2])*hGrid)**2 + soft_nuc);
        E_KK += uniqueNuclei[a].Z_eff * uniqueNuclei[b].Z_eff / d;
      }
    }
  }
  E = E_T + E_eK + E_ee + E_KK;
  E_bind = E - E_atoms_sum;

  if (!isFinite(E)) {
    gpuError = "Numerical instability at step " + tStep;
    return;
  }

  console.log("Step " + tStep + ": E=" + E.toFixed(6) + " T=" + E_T.toFixed(4) + " V_eK=" + E_eK.toFixed(4) + " V_ee=" + E_ee.toFixed(4) + " V_KK=" + E_KK.toFixed(4) + " (" + lastMs.toFixed(0) + "ms/" + n + "steps)");

  // Force readback and nuclear dynamics
  if (needForceReadback) {
    await forceSumsReadBuf.mapAsync(GPUMapMode.READ);
    const forceData = new Float32Array(forceSumsReadBuf.getMappedRange().slice(0));
    forceSumsReadBuf.unmap();
    // Store electronic forces (HF integral from GPU)
    for (let a = 0; a < NELEC; a++) {
      if (Z[a] === 0 && Z_nuc[a] === 0) { nucForceElec[a] = [0,0,0]; nucForceNuc[a] = [0,0,0]; nucForceTotal[a] = [0,0,0]; continue; }
      nucForceElec[a] = (Z[a] > 0 || Z_nuc[a] > 0) ? [forceData[a*3], forceData[a*3+1], forceData[a*3+2]] : [0,0,0];
      nucForceNuc[a] = [0, 0, 0];
    }


    // Compute nuclear-nuclear Coulomb repulsion forces
    if (!addNucRepulsion) {
      // Skip nuc-nuc forces (e.g. He internal entries)
      for (let a = 0; a < NELEC; a++) nucForceNuc[a] = [0, 0, 0];
    }
    for (let a = 0; addNucRepulsion && a < uniqueNuclei.length; a++) {
      const ea = uniqueNuclei[a].elecIndices[0];
      let fx = 0, fy = 0, fz = 0;
      for (let b = 0; b < uniqueNuclei.length; b++) {
        if (b === a) continue;
        const eb = uniqueNuclei[b].elecIndices[0];
        const dx = (nucPos[ea][0] - nucPos[eb][0]) * hGrid;
        const dy = (nucPos[ea][1] - nucPos[eb][1]) * hGrid;
        const dz = (nucPos[ea][2] - nucPos[eb][2]) * hGrid;
        const r2 = dx*dx + dy*dy + dz*dz + hGrid*hGrid;
        const r = Math.sqrt(r2);
        const inv_r3 = 1.0 / (r * r2);
        fx += uniqueNuclei[a].Z_eff * uniqueNuclei[b].Z_eff * dx * inv_r3;
        fy += uniqueNuclei[a].Z_eff * uniqueNuclei[b].Z_eff * dy * inv_r3;
        fz += uniqueNuclei[a].Z_eff * uniqueNuclei[b].Z_eff * dz * inv_r3;
      }
      for (const e of uniqueNuclei[a].elecIndices) {
        nucForceNuc[e] = [fx, fy, fz];
      }
    }
    // Compute total forces
    for (let a = 0; a < NELEC; a++) {
      if (Z[a] === 0 && Z_nuc[a] === 0) continue;
      nucForceTotal[a] = [
        nucForceElec[a][0] + nucForceNuc[a][0],
        nucForceElec[a][1] + nucForceNuc[a][1],
        nucForceElec[a][2] + nucForceNuc[a][2]
      ];
    }
    // Zero display forces on hinge atoms
    if (window.USER_HINGE_ATOMS) {
      for (const a of window.USER_HINGE_ATOMS) {
        if (a < NELEC) nucForceTotal[a] = [0, 0, 0];
      }
    }
    if (dynamicsEnabled) await moveNuclei(forceData);
  }
}

// Solve 2×2 or 3×3 generalized eigenproblem on CPU: H_sub * c = λ * S_sub * c
// Returns {lambda, coeffs: [c0, c1, c2]} for lowest eigenvalue
function solveRitz2x2(ip) {
  // ip: {xx, xhx, ww, whw, xw, xhw}
  // S = [[xx, xw], [xw, ww]], H = [[xhx, xhw], [xhw, whw]]
  // Generalized eigenproblem: (H - λS)c = 0
  // det(H - λS) = 0 → quadratic in λ
  const a11 = ip.xhx, a12 = ip.xhw, a22 = ip.whw;
  const s11 = ip.xx, s12 = ip.xw, s22 = ip.ww;
  // (a11 - λ*s11)*(a22 - λ*s22) - (a12 - λ*s12)² = 0
  const A = s11*s22 - s12*s12;
  const B = -(a11*s22 + a22*s11 - 2*a12*s12);
  const C = a11*a22 - a12*a12;
  const disc = B*B - 4*A*C;
  const sqrtDisc = Math.sqrt(Math.max(0, disc));
  const lam1 = (-B - sqrtDisc) / (2*A);
  const lam2 = (-B + sqrtDisc) / (2*A);
  const lam = Math.min(lam1, lam2);
  // Eigenvector: (H - λS)c = 0 → c1*(a11-λ*s11) + c2*(a12-λ*s12) = 0
  const r1 = a11 - lam*s11, r2 = a12 - lam*s12;
  let c0, c1;
  if (Math.abs(r1) > Math.abs(r2)) {
    c1 = 1; c0 = -r2/r1;
  } else if (Math.abs(r2) > 1e-15) {
    c0 = 1; c1 = -r1/r2;
  } else {
    c0 = 1; c1 = 0;
  }
  // Normalize so that c'*S*c = 1
  const norm2 = c0*c0*s11 + 2*c0*c1*s12 + c1*c1*s22;
  const inv = 1/Math.sqrt(Math.max(1e-30, norm2));
  return { lambda: lam, coeffs: [c0*inv, c1*inv, 0] };
}

function solveRitz3x3(ip) {
  // ip has: xx, xhx, ww, whw, xw, xhw, pp, php, xp, xhp, wp, whp
  // Matrices S and H (symmetric 3×3)
  const S = [
    [ip.xx, ip.xw, ip.xp],
    [ip.xw, ip.ww, ip.wp],
    [ip.xp, ip.wp, ip.pp]
  ];
  const H = [
    [ip.xhx, ip.xhw, ip.xhp],
    [ip.xhw, ip.whw, ip.whp],
    [ip.xhp, ip.whp, ip.php]
  ];
  // Cholesky of S: S = L*L^T, then solve L^{-1}*H*L^{-T} * y = λ*y
  const L = [[0,0,0],[0,0,0],[0,0,0]];
  L[0][0] = Math.sqrt(Math.max(1e-30, S[0][0]));
  L[1][0] = S[1][0] / L[0][0];
  L[2][0] = S[2][0] / L[0][0];
  L[1][1] = Math.sqrt(Math.max(1e-30, S[1][1] - L[1][0]*L[1][0]));
  L[2][1] = (S[2][1] - L[2][0]*L[1][0]) / L[1][1];
  L[2][2] = Math.sqrt(Math.max(1e-30, S[2][2] - L[2][0]*L[2][0] - L[2][1]*L[2][1]));

  // If Cholesky fails (S not positive definite), fall back to 2×2
  if (!isFinite(L[2][2]) || L[0][0] < 1e-15 || L[1][1] < 1e-15 || L[2][2] < 1e-15) {
    return solveRitz2x2(ip);
  }

  // L^{-1}
  const Li = [[1/L[0][0],0,0],[0,1/L[1][1],0],[0,0,1/L[2][2]]];
  Li[1][0] = -L[1][0]*Li[0][0]*Li[1][1];
  Li[2][0] = -(L[2][0]*Li[0][0] + L[2][1]*Li[1][0])*Li[2][2];
  Li[2][1] = -L[2][1]*Li[1][1]*Li[2][2];

  // A = Li * H * Li^T (standard eigenproblem)
  // First B = H * Li^T
  const B = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let s = 0;
    for (let k = 0; k < 3; k++) s += H[i][k] * Li[j][k];  // Li^T[k][j] = Li[j][k]
    B[i][j] = s;
  }
  const A = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let s = 0;
    for (let k = 0; k < 3; k++) s += Li[i][k] * B[k][j];
    A[i][j] = s;
  }

  // 3×3 symmetric eigenvalue via Jacobi rotations (small matrix, few iterations)
  const V = [[1,0,0],[0,1,0],[0,0,1]];
  const D = [A[0][0], A[1][1], A[2][2]];
  const offD = [A[0][1], A[0][2], A[1][2]]; // (0,1), (0,2), (1,2)
  for (let sweep = 0; sweep < 20; sweep++) {
    const pairs = [[0,1],[0,2],[1,2]];
    for (const [p, q] of pairs) {
      const apq = A[p][q];
      if (Math.abs(apq) < 1e-15) continue;
      const tau = (A[q][q] - A[p][p]) / (2*apq);
      const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau*tau));
      const c = 1/Math.sqrt(1 + t*t);
      const s = t*c;
      // Rotate A
      const app = A[p][p], aqq = A[q][q];
      A[p][p] = c*c*app - 2*s*c*apq + s*s*aqq;
      A[q][q] = s*s*app + 2*s*c*apq + c*c*aqq;
      A[p][q] = 0; A[q][p] = 0;
      for (let r = 0; r < 3; r++) {
        if (r === p || r === q) continue;
        const arp = A[r][p], arq = A[r][q];
        A[r][p] = c*arp - s*arq; A[p][r] = A[r][p];
        A[r][q] = s*arp + c*arq; A[q][r] = A[r][q];
      }
      // Rotate V
      for (let r = 0; r < 3; r++) {
        const vrp = V[r][p], vrq = V[r][q];
        V[r][p] = c*vrp - s*vrq;
        V[r][q] = s*vrp + c*vrq;
      }
    }
  }

  // Find smallest eigenvalue
  let minIdx = 0;
  if (A[1][1] < A[minIdx][minIdx]) minIdx = 1;
  if (A[2][2] < A[minIdx][minIdx]) minIdx = 2;
  const lam = A[minIdx][minIdx];
  const y = [V[0][minIdx], V[1][minIdx], V[2][minIdx]];
  // Transform back: c = Li^T * y
  const c = [
    Li[0][0]*y[0] + Li[1][0]*y[1] + Li[2][0]*y[2],
    Li[1][1]*y[1] + Li[2][1]*y[2],
    Li[2][2]*y[2]
  ];
  return { lambda: lam, coeffs: c };
}

async function doLOBPCGStep() {
  const t0 = performance.now();
  device.pushErrorScope('validation');
  device.pushErrorScope('out-of-memory');

  // Phase 1: Compute rho + Poisson + SIC (same as doSteps but single iteration)
  {
    const enc = device.createCommandEncoder();
    let vp;

    if (N_ELECTRONS > 1 && USE_DIRECT_POTHER) {
      // Direct per-electron Pother
      for (let m = 0; m < NELEC; m++) {
        if (Z[m] === 0) continue;
        vp = enc.beginComputePass(); vp.setPipeline(computeRhoOtherPL); vp.setBindGroup(0, computeRhoOtherBG[m][cur]); dispatchLinear(vp, INTERIOR); vp.end();
        for (let js = 0; js < 10; js++) { vp = enc.beginComputePass(); vp.setPipeline(jacobiSmoothPL); vp.setBindGroup(0, jacobiDirectBG[m][js % 2]); dispatchLinear(vp, INTERIOR); vp.end(); }
        vp = enc.beginComputePass(); vp.setPipeline(copyPotherForLabelPL); vp.setBindGroup(0, copyPotherForLabelBG[m]); dispatchLinear(vp, INTERIOR); vp.end();
      }
    } else if (N_ELECTRONS > 1) {
      // Compute rho
      vp = enc.beginComputePass(); vp.setPipeline(computeRhoPL); vp.setBindGroup(0, computeRhoBG[cur]); dispatchLinear(vp, INTERIOR); vp.end();
      // Poisson solve
      for (let js = 0; js < 4; js++) { vp = enc.beginComputePass(); vp.setPipeline(jacobiSmoothPL); vp.setBindGroup(0, jacobiFineBG[js % 2]); dispatchLinear(vp, INTERIOR); vp.end(); }
      if (vcycleEnabled) {
        vp = enc.beginComputePass(); vp.setPipeline(computeResidualPL); vp.setBindGroup(0, residualBG[0]); dispatchLinear(vp, INTERIOR); vp.end();
        vp = enc.beginComputePass(); vp.setPipeline(restrictPL); vp.setBindGroup(0, restrictBG); vp.dispatchWorkgroups(WG_COARSE); vp.end();
        enc.clearBuffer(Pc_buf[0]);
        for (let cs = 0; cs < 10; cs++) { vp = enc.beginComputePass(); vp.setPipeline(coarseSmoothPL); vp.setBindGroup(0, coarseSmoothBG[cs % 2]); vp.dispatchWorkgroups(WG_COARSE); vp.end(); }
        vp = enc.beginComputePass(); vp.setPipeline(prolongCorrectPL); vp.setBindGroup(0, prolongCorrectBG); dispatchLinear(vp, INTERIOR); vp.end();
      }
    }

    // SIC (only for non-direct path)
    if (N_ELECTRONS <= 1 || window.NO_VEE) {
      enc.clearBuffer(PotherBuf);
    } else if (USE_DIRECT_POTHER) {
      // already built above
    } else {
      enc.copyBufferToBuffer(P_buf[0], 0, PotherBuf, 0, S3 * 4);
      for (let m = 0; m < NELEC; m++) {
        if (Z[m] === 0) continue;
        let sp = enc.beginComputePass(); sp.setPipeline(computeRhoSelfPL); sp.setBindGroup(0, computeRhoSelfBG[m][cur]); dispatchLinear(sp, INTERIOR); sp.end();
        enc.clearBuffer(sicBuf);
        for (let js = 0; js < 4; js++) { sp = enc.beginComputePass(); sp.setPipeline(jacobiSmoothPL); sp.setBindGroup(0, jacobiSelfBG[js % 2]); dispatchLinear(sp, INTERIOR); sp.end(); }
        sp = enc.beginComputePass(); sp.setPipeline(computeResidualPL); sp.setBindGroup(0, sicResidualBG); dispatchLinear(sp, INTERIOR); sp.end();
        sp = enc.beginComputePass(); sp.setPipeline(restrictPL); sp.setBindGroup(0, sicRestrictBG); sp.dispatchWorkgroups(WG_COARSE); sp.end();
        enc.clearBuffer(Pc_buf[0]);
        for (let cs = 0; cs < 10; cs++) { sp = enc.beginComputePass(); sp.setPipeline(coarseSmoothPL); sp.setBindGroup(0, coarseSmoothBG[cs % 2]); sp.dispatchWorkgroups(WG_COARSE); sp.end(); }
        sp = enc.beginComputePass(); sp.setPipeline(prolongCorrectPL); sp.setBindGroup(0, sicProlongBG); dispatchLinear(sp, INTERIOR); sp.end();
        for (let js = 0; js < 4; js++) { sp = enc.beginComputePass(); sp.setPipeline(jacobiSmoothPL); sp.setBindGroup(0, jacobiSelfBG[js % 2]); dispatchLinear(sp, INTERIOR); sp.end(); }
        sp = enc.beginComputePass(); sp.setPipeline(subtractPselfPL); sp.setBindGroup(0, subtractPselfBG[m]); dispatchLinear(sp, INTERIOR); sp.end();
      }
    }

    device.queue.submit([enc.finish()]);
  }

  // Phase 2: LOBPCG iterations per domain
  // Clear P buffer on first call (no conjugate direction yet)
  if (frameCount === 0) {
    const enc = device.createCommandEncoder();
    enc.clearBuffer(P_lobpcg_buf);
    enc.clearBuffer(HP_buf);
    device.queue.submit([enc.finish()]);
  }

  for (let m = 0; m < NELEC; m++) {
    if (Z[m] === 0) continue;

    for (let iter = 0; iter < LOBPCG_ITERS; iter++) {
      const hasP = frameCount > 0 || iter > 0;

      // Step 1: Apply H to X → HX
      {
        const enc = device.createCommandEncoder();
        let cp = enc.beginComputePass();
        cp.setPipeline(applyH_PL);
        cp.setBindGroup(0, applyH_BG_X[cur]);
        dispatchLinear(cp, INTERIOR);
        cp.end();

        // Step 2: Compute inner products (just <X|X> and <X|HX> for lambda)
        // We compute all 12 at once for the Ritz problem
        // But first need W and HW — compute after we know lambda
        // For first pass: compute <X|X>, <X|HX> via inner product shader
        // (W and P contributions will be zero on first pass)
        if (iter === 0 && !hasP) {
          // Clear W and P so inner products are clean
          enc.clearBuffer(W_lobpcg_buf);
          enc.clearBuffer(HW_buf);
        }

        cp = enc.beginComputePass();
        cp.setPipeline(lobpcgInnerPL);
        cp.setBindGroup(0, lobpcgInnerBG[cur][m]);
        cp.dispatchWorkgroups(MAX_LOBPCG_WG);
        cp.end();

        cp = enc.beginComputePass();
        cp.setPipeline(lobpcgFinalizePL);
        cp.setBindGroup(0, lobpcgFinalizeBG);
        cp.dispatchWorkgroups(1);
        cp.end();

        enc.copyBufferToBuffer(lobpcgSumsBuf, 0, lobpcgSumsReadBuf, 0, NRED_LOBPCG * 4);
        device.queue.submit([enc.finish()]);
      }

      // Read back inner products
      await lobpcgSumsReadBuf.mapAsync(GPUMapMode.READ);
      const ipData = new Float32Array(lobpcgSumsReadBuf.getMappedRange().slice(0));
      lobpcgSumsReadBuf.unmap();

      const ip = {
        xx: ipData[0], xhx: ipData[1], ww: ipData[2], whw: ipData[3],
        xw: ipData[4], xhw: ipData[5], pp: ipData[6], php: ipData[7],
        xp: ipData[8], xhp: ipData[9], wp: ipData[10], whp: ipData[11]
      };

      const lambda = ip.xx > 0 ? ip.xhx / ip.xx : 0;

      // Step 3: Compute residual W = (HX - λX) / diag(H) (preconditioned)
      {
        const enc = device.createCommandEncoder();
        device.queue.writeBuffer(lambdaBuf, 0, new Float32Array([lambda]));
        device.queue.writeBuffer(lambdaBuf, 4, new Uint32Array([m]));

        let cp = enc.beginComputePass();
        cp.setPipeline(lobpcgResidualPL);
        cp.setBindGroup(0, lobpcgResidualBG[cur]);
        dispatchLinear(cp, INTERIOR);
        cp.end();

        // Step 4: Apply H to W → HW
        cp = enc.beginComputePass();
        cp.setPipeline(applyH_PL);
        cp.setBindGroup(0, applyH_BG_W);
        dispatchLinear(cp, INTERIOR);
        cp.end();

        // Step 5: Apply H to P → HP (if we have P)
        if (hasP) {
          cp = enc.beginComputePass();
          cp.setPipeline(applyH_PL);
          cp.setBindGroup(0, applyH_BG_P);
          dispatchLinear(cp, INTERIOR);
          cp.end();
        }

        // Step 6: Recompute all inner products with updated W (and P)
        cp = enc.beginComputePass();
        cp.setPipeline(lobpcgInnerPL);
        cp.setBindGroup(0, lobpcgInnerBG[cur][m]);
        cp.dispatchWorkgroups(MAX_LOBPCG_WG);
        cp.end();

        cp = enc.beginComputePass();
        cp.setPipeline(lobpcgFinalizePL);
        cp.setBindGroup(0, lobpcgFinalizeBG);
        cp.dispatchWorkgroups(1);
        cp.end();

        enc.copyBufferToBuffer(lobpcgSumsBuf, 0, lobpcgSumsReadBuf, 0, NRED_LOBPCG * 4);
        device.queue.submit([enc.finish()]);
      }

      // Read updated inner products
      await lobpcgSumsReadBuf.mapAsync(GPUMapMode.READ);
      const ipData2 = new Float32Array(lobpcgSumsReadBuf.getMappedRange().slice(0));
      lobpcgSumsReadBuf.unmap();

      const ip2 = {
        xx: ipData2[0], xhx: ipData2[1], ww: ipData2[2], whw: ipData2[3],
        xw: ipData2[4], xhw: ipData2[5], pp: ipData2[6], php: ipData2[7],
        xp: ipData2[8], xhp: ipData2[9], wp: ipData2[10], whp: ipData2[11]
      };

      // Step 7: Solve Ritz problem
      let ritz;
      if (hasP && ip2.pp > 1e-20) {
        ritz = solveRitz3x3(ip2);
      } else {
        ritz = solveRitz2x2(ip2);
      }

      // Step 8: Update X and P
      {
        const enc = device.createCommandEncoder();
        device.queue.writeBuffer(coeffBuf, 0, new Float32Array([ritz.coeffs[0], ritz.coeffs[1], ritz.coeffs[2]]));
        device.queue.writeBuffer(coeffBuf, 12, new Uint32Array([m]));

        let cp = enc.beginComputePass();
        cp.setPipeline(lobpcgUpdatePL);
        cp.setBindGroup(0, lobpcgUpdateBG[cur]);
        dispatchLinear(cp, INTERIOR);
        cp.end();

        device.queue.submit([enc.finish()]);
      }
    }
  }

  // Phase 3: Normalize + energy reduce + boundary evolution + extract (same as doSteps)
  {
    const enc = device.createCommandEncoder();

    // Normalize all domains
    enc.clearBuffer(normAtomicBuf);
    let cp = enc.beginComputePass();
    cp.setPipeline(accumNormsPL);
    cp.setBindGroup(0, accumNormsBG[cur]);
    dispatchLinear(cp, INTERIOR);
    cp.end();

    cp = enc.beginComputePass();
    cp.setPipeline(decodeNormsPL);
    cp.setBindGroup(0, decodeNormsBG);
    cp.dispatchWorkgroups(Math.ceil(NELEC / 64));
    cp.end();

    cp = enc.beginComputePass();
    cp.setPipeline(normalizePL);
    cp.setBindGroup(0, normalizeBG[cur]);
    dispatchLinear(cp, INTERIOR);
    cp.end();

    // Handle analytical inside for bare atoms (same as updateU)
    // The updateU shader handles this — run one update step to fix analytical region
    const next = 1 - cur;
    cp = enc.beginComputePass();
    cp.setPipeline(updatePL);
    cp.setBindGroup(0, updateBG[cur]);
    dispatchLinear(cp, INTERIOR);
    cp.end();

    // Energy reduce
    cp = enc.beginComputePass();
    cp.setPipeline(reduceEnergyPL);
    cp.setBindGroup(0, reduceEnergyBG[next]);
    cp.dispatchWorkgroups(WG_REDUCE);
    cp.end();

    cp = enc.beginComputePass();
    cp.setPipeline(finalizeEnergyPL);
    cp.setBindGroup(0, finalizeEnergyBG);
    cp.dispatchWorkgroups(1);
    cp.end();

    // Boundary evolution
    frameCount++;
    if (!window.FREEZE_BOUNDARY) {
    for (let s = 0; s < W_STEPS_PER_FRAME; s++) {
      let bp = enc.beginComputePass();
      bp.setPipeline(evolveBoundaryPL);
      bp.setBindGroup(0, evolveBoundaryBG[next]);
      dispatchLinear(bp, INTERIOR);
      bp.end();
      enc.copyBufferToBuffer(label2Buf, 0, labelBuf, 0, S3 * 4);
    }
    }

    // Extract slice
    const ep = enc.beginComputePass();
    ep.setPipeline(extractPL);
    ep.setBindGroup(0, extractBG[next]);
    ep.dispatchWorkgroups(WG_EXTRACT, WG_EXTRACT);
    ep.end();

    enc.copyBufferToBuffer(sumsBuf, 0, sumsReadBuf, 0, SUMS_BYTES);
    enc.copyBufferToBuffer(sliceBuf, 0, sliceReadBuf, 0, SLICE_SIZE);
    device.queue.submit([enc.finish()]);

    cur = next;
  }

  const oomErr = await device.popErrorScope();
  const valErr = await device.popErrorScope();
  if (oomErr) { console.error("GPU OOM:", oomErr.message); }
  if (valErr) { console.error("GPU Validation:", valErr.message); }

  await sumsReadBuf.mapAsync(GPUMapMode.READ);
  const sumsData = new Float32Array(sumsReadBuf.getMappedRange().slice(0));
  sumsReadBuf.unmap();
  E_T = sumsData[0];
  E_eK = sumsData[1];
  E_ee = sumsData[2];
  // Per-H correction (same as ITP path)
  {
    const H_RAW = { 100: [0.5, -1.0], 200: [0.503, -1.028], 300: [0.5, -1.0] };
    const raw = H_RAW[NN] || H_RAW[200];
    let nH = 0;
    for (let a = 0; a < NELEC; a++) { if (Z[a] === 1 && r_cut[a] === 0) nH++; }
    E_T  += nH * (0.5 - raw[0]);
    E_eK += nH * (-1.0 - raw[1]);
  }
  {
    let dip_x = 0, dip_y = 0, dip_z = 0;
    for (let a = 0; a < NELEC; a++) {
      if (Z[a] === 0) continue;
      dip_x += Z[a] * nucPos[a][0] * hGrid;
      dip_y += Z[a] * nucPos[a][1] * hGrid;
      dip_z += Z[a] * nucPos[a][2] * hGrid;
    }
    dip_x -= sumsData[3]; dip_y -= sumsData[4]; dip_z -= sumsData[5];
    dipole_au = Math.sqrt(dip_x*dip_x + dip_y*dip_y + dip_z*dip_z);
    dipole_D = dipole_au * 2.5417;
  }

  await sliceReadBuf.mapAsync(GPUMapMode.READ);
  sliceData = new Float32Array(sliceReadBuf.getMappedRange().slice(0));
  sliceReadBuf.unmap();

  tStep += LOBPCG_ITERS;
  lastMs = performance.now() - t0;

  E_KK = 0;
  if (addNucRepulsion) {
    const soft_nuc = 0.04 * h2v;
    for (let a = 0; a < uniqueNuclei.length; a++) {
      const ea = uniqueNuclei[a].elecIndices[0];
      for (let b = a + 1; b < uniqueNuclei.length; b++) {
        const eb = uniqueNuclei[b].elecIndices[0];
        const d = Math.sqrt(
          ((nucPos[ea][0]-nucPos[eb][0])*hGrid)**2 +
          ((nucPos[ea][1]-nucPos[eb][1])*hGrid)**2 +
          ((nucPos[ea][2]-nucPos[eb][2])*hGrid)**2 + soft_nuc);
        E_KK += uniqueNuclei[a].Z_eff * uniqueNuclei[b].Z_eff / d;
      }
    }
  }
  E = E_T + E_eK + E_ee + E_KK;
  E_bind = E - E_atoms_sum;

  if (!isFinite(E)) { gpuError = "LOBPCG instability"; return; }
  console.log("LOBPCG step " + tStep + ": E=" + E.toFixed(6) + " (" + lastMs.toFixed(0) + "ms)");
}

async function moveNuclei(gpuForces) {
  // Start with electron density gradient forces from GPU
  for (let a = 0; a < NELEC; a++) {
    if (Z[a] === 0) { nucForce[a] = [0,0,0]; nucForceElec[a] = [0,0,0]; continue; }
    nucForce[a] = [gpuForces[a*3], gpuForces[a*3+1], gpuForces[a*3+2]];
    nucForceElec[a] = [gpuForces[a*3], gpuForces[a*3+1], gpuForces[a*3+2]];
  }

  // Add nuclear-nuclear (kernel-kernel) Coulomb repulsion forces using unique nuclei
  // F_A += sum_{B≠A} Z_eff_A * Z_eff_B * (R_A - R_B) / |R_A - R_B|^3
  for (let a = 0; a < uniqueNuclei.length; a++) {
    const ea = uniqueNuclei[a].elecIndices[0];
    let fx = 0, fy = 0, fz = 0;
    for (let b = 0; b < uniqueNuclei.length; b++) {
      if (b === a) continue;
      const eb = uniqueNuclei[b].elecIndices[0];
      const dx = (nucPos[ea][0] - nucPos[eb][0]) * hGrid;
      const dy = (nucPos[ea][1] - nucPos[eb][1]) * hGrid;
      const dz = (nucPos[ea][2] - nucPos[eb][2]) * hGrid;
      const r2 = dx*dx + dy*dy + dz*dz + hGrid*hGrid;
      const r = Math.sqrt(r2);
      const inv_r3 = 1.0 / (r * r2);
      fx += uniqueNuclei[a].Z_eff * uniqueNuclei[b].Z_eff * dx * inv_r3;
      fy += uniqueNuclei[a].Z_eff * uniqueNuclei[b].Z_eff * dy * inv_r3;
      fz += uniqueNuclei[a].Z_eff * uniqueNuclei[b].Z_eff * dz * inv_r3;
    }
    // Apply same force to all electrons of this nucleus
    for (const e of uniqueNuclei[a].elecIndices) {
      nucForce[e][0] += fx;
      nucForce[e][1] += fy;
      nucForce[e][2] += fz;
    }
  }

  // Zero out forces on hinge atoms (free pivot)
  if (window.USER_HINGE_ATOMS) {
    for (const a of window.USER_HINGE_ATOMS) {
      if (a < NELEC) nucForce[a] = [0, 0, 0];
    }
  }

  console.log("Forces (elec+nuc): " + nucForce.filter((_,i) => Z[i]>0).map((f,i) =>
    atomLabels[i]+"=("+f.map(x=>x.toExponential(3)).join(",")+")").join(" "));

  // Log net force on each branch for fold analysis (before elastic update)
  if (window.USER_FOLD_ATOMS) {
    const fa = window.USER_FOLD_ATOMS;
    const hingeAtom = fa[1];
    let s1fx=0, s1fy=0, s2fx=0, s2fy=0;
    let s1n=0, s2n=0;
    for (let a = 0; a < NELEC; a++) {
      if (Z[a] === 0) continue;
      if (a < hingeAtom) { s1fx += nucForce[a][0]; s1fy += nucForce[a][1]; s1n++; }
      else { s2fx += nucForce[a][0]; s2fy += nucForce[a][1]; s2n++; }
    }
    let s1cx=0, s1cy=0, s2cx=0, s2cy=0;
    for (let a = 0; a < NELEC; a++) {
      if (Z[a] === 0) continue;
      if (a < hingeAtom) { s1cx += nucPos[a][0]; s1cy += nucPos[a][1]; }
      else { s2cx += nucPos[a][0]; s2cy += nucPos[a][1]; }
    }
    s1cx /= s1n; s1cy /= s1n; s2cx /= s2n; s2cy /= s2n;
    const dx12 = s2cx - s1cx, dy12 = s2cy - s1cy;
    const d12 = Math.sqrt(dx12*dx12 + dy12*dy12);
    const s1proj = (s1fx * (-dx12) + s1fy * (-dy12)) / d12;
    const s2proj = (s2fx * dx12 + s2fy * dy12) / d12;
    const netUnfold = s1proj + s2proj;
    window._foldNetUnfold = netUnfold;
    window._foldS1proj = s1proj;
    window._foldS2proj = s2proj;
    console.log("FOLD FORCES: NET=" + netUnfold.toExponential(3) + (netUnfold > 0 ? " UNFOLDING" : " FOLDING"));
  }

  // Classical MD: bonds + angles + quantum forces, then full restart
  const cmd = window.USER_CLASSICAL_MD;
  const eb = window.USER_ELASTIC_BACKBONE; // keep for backward compat
  if (cmd && cmd.caIndices) {
    const mdSteps = cmd.mdSteps;
    const mdDt = cmd.mdDt;

    // Coarse-grained approach: move Ca atoms by per-residue net quantum force
    // then SHAKE Ca-Ca distances, then rebuild all atoms from Ca + offsets

    // Need residue groups and Ca indices — reuse from config or detect
    const caIdx = cmd.caIndices;
    const groups = cmd.groups;
    const offsets = cmd.offsets;
    const nRes = caIdx.length;

    // 1. Sum quantum forces per residue
    const resFx = new Array(nRes).fill(0);
    const resFy = new Array(nRes).fill(0);
    const resFz = new Array(nRes).fill(0);
    for (let g = 0; g < nRes; g++) {
      for (const a of groups[g]) {
        if (a < NELEC && Z[a] > 0) {
          resFx[g] += nucForce[a][0];
          resFy[g] += nucForce[a][1];
          resFz[g] += nucForce[a][2];
        }
      }
    }

    // SASA hydrophobic pressure: exposed residues get inward force toward centroid
    const sasa = window.USER_SASA;
    if (sasa) {
      const probeR = sasa.probeRadius || 15.0;  // neighbor cutoff in grid units
      const gamma = sasa.gamma || 0.5;           // surface tension (force scale)
      const maxNeighbors = sasa.maxNeighbors || 6; // fully buried threshold

      // Protein centroid (Ca atoms only)
      let cx = 0, cy = 0, cz = 0;
      for (let g = 0; g < nRes; g++) {
        cx += nucPos[caIdx[g]][0];
        cy += nucPos[caIdx[g]][1];
        cz += nucPos[caIdx[g]][2];
      }
      cx /= nRes; cy /= nRes; cz /= nRes;

      // Per-residue: count neighbors, compute exposure, add inward force
      for (let g = 0; g < nRes; g++) {
        let neighbors = 0;
        const px = nucPos[caIdx[g]][0], py = nucPos[caIdx[g]][1], pz = nucPos[caIdx[g]][2];
        for (let h = 0; h < nRes; h++) {
          if (h === g) continue;
          const dx = nucPos[caIdx[h]][0] - px;
          const dy = nucPos[caIdx[h]][1] - py;
          const dz = nucPos[caIdx[h]][2] - pz;
          if (dx*dx + dy*dy + dz*dz < probeR * probeR) neighbors++;
        }
        // Exposure: 1 = fully exposed, 0 = fully buried
        const exposure = Math.max(0, 1 - neighbors / maxNeighbors);
        if (exposure > 0) {
          // Force toward centroid, proportional to exposure
          const dx = cx - px, dy = cy - py, dz = cz - pz;
          const d = Math.sqrt(dx*dx + dy*dy + dz*dz) + 0.01;
          resFx[g] += gamma * exposure * dx / d;
          resFy[g] += gamma * exposure * dy / d;
          resFz[g] += gamma * exposure * dz / d;
        }
      }
      if (phaseSteps % 500 === 0) {
        console.log("SASA: gamma=" + gamma + " probeR=" + probeR);
      }
    }

    // 2. Move residues — mode-dependent
    if (cmd.mode === 'translate') {
      // Per-residue rigid translation: each residue group moves by its net force (3D)
      // Pin first residue only — chain can compress from free end
      const pinEnds = cmd.pinEnds !== false; // default true
      const maxDisp = 1.0; // max displacement per step (grid cells)
      // During contact-driven folding, dampen quantum repulsion so biases can work
      let qDamp = 1.0;
      if (cmd.contactBias) {
        // Check if any contacts are far from target — if so, dampen quantum forces
        const cb = cmd.contactBias;
        let maxExcess = 0;
        const allPairs = [].concat(cb.hbonds || [], cb.contacts || []);
        for (const p of allPairs) {
          let dx = nucPos[p.b][0]-nucPos[p.a][0], dy = nucPos[p.b][1]-nucPos[p.a][1], dz = nucPos[p.b][2]-nucPos[p.a][2];
          let distAu = Math.sqrt(dx*dx+dy*dy+dz*dz) * hGrid;
          let excess = (distAu - p.target) / p.target;
          if (excess > maxExcess) maxExcess = excess;
        }
        // Scale: if worst contact is >2x target, zero quantum; if >1.5x, 10%; etc.
        if (maxExcess > 1.0) qDamp = 0.0;
        else if (maxExcess > 0.5) qDamp = 0.1;
        else if (maxExcess > 0.2) qDamp = 0.5;
        // else full quantum forces (contacts nearly formed)
      }
      for (let g = 0; g < nRes; g++) {
        // End damping: scale down forces on terminal residues instead of freezing
        const pinN = (typeof pinEnds === 'number') ? pinEnds : (pinEnds === true ? 2 : 0);
        let endDamp = 1.0;
        if (pinN > 0) {
          const distFromEnd = Math.min(g, nRes - 1 - g);
          if (distFromEnd < pinN) endDamp = (distFromEnd + 1) / (pinN + 1); // gradual: 0.25, 0.5, 0.75...
        }
        let dx = resFx[g] * forceScale * mdDt * 0.01 * endDamp * qDamp;
        let dy = resFy[g] * forceScale * mdDt * 0.01 * endDamp * qDamp;
        let dz = resFz[g] * forceScale * mdDt * 0.01 * endDamp * qDamp;
        let mag = Math.sqrt(dx*dx + dy*dy + dz*dz);
        if (mag > maxDisp) { dx *= maxDisp/mag; dy *= maxDisp/mag; dz *= maxDisp/mag; }
        for (const a of groups[g]) {
          if (a >= NELEC || Z[a] === 0) continue;
          nucPos[a][0] += dx;
          nucPos[a][1] += dy;
          nucPos[a][2] += dz;
        }
      }
      // C-term OH follows last residue
      if (groups.length > nRes) {
        let dx = resFx[nRes-1] * forceScale * mdDt * 0.01;
        let dy = resFy[nRes-1] * forceScale * mdDt * 0.01;
        let dz = resFz[nRes-1] * forceScale * mdDt * 0.01;
        let mag = Math.sqrt(dx*dx + dy*dy + dz*dz);
        if (mag > maxDisp) { dx *= maxDisp/mag; dy *= maxDisp/mag; dz *= maxDisp/mag; }
        for (const a of groups[nRes]) {
          if (a >= NELEC || Z[a] === 0) continue;
          nucPos[a][0] += dx;
          nucPos[a][1] += dy;
          nucPos[a][2] += dz;
        }
      }
      // Helical constraint: as chain compresses, rotate residues around axis
      // Alpha helix: 100° per residue around z-axis
      if (cmd.helicalBias) {
        // Find chain axis (first Ca to last Ca)
        const ca0 = nucPos[caIdx[0]], caL = nucPos[caIdx[nRes-1]];
        let hax = caL[0]-ca0[0], hay = caL[1]-ca0[1], haz = caL[2]-ca0[2];
        let haLen = Math.sqrt(hax*hax+hay*hay+haz*haz) + 1e-10;
        hax /= haLen; hay /= haLen; haz /= haLen;
        // Centroid of chain
        let ccx=0, ccy=0, ccz=0;
        for (let g = 0; g < nRes; g++) {
          ccx += nucPos[caIdx[g]][0]; ccy += nucPos[caIdx[g]][1]; ccz += nucPos[caIdx[g]][2];
        }
        ccx /= nRes; ccy /= nRes; ccz /= nRes;

        const idealAnglePerRes = (2 * Math.PI / 3.6); // 100° in radians
        const idealRadiusGrid = (cmd.helicalBias.idealRadius || 4.35) / hGrid;
        const biasStrength = cmd.helicalBias.strength || 0.02;

        for (let g = 1; g < nRes; g++) { // skip pinned res 0
          const ca = nucPos[caIdx[g]];
          // Project Ca onto axis to get axial position
          let rx = ca[0]-ccx, ry = ca[1]-ccy, rz = ca[2]-ccz;
          let axialProj = rx*hax + ry*hay + rz*haz;
          // Perpendicular vector from axis to Ca
          let px = rx - axialProj*hax;
          let py = ry - axialProj*hay;
          let pz = rz - axialProj*haz;
          let pLen = Math.sqrt(px*px + py*py + pz*pz) + 1e-10;

          // Target angle for this residue
          let targetAngle = g * idealAnglePerRes;
          // Target position: on helix at idealRadius from axis
          // Use axis-perpendicular plane basis vectors
          // e1 = initial perp direction of res 1, e2 = axis × e1
          // Simplified: nudge radially outward if too close to axis
          let radialForce = (idealRadiusGrid - pLen) * biasStrength;
          let nudgeX = (px / pLen) * radialForce;
          let nudgeY = (py / pLen) * radialForce;
          let nudgeZ = (pz / pLen) * radialForce;

          for (const a of groups[g]) {
            if (a >= NELEC || Z[a] === 0) continue;
            nucPos[a][0] += nudgeX;
            nucPos[a][1] += nudgeY;
            nucPos[a][2] += nudgeZ;
          }
        }
      }

      // SHAKE: enforce min/max Ca-Ca distance to prevent over-compression
      // Target Ca-Ca ~ 3.8 A = 7.18 au; allow ±30%
      const targetCaCa = (cmd.targetCaCa || 7.18) / hGrid; // in grid cells
      const shakeTol = cmd.shakeTolerance || 0.15; // ±15% default
      const minCaCa = targetCaCa * (1 - shakeTol);
      const maxCaCa = targetCaCa * (1 + shakeTol);
      for (let iter = 0; iter < 5; iter++) {
        for (let g = 0; g < nRes - 1; g++) {
          const a1 = caIdx[g], a2 = caIdx[g+1];
          let dx = nucPos[a2][0] - nucPos[a1][0];
          let dy = nucPos[a2][1] - nucPos[a1][1];
          let dz = nucPos[a2][2] - nucPos[a1][2];
          let dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
          if (dist < minCaCa || dist > maxCaCa) {
            let target = dist < minCaCa ? minCaCa : maxCaCa;
            let corr = (target - dist) / (2 * dist + 1e-10);
            let cx = dx * corr, cy = dy * corr, cz = dz * corr;
            // Move both Ca and their entire residue groups
            for (const a of groups[g]) {
              if (a >= NELEC || Z[a] === 0) continue;
              nucPos[a][0] -= cx; nucPos[a][1] -= cy; nucPos[a][2] -= cz;
            }
            for (const a of groups[g+1]) {
              if (a >= NELEC || Z[a] === 0) continue;
              nucPos[a][0] += cx; nucPos[a][1] += cy; nucPos[a][2] += cz;
            }
          }
        }
      }

      // Steric repulsion: prevent non-bonded Ca atoms from getting too close
      // Mimics van der Waals excluded volume — pushes apart residues closer than minNB
      {
        const minNB_au = 3.0;  // minimum non-bonded Ca-Ca distance in Å
        const minNB = minNB_au * 1.8897 / hGrid;  // in grid cells
        const stericStr = 0.5;  // grid cells per step repulsion
        for (let g1 = 0; g1 < nRes; g1++) {
          for (let g2 = g1 + 2; g2 < nRes; g2++) {  // skip bonded neighbors (g1±1)
            const a1 = caIdx[g1], a2 = caIdx[g2];
            let dx = nucPos[a2][0] - nucPos[a1][0];
            let dy = nucPos[a2][1] - nucPos[a1][1];
            let dz = nucPos[a2][2] - nucPos[a1][2];
            let dist = Math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-10;
            if (dist < minNB) {
              let push = stericStr * (minNB - dist) / dist;
              let px = dx * push * 0.5, py = dy * push * 0.5, pz = dz * push * 0.5;
              for (const a of groups[g1]) {
                if (a >= NELEC || Z[a] === 0) continue;
                nucPos[a][0] -= px; nucPos[a][1] -= py; nucPos[a][2] -= pz;
              }
              for (const a of groups[g2]) {
                if (a >= NELEC || Z[a] === 0) continue;
                nucPos[a][0] += px; nucPos[a][1] += py; nucPos[a][2] += pz;
              }
            }
          }
        }
      }

      // Contact bias: targeted attractions for H-bonds and hydrophobic contacts
      if (cmd.contactBias) {
        const cb = cmd.contactBias;
        const hbStr = (cb.hbondStrength || 0.5) ;
        const ctStr = (cb.contactStrength || 1.0);

        // Helper: apply pairwise attractive bias between two atoms
        // Moves the residue groups containing each atom toward each other
        function applyPairBias(atomA, atomB, targetAu, strength) {
          let dx = nucPos[atomB][0] - nucPos[atomA][0];
          let dy = nucPos[atomB][1] - nucPos[atomA][1];
          let dz = nucPos[atomB][2] - nucPos[atomA][2];
          let distGrid = Math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-10;
          let distAu = distGrid * hGrid;
          if (distAu <= targetAu) return distAu; // already close enough
          let excess = (distAu - targetAu) / distAu;
          let nudge = strength * excess;
          let nx = dx / distGrid * nudge, ny = dy / distGrid * nudge, nz = dz / distGrid * nudge;
          // Find which residue groups these atoms belong to
          let gA = -1, gB = -1;
          for (let g = 0; g < groups.length; g++) {
            if (groups[g].indexOf(atomA) >= 0) gA = g;
            if (groups[g].indexOf(atomB) >= 0) gB = g;
          }
          if (gA >= 0) {
            for (const a of groups[gA]) {
              if (a >= NELEC || Z[a] === 0) continue;
              nucPos[a][0] += nx * 0.5; nucPos[a][1] += ny * 0.5; nucPos[a][2] += nz * 0.5;
            }
          }
          if (gB >= 0) {
            for (const a of groups[gB]) {
              if (a >= NELEC || Z[a] === 0) continue;
              nucPos[a][0] -= nx * 0.5; nucPos[a][1] -= ny * 0.5; nucPos[a][2] -= nz * 0.5;
            }
          }
          return distAu;
        }

        // H-bond biases
        let hbLog = [];
        if (cb.hbonds) {
          for (const hb of cb.hbonds) {
            let d = applyPairBias(hb.a, hb.b, hb.target, hbStr);
            hbLog.push(hb.label + "=" + (d * 0.529177).toFixed(1));
          }
        }
        // Hydrophobic contact biases
        let ctLog = [];
        if (cb.contacts) {
          for (const ct of cb.contacts) {
            let d = applyPairBias(ct.a, ct.b, ct.target, ctStr);
            ctLog.push(ct.label + "=" + (d * 0.529177).toFixed(1));
          }
        }

        // Compute and report Rg
        let cmx = 0, cmy = 0, cmz = 0;
        for (let g = 0; g < nRes; g++) { cmx += nucPos[caIdx[g]][0]; cmy += nucPos[caIdx[g]][1]; cmz += nucPos[caIdx[g]][2]; }
        cmx /= nRes; cmy /= nRes; cmz /= nRes;
        let rg2 = 0;
        for (let g = 0; g < nRes; g++) { let ddx = nucPos[caIdx[g]][0]-cmx, ddy = nucPos[caIdx[g]][1]-cmy, ddz = nucPos[caIdx[g]][2]-cmz; rg2 += ddx*ddx+ddy*ddy+ddz*ddz; }
        let rgAu = Math.sqrt(rg2 / nRes) * hGrid;
        window._compactRg = rgAu;
        window._compactTargetRg = 0; // no global target, contact-driven

        if (nucStepCount % 5 === 0) console.log("CONTACTS: Rg=" + (rgAu*0.529177).toFixed(1) + "A  HB:[" + hbLog.join(" ") + "]  CT:[" + ctLog.join(" ") + "]");
      }

      console.log("TRANSLATE MD: " + nRes + " residues, forces=" +
        resFx.map((f,i) => "(" + f.toExponential(1) + "," + resFy[i].toExponential(1) + "," + resFz[i].toExponential(1) + ")").join(" "));
    } else {
      // Default: Two rigid strands pivoting at hinge (for hairpin folding)
      const hingeRes = cmd.hingeRes || Math.floor(nRes/2);
      const hingeX = (nucPos[caIdx[hingeRes-1]][0] + nucPos[caIdx[hingeRes]][0]) / 2;
      const hingeY = (nucPos[caIdx[hingeRes-1]][1] + nucPos[caIdx[hingeRes]][1]) / 2;

      let torque1 = 0, torque2 = 0;
      for (let g = 0; g < hingeRes; g++) {
        const rx = nucPos[caIdx[g]][0] - hingeX;
        const ry = nucPos[caIdx[g]][1] - hingeY;
        torque1 += rx * resFy[g] - ry * resFx[g];
      }
      for (let g = hingeRes; g < nRes; g++) {
        const rx = nucPos[caIdx[g]][0] - hingeX;
        const ry = nucPos[caIdx[g]][1] - hingeY;
        torque2 += rx * resFy[g] - ry * resFx[g];
      }

      const maxAngle = 0.02;
      let dTheta1 = torque1 * forceScale * mdDt * 0.0001;
      let dTheta2 = torque2 * forceScale * mdDt * 0.0001;
      dTheta1 = Math.max(-maxAngle, Math.min(maxAngle, dTheta1));
      dTheta2 = Math.max(-maxAngle, Math.min(maxAngle, dTheta2));

      const cos1 = Math.cos(dTheta1), sin1 = Math.sin(dTheta1);
      for (let g = 0; g < hingeRes; g++) {
        for (const a of groups[g]) {
          if (a >= NELEC || Z[a] === 0) continue;
          const rx = nucPos[a][0] - hingeX;
          const ry = nucPos[a][1] - hingeY;
          nucPos[a][0] = hingeX + rx * cos1 - ry * sin1;
          nucPos[a][1] = hingeY + rx * sin1 + ry * cos1;
        }
      }
      const cos2 = Math.cos(dTheta2), sin2 = Math.sin(dTheta2);
      for (let g = hingeRes; g < nRes; g++) {
        for (const a of groups[g]) {
          if (a >= NELEC || Z[a] === 0) continue;
          const rx = nucPos[a][0] - hingeX;
          const ry = nucPos[a][1] - hingeY;
          nucPos[a][0] = hingeX + rx * cos2 - ry * sin2;
          nucPos[a][1] = hingeY + rx * sin2 + ry * cos2;
        }
      }
      if (groups.length > nRes) {
        for (const a of groups[nRes]) {
          if (a >= NELEC || Z[a] === 0) continue;
          const rx = nucPos[a][0] - hingeX;
          const ry = nucPos[a][1] - hingeY;
          nucPos[a][0] = hingeX + rx * cos2 - ry * sin2;
          nucPos[a][1] = hingeY + rx * sin2 + ry * cos2;
        }
      }
      console.log("RIGID STRANDS: torque1=" + torque1.toExponential(2) +
        " torque2=" + torque2.toExponential(2) +
        " dTheta1=" + (dTheta1*180/Math.PI).toFixed(3) + "° dTheta2=" + (dTheta2*180/Math.PI).toFixed(3) + "°");
    }

    // Boundary clamp (3D)
    for (let a = 0; a < NELEC; a++) {
      if (Z[a] === 0) continue;
      nucPos[a][0] = Math.max(5, Math.min(NN - 5, nucPos[a][0]));
      nucPos[a][1] = Math.max(5, Math.min(NN - 5, nucPos[a][1]));
      nucPos[a][2] = Math.max(5, Math.min(NN - 5, nucPos[a][2]));
    }

    console.log("CLASSICAL MD: SHAKE + offsets, " + nRes + " residues — FULL RESTART");
    nucStepCount++;

    // Full restart: re-initialize wavefunctions, labels, K, P from scratch
    fillAtomBuf();

    // Clear K, labels, U, P
    const clrEnc = device.createCommandEncoder();
    clrEnc.clearBuffer(K_buf);
    clrEnc.clearBuffer(labelBuf);
    clrEnc.clearBuffer(U_buf[0]);
    clrEnc.clearBuffer(P_buf[0]);
    device.queue.submit([clrEnc.finish()]);

    // Clear bestR2
    const bU = new Float32Array(S3);
    bU.fill(0.0);
    device.queue.writeBuffer(bestR2Buf, 0, bU);

    // Batched init: recompute K and assign labels
    const nBatches = Math.ceil(NELEC / INIT_BATCH);
    for (let b = 0; b < nBatches; b++) {
      const start = b * INIT_BATCH;
      const count = Math.min(INIT_BATCH, NELEC - start);
      device.queue.writeBuffer(initRangeBuf, 0, new Uint32Array([start, count, 0, 0]));
      const enc = device.createCommandEncoder();
      const ip = enc.beginComputePass();
      ip.setPipeline(gpuInitAccumPL);
      ip.setBindGroup(0, gpuInitAccumBG);
      dispatchLinear(ip, S3);
      ip.end();
      device.queue.submit([enc.finish()]);
    }

    // Final pass: set U and P from bestR2
    {
      const enc = device.createCommandEncoder();
      const ip = enc.beginComputePass();
      ip.setPipeline(gpuInitFinalPL);
      ip.setBindGroup(0, gpuInitFinalBG);
      dispatchLinear(ip, S3);
      ip.end();
      device.queue.submit([enc.finish()]);
    }

    // Copy U[0] to U[1]
    {
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(U_buf[0], 0, U_buf[1], 0, S3 * 4);
      device.queue.submit([enc.finish()]);
    }

    await device.queue.onSubmittedWorkDone();

    // Reset step counter so it runs N_MOVE steps before next force computation
    tStep = 0;
    phaseSteps = 0;

    return; // skip normal per-atom dynamics
  }

  // Log net force on each branch for fold analysis
  if (window.USER_FOLD_ATOMS) {
    const fa = window.USER_FOLD_ATOMS; // [strand1_end, hinge, strand2_end]
    // Strand1 atoms: indices 0..44 (residues 0-5), Strand2: 45..86 (residues 6-11)
    const hingeAtom = fa[1]; // atom 45
    let s1fx=0, s1fy=0, s2fx=0, s2fy=0;
    let s1n=0, s2n=0;
    for (let a = 0; a < NELEC; a++) {
      if (Z[a] === 0) continue;
      if (a < hingeAtom) { s1fx += nucForce[a][0]; s1fy += nucForce[a][1]; s1n++; }
      else { s2fx += nucForce[a][0]; s2fy += nucForce[a][1]; s2n++; }
    }
    // Compute center of mass of each strand
    let s1cx=0, s1cy=0, s2cx=0, s2cy=0;
    for (let a = 0; a < NELEC; a++) {
      if (Z[a] === 0) continue;
      if (a < hingeAtom) { s1cx += nucPos[a][0]; s1cy += nucPos[a][1]; s1n; }
      else { s2cx += nucPos[a][0]; s2cy += nucPos[a][1]; s2n; }
    }
    s1cx /= s1n; s1cy /= s1n; s2cx /= s2n; s2cy /= s2n;
    // Vector from strand1 COM to strand2 COM
    const dx12 = s2cx - s1cx, dy12 = s2cy - s1cy;
    const d12 = Math.sqrt(dx12*dx12 + dy12*dy12);
    // Project net forces along inter-strand axis (positive = strands moving apart = unfolding)
    const s1proj = (s1fx * (-dx12) + s1fy * (-dy12)) / d12; // strand1 force away from strand2
    const s2proj = (s2fx * dx12 + s2fy * dy12) / d12;       // strand2 force away from strand1
    const netUnfold = s1proj + s2proj; // positive = unfolding, negative = folding
    window._foldNetUnfold = netUnfold;
    window._foldS1proj = s1proj;
    window._foldS2proj = s2proj;
    console.log("FOLD FORCES: strand1=(" + s1fx.toExponential(3) + "," + s1fy.toExponential(3) +
      ") strand2=(" + s2fx.toExponential(3) + "," + s2fy.toExponential(3) +
      ") | along axis: s1=" + s1proj.toExponential(3) + " s2=" + s2proj.toExponential(3) +
      " | NET=" + netUnfold.toExponential(3) + (netUnfold > 0 ? " UNFOLDING" : " FOLDING"));
  }

  // If USER_RIGID_GROUPS defined, move each group as rigid body (average force)
  // Format: [[a0,a1,...], [a2,a3,...], ...] — atom indices per group
  const rigidGroups = window.USER_RIGID_GROUPS;

  // Harmonic bond-length constraints: V = ½·k·(r-r0)² for each [i, j, r0, k].
  // USER_BOND_CONSTRAINTS = [[i, j, r0_au, k_Ha/au²], ...]
  const bondConstraints = window.USER_BOND_CONSTRAINTS;
  if (bondConstraints && bondConstraints.length > 0) {
    for (let bc = 0; bc < bondConstraints.length; bc++) {
      const i1 = bondConstraints[bc][0], i2 = bondConstraints[bc][1];
      const r0 = bondConstraints[bc][2], kBond = bondConstraints[bc][3];
      const rx = (nucPos[i1][0] - nucPos[i2][0]) * hGrid;
      const ry = (nucPos[i1][1] - nucPos[i2][1]) * hGrid;
      const rz = (nucPos[i1][2] - nucPos[i2][2]) * hGrid;
      const r = Math.sqrt(rx*rx + ry*ry + rz*rz);
      if (r < 1e-8) continue;
      const ux = rx/r, uy = ry/r, uz = rz/r;
      const f = -kBond * (r - r0);  // restoring magnitude (negative if stretched)
      nucForce[i1][0] += f * ux; nucForce[i1][1] += f * uy; nucForce[i1][2] += f * uz;
      nucForce[i2][0] -= f * ux; nucForce[i2][1] -= f * uy; nucForce[i2][2] -= f * uz;
    }
  }

  // Harmonic H-O-H angle restoring force: V = ½·k·(θ−θ0)² per water triplet.
  // Prevents drift of water's bend angle away from 104.5° in reduced-O models.
  const triplets = window.USER_WATER_TRIPLETS;
  if (triplets && triplets.length > 0) {
    const theta0 = (window.USER_HOH_EQ || 104.5) * Math.PI / 180;
    const kBend = window.USER_BEND_K !== undefined ? window.USER_BEND_K : 1.5;  // Ha/rad²
    for (let t = 0; t < triplets.length; t++) {
      const o = triplets[t][0], h1 = triplets[t][1], h2 = triplets[t][2];
      const r1x = (nucPos[h1][0] - nucPos[o][0]) * hGrid;
      const r1y = (nucPos[h1][1] - nucPos[o][1]) * hGrid;
      const r1z = (nucPos[h1][2] - nucPos[o][2]) * hGrid;
      const r3x = (nucPos[h2][0] - nucPos[o][0]) * hGrid;
      const r3y = (nucPos[h2][1] - nucPos[o][1]) * hGrid;
      const r3z = (nucPos[h2][2] - nucPos[o][2]) * hGrid;
      const d1 = Math.sqrt(r1x*r1x + r1y*r1y + r1z*r1z);
      const d3 = Math.sqrt(r3x*r3x + r3y*r3y + r3z*r3z);
      if (d1 < 1e-6 || d3 < 1e-6) continue;
      const u1x = r1x/d1, u1y = r1y/d1, u1z = r1z/d1;
      const u3x = r3x/d3, u3y = r3y/d3, u3z = r3z/d3;
      let cos = u1x*u3x + u1y*u3y + u1z*u3z;
      if (cos > 0.9999) cos = 0.9999;
      if (cos < -0.9999) cos = -0.9999;
      const theta = Math.acos(cos);
      const sinT = Math.sin(theta);
      if (sinT < 1e-6) continue;
      const coeff = kBend * (theta - theta0) / sinT;
      const f1x = coeff * (u3x - cos*u1x) / d1;
      const f1y = coeff * (u3y - cos*u1y) / d1;
      const f1z = coeff * (u3z - cos*u1z) / d1;
      const f3x = coeff * (u1x - cos*u3x) / d3;
      const f3y = coeff * (u1y - cos*u3y) / d3;
      const f3z = coeff * (u1z - cos*u3z) / d3;
      nucForce[h1][0] += f1x; nucForce[h1][1] += f1y; nucForce[h1][2] += f1z;
      nucForce[h2][0] += f3x; nucForce[h2][1] += f3y; nucForce[h2][2] += f3z;
      nucForce[o][0]  -= (f1x + f3x);
      nucForce[o][1]  -= (f1y + f3y);
      nucForce[o][2]  -= (f1z + f3z);
    }
  }

  for (let sub = 0; sub < NUC_SUBSTEPS; sub++) {
    if (rigidGroups) {
      // Rigid group dynamics: average force over group, move all atoms together
      for (const group of rigidGroups) {
        let gfx = 0, gfy = 0, gfz = 0, totalMass = 0;
        for (const a of group) {
          if (a >= NELEC || Z[a] === 0) continue;
          gfx += nucForce[a][0];
          gfy += nucForce[a][1];
          gfz += nucForce[a][2];
          totalMass += nucMass(Z[a]);
        }
        if (totalMass === 0) continue;
        // Shared velocity for the group (use first atom's velocity as group velocity)
        const g0 = group[0];
        for (let d = 0; d < 3; d++) {
          const gf = d === 0 ? gfx : d === 1 ? gfy : gfz;
          nucVel[g0][d] += gf / totalMass * DT_NUC * forceScale;
          if (langevinKT <= 0) nucVel[g0][d] *= DAMPING;
          nucVel[g0][d] = Math.max(-MAX_VEL, Math.min(MAX_VEL, nucVel[g0][d]));
        }
        // Move all atoms in group by same displacement
        for (const a of group) {
          if (a >= NELEC || Z[a] === 0) continue;
          for (let d = 0; d < 3; d++) {
            nucPos[a][d] += nucVel[g0][d] * DT_NUC / hGrid;
            nucPos[a][d] = Math.max(5, Math.min(NN - 5, nucPos[a][d]));
          }
        }
      }
    } else {
      // Per-atom dynamics with optional thermostat
      const frozenAtoms = window.USER_FROZEN_ATOMS || [];
      // Optional shear body force: F_x = m · shearRate · (y - y_center).
      // Produces steady-state v_x ∝ (y - y_center) under Langevin damping.
      const shearRate = window.USER_SHEAR_RATE || 0;
      let shearYC = 0;
      if (shearRate !== 0) {
        let yS = 0, nS = 0;
        for (let a = 0; a < NELEC; a++) {
          if (Z[a] === 0 && Z_nuc[a] === 0) continue;
          if (frozenAtoms.indexOf(a) >= 0) continue;
          yS += nucPos[a][1]; nS++;
        }
        shearYC = nS > 0 ? yS / nS : 0;
        window._shearYC = shearYC;
      }
      for (let a = 0; a < NELEC; a++) {
        if (Z[a] === 0 && Z_nuc[a] === 0) continue;
        if (frozenAtoms.indexOf(a) >= 0) continue;
        const m = nucMass(Z_nuc[a] || Z[a]);
        if (shearRate !== 0) {
          // nucPos is in grid cells; multiply by hGrid to get au. F in Ha/au.
          const yA_au = (nucPos[a][1] - shearYC) * hGrid;
          const fShear = m * shearRate * yA_au;
          nucForce[a][0] += fShear;
          if (nucForceTotal[a]) nucForceTotal[a][0] += fShear;  // make visible in arrows
          // Work done by TOTAL force (electronic + nuclear + shear) this substep: F · v · dt.
          const dt_au = DT_NUC;
          const wDx = nucForce[a][0] * nucVel[a][0] * dt_au;
          const wDy = nucForce[a][1] * nucVel[a][1] * dt_au;
          const wDz = nucForce[a][2] * nucVel[a][2] * dt_au;
          window._shearWork = (window._shearWork || 0) + wDx + wDy + wDz;
        }
        if (window.USER_BROWNIAN) {
          // Overdamped Langevin: x_new = x + F·dt/γ + √(2 kT dt/γ)·gaussRand()
          const gammaBD = window.USER_BROWNIAN_GAMMA || 1.0;
          const driftCoeff = DT_NUC * forceScale / gammaBD;
          const noiseCoeff = Math.sqrt(2 * Math.max(langevinKT, 0) * DT_NUC / gammaBD);
          function gR_BD() { let u1 = Math.random(), u2 = Math.random(); return Math.sqrt(-2*Math.log(u1)) * Math.cos(2*Math.PI*u2); }
          for (let d = 0; d < 3; d++) {
            const dx = (nucForce[a][d] * driftCoeff + noiseCoeff * gR_BD()) / hGrid;
            nucPos[a][d] += dx;
            nucPos[a][d] = Math.max(5, Math.min(NN - 5, nucPos[a][d]));
          }
        } else {
          for (let d = 0; d < 3; d++) {
            nucVel[a][d] += nucForce[a][d] / m * DT_NUC * forceScale;
            if (langevinKT <= 0) nucVel[a][d] *= DAMPING;
            nucVel[a][d] = Math.max(-MAX_VEL, Math.min(MAX_VEL, nucVel[a][d]));
            nucPos[a][d] += nucVel[a][d] * DT_NUC / hGrid;
            nucPos[a][d] = Math.max(5, Math.min(NN - 5, nucPos[a][d]));
          }
        }
      }
      // Andersen thermostat: random velocity + position kicks (skipped when Brownian active)
      if (langevinKT > 0 && !window.USER_BROWNIAN) {
        function gaussRand() {
          let u1 = Math.random(), u2 = Math.random();
          return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        }
        const collisionRate = LANGEVIN_GAMMA;
        for (let a = 0; a < NELEC; a++) {
          if (Z[a] === 0 && Z_nuc[a] === 0) continue;
          if (frozenAtoms.indexOf(a) >= 0) continue;
          if (Math.random() < collisionRate) {
            const m = nucMass(Z_nuc[a] || Z[a]);
            const sigma_v = Math.sqrt(langevinKT / m);
            nucVel[a][0] = sigma_v * gaussRand();
            nucVel[a][1] = sigma_v * gaussRand();
            nucVel[a][2] = sigma_v * gaussRand();
            // Position kick: scale by USER_THERMAL_DISPLACEMENT (grid cells) if set,
            // otherwise use sigma_v * DT_NUC / hGrid (physically correct but tiny)
            const sigma_x = window.USER_THERMAL_KICK || (sigma_v * DT_NUC / hGrid);
            nucPos[a][0] += sigma_x * gaussRand();
            nucPos[a][1] += sigma_x * gaussRand();
            nucPos[a][2] += sigma_x * gaussRand();
            nucPos[a][0] = Math.max(5, Math.min(NN - 5, nucPos[a][0]));
            nucPos[a][1] = Math.max(5, Math.min(NN - 5, nucPos[a][1]));
            nucPos[a][2] = Math.max(5, Math.min(NN - 5, nucPos[a][2]));
          }
        }
      }
    }
  }

  nucStepCount++;
  // Report temperature if thermostat active
  if (langevinKT > 0) {
    let _ke = 0, _na = 0;
    const _frozen = window.USER_FROZEN_ATOMS || [];
    for (let a = 0; a < NELEC; a++) {
      if (Z[a] === 0 && Z_nuc[a] === 0) continue;
      if (_frozen.indexOf(a) >= 0) continue;
      const _m = nucMass(Z_nuc[a] || Z[a]);
      _ke += 0.5 * _m * (nucVel[a][0]**2 + nucVel[a][1]**2 + nucVel[a][2]**2);
      _na++;
    }
    const _T = _na > 0 && _ke > 0 ? (2 * _ke / (3 * _na) * 315775) : 0;
    window._thermoT = _T;
    console.log("Nuc step " + nucStepCount + " T=" + _T.toFixed(0) + " K (target " + (langevinKT*315775).toFixed(0) + " K) KE=" + _ke.toExponential(3) + " nAtoms=" + _na + " NELEC=" + NELEC);
  } else {
    console.log("Nuc step " + nucStepCount + ": " +
      nucPos.filter((_, i) => Z[i] > 0).map(p => "(" + p.map(x => x.toFixed(2)).join(",") + ")").join(" "));
  }

  // Flag that atom positions changed — draw() will capture frame after rendering
  window._nucUpdated = true;

  // Update atomBuf with new positions, recompute K on GPU (batched, keep converged U, labels, P)
  fillAtomBuf();
  // Clear K, then re-accumulate in batches (single-pass 1000-atom shader times out on GPU)
  const clrEnc = device.createCommandEncoder();
  clrEnc.clearBuffer(K_buf);
  device.queue.submit([clrEnc.finish()]);
  const nBatches = Math.ceil(NELEC / INIT_BATCH);
  for (let b = 0; b < nBatches; b++) {
    const start = b * INIT_BATCH;
    const count = Math.min(INIT_BATCH, NELEC - start);
    device.queue.writeBuffer(initRangeBuf, 0, new Uint32Array([start, count, 0, 0]));
    const enc = device.createCommandEncoder();
    const kp = enc.beginComputePass();
    kp.setPipeline(gpuInitAccumPL);
    kp.setBindGroup(0, gpuInitAccumBG);
    dispatchLinear(kp, S3);
    kp.end();
    device.queue.submit([enc.finish()]);
  }
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
    if (initProgress > 0 && initProgress < 1) {
      text("Initializing " + NELEC + " atoms... " + Math.round(initProgress * 100) + "%", 10, 200);
      noFill(); stroke(255); rect(10, 210, 380, 10);
      fill(0, 255, 0); noStroke(); rect(10, 210, 380 * initProgress, 10);
    } else {
      text("Initializing WebGPU...", 10, 200);
    }
    return;
  }

  if (!computing && phase === 0) {
    computing = true;
    computePromise = (useLOBPCG ? doLOBPCGStep() : doSteps(STEPS_PER_FRAME)).then(() => {
      computing = false;
      phaseSteps += useLOBPCG ? LOBPCG_ITERS : STEPS_PER_FRAME;
      if (isFinite(E) && E < E_min) E_min = E;


      // Unfreeze boundary and perturb Z at specified step
      if (window.FREEZE_BOUNDARY && window.UNLOCK_AT_STEP && phaseSteps >= window.UNLOCK_AT_STEP) {
        window.FREEZE_BOUNDARY = false;
        console.log("=== Boundary unfrozen at step " + phaseSteps + " ===");
      }
      if (window.PERTURB_Z_AT_STEP && phaseSteps >= window.PERTURB_Z_AT_STEP) {
        Z[0] = 1.01; Z[1] = 0.99;
        updateParamsBuf();
        console.log("=== Z perturbed to [1.01, 0.99] at step " + phaseSteps + " ===");
        delete window.PERTURB_Z_AT_STEP;
      }
      if (window.RESTORE_Z_AT_STEP && phaseSteps >= window.RESTORE_Z_AT_STEP) {
        Z[0] = 1; Z[1] = 1;
        updateParamsBuf();
        console.log("=== Z restored to [1, 1] at step " + phaseSteps + " ===");
        delete window.RESTORE_Z_AT_STEP;
      }

      // Auto-enable dynamics after convergence (skip if adaptive sweep)
      if (!dynamicsEnabled && !window.CONVERGENCE_THRESHOLD && !window.NO_AUTO_DYNAMICS && phaseSteps >= 10000) {
        dynamicsEnabled = true;
        console.log("=== Dynamics auto-enabled at step " + phaseSteps + " ===");
      }

      // Adaptive convergence check
      if (!dynamicsEnabled && window.CONVERGENCE_THRESHOLD && phaseSteps > 200) {
        if (window._prevE !== undefined && Math.abs(E - window._prevE) < window.CONVERGENCE_THRESHOLD && isFinite(E)) {
          if (!window._convCount) window._convCount = 0;
          window._convCount++;
          if (window._convCount >= 3) {  // 3 consecutive frames within threshold
            console.log("=== CONVERGED at step " + phaseSteps + ": E=" + E.toFixed(6) + " (dE=" + Math.abs(E - window._prevE).toFixed(6) + ") ===");
            phase = 1;
            window._convCount = 0;
            if (window.onSweepDone) window.onSweepDone(E, phaseSteps, E_T, E_eK, E_ee, E_KK);
          }
        } else {
          window._convCount = 0;
        }
        window._prevE = E;
      }

      if (!dynamicsEnabled && phaseSteps >= TOTAL_STEPS) {
        console.log("=== DONE: E=" + E.toFixed(6) + " ===");
        phase = 1;  // done
        if (window.onSweepDone) window.onSweepDone(E, phaseSteps, E_T, E_eK, E_ee, E_KK);
      }
    }).catch((e) => {
      gpuError = e.message || String(e);
      console.error("GPU step failed:", e);
      computing = false;
    }).finally(() => { computePromise = null; });
  }

  if (sliceData) {
    const SS = S;
    const SS2 = SS * SS;
    // Diagnostic: check density values
    {
      let maxD = 0, nanCount = 0, posCount = 0, negCount = 0;
      for (let i = 1; i < NN; i++) {
        for (let j = 1; j < NN; j++) {
          const v = sliceData[i * SS + j];
          if (isNaN(v)) nanCount++;
          else if (v > 0) { posCount++; if (v > maxD) maxD = v; }
          else if (v < 0) negCount++;
        }
      }
      window._diagMaxD = maxD;
      window._diagPos = posCount;
      window._diagNan = nanCount;
      window._diagNeg = negCount;
      if (frameCount <= 3 || frameCount % 100 === 0) {
        console.log("frame=" + frameCount + " maxDens=" + maxD.toExponential(3) +
          " posCount=" + posCount + " nanCount=" + nanCount + " negCount=" + negCount +
          " E_T=" + E_T.toExponential(3) + " E_eK=" + E_eK.toExponential(3));
      }
    }
    // Skip 2D density rendering when in 3D view mode
    if (!window._view3D) {
    loadPixels();
    const d = pixelDensity();
    const W = CANVAS_SIZE * d, H = CANVAS_SIZE * d;
    for (let p = 0; p < W * H * 4; p += 4) {
      pixels[p] = 0; pixels[p+1] = 0; pixels[p+2] = 0; pixels[p+3] = 255;
    }
    // Per-domain colors: cycle through distinct hues
    const domainRGB = [
      [1,0.3,0.3], [0.3,0.6,1], [0.3,1,0.3], [1,1,0],
      [1,0.5,0], [0.6,0.3,1], [0,1,1], [1,0.3,0.7],
      [0.5,1,0.5], [1,0.7,0.3], [0.3,1,0.8], [0.8,0.5,1],
    ];
    // Element colors fallback: H=yellow, O=red, N=blue, C=green
    const zRGB = {1:[1,1,0], 2:[1,0,0], 3:[0,0.5,1], 4:[0,1,0]};
    // Auto-scale: use 99th percentile to avoid nuclear spikes dominating
    const densVals = [];
    for (let i = 1; i < NN; i++) {
      for (let j = 1; j < NN; j++) {
        const v = sliceData[i * SS + j];
        if (v > 0) densVals.push(v);
      }
    }
    densVals.sort((a, b) => a - b);
    const p99 = densVals.length > 0 ? densVals[Math.floor(densVals.length * 0.99)] : 1;
    const invMax = p99 > 0 ? 1.0 / p99 : 1.0;
    for (let i = 1; i < NN; i++) {
      const px0 = Math.floor(PX * i * d);
      const px1 = Math.floor(PX * (i + 1) * d);
      for (let j = 1; j < NN; j++) {
        const py0 = Math.floor(PX * j * d);
        const py1 = Math.floor(PX * (j + 1) * d);
        const b = i * SS + j;
        const dens = sliceData[b];              // slot 0: total density
        const packed = sliceData[SS2 + b];       // slot 1: label + Z*1000
        const domLabel = Math.round(packed % 1000);
        const Zel = Math.round(packed / 1000);
        const bnd = sliceData[2 * SS2 + b];     // slot 2: boundary
        const norm = Math.min(1.0, dens * invMax);
        const brightness = 255 * Math.sqrt(norm);
        // Use domain color if multiple atoms share same element, else element color
        const rgb = domainRGB[domLabel % domainRGB.length] || zRGB[Zel] || [0.5, 0.5, 0.5];
        let ri = Math.min(255, Math.floor(brightness * rgb[0]));
        let gi = Math.min(255, Math.floor(brightness * rgb[1]));
        let bi = Math.min(255, Math.floor(brightness * rgb[2]));
        // Boundary overlay: active=brighten, broken=bright white
        if (bnd > 1.5) {
          // Broken boundary — no density on either side — bright white
          ri = 255; gi = 255; bi = 255;
        } else if (bnd > 0.5) {
          // Active boundary — density present — brighten slightly
          ri = Math.min(255, ri + 40);
          gi = Math.min(255, gi + 40);
          bi = Math.min(255, bi + 40);
        }
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

    // Density line plots: up to 5 lines evenly spaced through atom region
    const kernelJs = [];
    for (let n = 0; n < NELEC; n++) {
      if (Z[n] > 0) kernelJs.push(Math.round(nucPos[n][1]));
    }
    kernelJs.sort((a,b) => a - b);
    const jLo = kernelJs.length > 0 ? kernelJs[0] : N2;
    const jHi = kernelJs.length > 0 ? kernelJs[kernelJs.length - 1] : N2;
    const maxLines = 5;
    const lineRows = [];
    for (let li = 0; li < maxLines; li++) {
      lineRows.push(Math.round(jLo + (jHi - jLo) * li / (maxLines - 1)));
    }
    const nLines = lineRows.length;
    let globalMax = 0;
    for (let li = 0; li < nLines; li++) {
      for (let i = 1; i < NN; i++) {
        const v = sliceData[i * SS + lineRows[li]];
        if (v > globalMax) globalMax = v;
      }
    }
    if (globalMax > 0) {
      const domRGBplot = domainRGB.map(c => [Math.round(c[0]*255), Math.round(c[1]*255), Math.round(c[2]*255)]);
      const lineH = 60;
      const sc = lineH / globalMax;
      for (let li = 0; li < nLines; li++) {
        const row = lineRows[li];
        const rowY = PX * row;
        stroke(255, 255, 255, 40); strokeWeight(1);
        line(0, rowY, CANVAS_SIZE, rowY);
        strokeWeight(2); noFill();
        for (let i = 1; i < NN - 1; i++) {
          const v1 = sliceData[i * SS + row];
          const v2 = sliceData[(i+1) * SS + row];
          if (v1 < 1e-12 && v2 < 1e-12) continue;
          const pk = sliceData[SS2 + i * SS + row];
          const dl = Math.round(pk % 1000);
          const c = domRGBplot[dl % domRGBplot.length] || [180,180,180];
          stroke(c[0], c[1], c[2], 220);
          line(PX * i, rowY - v1 * sc, PX * (i+1), rowY - v2 * sc);
        }
      }
    }

    // W line plot along the slice axis (through nuclei)
    const kBase = 3 * SS * SS;
    const wBase = kBase + SS;
    if (sliceData.length > wBase + NN) {
      // Plot W as cyan line at bottom
      const wPlotY = CANVAS_SIZE - 80;
      const wPlotH = 60;
      stroke(0, 255, 255, 180); strokeWeight(2); noFill();
      for (let i = 1; i < NN - 1; i++) {
        const w1 = sliceData[wBase + i];
        const w2 = sliceData[wBase + i + 1];
        line(PX * i, wPlotY - w1 * wPlotH, PX * (i+1), wPlotY - w2 * wPlotH);
      }
      fill(0, 255, 255); noStroke(); textSize(10);
      text("W (domain weight)", 5, wPlotY - wPlotH - 2);
    }
  }
  } // end skip 2D when _view3D

  // 3D view mode: press '3' to toggle
  if (window._view3D) {
    background(0); // clear to black — no 2D heatmap underneath
    // Draw all atoms as 3D projection with auto-rotation
    const fixedYrot = window.USER_VIEW_YROT !== undefined ? window.USER_VIEW_YROT : Math.PI/2;
    const fixedTilt = window.USER_VIEW_TILT !== undefined ? window.USER_VIEW_TILT : Math.PI/4;
    if (window._view3D_yrot === undefined) window._view3D_yrot = fixedYrot;
    if (window._view3D_tilt === undefined) window._view3D_tilt = fixedTilt;
    const t3d = window._view3D_fixed ? window._view3D_yrot : (frameCount || 0) * (window.USER_ROT_SPEED || 0.02);
    const tilt = window._view3D_fixed ? window._view3D_tilt : 0.3;
    const cosT = Math.cos(t3d), sinT = Math.sin(t3d);
    const cosT2 = Math.cos(tilt), sinT2 = Math.sin(tilt); // tilt
    const cx3 = NN / 2, cy3 = NN / 2, cz3 = NN / 2;
    const scale3 = PX * 0.8;
    const canvMid = CANVAS_SIZE / 2;

    // Element colors
    const elCol = {1:[255,255,255], 2:[255,50,50], 3:[50,50,255], 4:[100,255,100]};

    // Draw bonds: covalent (solid) and H-bonds (colored by distance)
    // Helper to project a point
    function proj3D(n) {
      let ax = nucPos[n][0]-cx3, ay = nucPos[n][1]-cy3, az = nucPos[n][2]-cz3;
      let rx = ax*cosT + az*sinT, rz = -ax*sinT + az*cosT;
      let ry = ay*cosT2 - rz*sinT2;
      return { x: canvMid + rx*scale3, y: canvMid + ry*scale3 };
    }
    const ANG = 0.529177;  // bohr to angstrom
    const _hidden = window.USER_HIDDEN_ATOMS || [];
    for (let a = 0; a < NELEC; a++) {
      if (Z[a] === 0 || _hidden.indexOf(a) >= 0) continue;
      for (let b = a+1; b < NELEC; b++) {
        if (Z[b] === 0 || _hidden.indexOf(b) >= 0) continue;
        const dx = (nucPos[a][0]-nucPos[b][0])*hGrid;
        const dy = (nucPos[a][1]-nucPos[b][1])*hGrid;
        const dz = (nucPos[a][2]-nucPos[b][2])*hGrid;
        const d = Math.sqrt(dx*dx+dy*dy+dz*dz);
        const dAng = d * ANG;
        // Classify bond type
        const isHO = (Z[a] === 1 && Z[b] === 2) || (Z[a] === 2 && Z[b] === 1);
        const isOO = (Z[a] === 2 && Z[b] === 2);
        if (isHO && dAng < 1.3) {
          // Covalent O-H bond — solid white
          stroke(200); strokeWeight(2);
          const pa = proj3D(a), pb = proj3D(b);
          line(pa.x, pa.y, pb.x, pb.y);
        } else if (isHO && dAng >= 1.3 && dAng < 2.8) {
          // H-bond: green < 2.2, yellow 2.2-2.5, red > 2.5
          let r, g, bl;
          if (dAng < 2.2) { r = 0; g = 255; bl = 100; }        // intact — green
          else if (dAng < 2.5) { r = 255; g = 255; bl = 0; }    // stretched — yellow
          else { r = 255; g = 60; bl = 60; }                     // breaking — red
          stroke(r, g, bl, 180); strokeWeight(1);
          const pa = proj3D(a), pb = proj3D(b);
          // Dashed appearance: draw shorter line
          const mx = (pa.x + pb.x) / 2, my = (pa.y + pb.y) / 2;
          line(pa.x, pa.y, mx, my);
          // gap then second half
          const qx = (mx + pb.x) / 2, qy = (my + pb.y) / 2;
          line(qx, qy, pb.x, pb.y);
        } else if (!isHO && !isOO && d < 5.5) {
          // Other covalent bonds (C-N, C-C, N-H etc)
          stroke(80); strokeWeight(1);
          const pa = proj3D(a), pb = proj3D(b);
          line(pa.x, pa.y, pb.x, pb.y);
        }
      }
    }

    // Draw atoms (sorted by depth for proper overlap)
    let atomList = [];
    for (let n = 0; n < NELEC; n++) {
      if (Z[n] === 0 && Z_nuc[n] === 0) continue;  // skip empty
      if (Z[n] > 0 && Z_nuc[n] === 0) continue;  // skip electron-only (no kernel)
      if (_hidden.indexOf(n) >= 0) continue;  // skip hidden atoms
      let ax = nucPos[n][0]-cx3, ay = nucPos[n][1]-cy3, az = nucPos[n][2]-cz3;
      let rx = ax*cosT + az*sinT, rz = -ax*sinT + az*cosT;
      let ry = ay*cosT2 - rz*sinT2, rz2 = ay*sinT2 + rz*cosT2;
      atomList.push({n:n, sx: canvMid + rx*scale3, sy: canvMid + ry*scale3, depth: rz2, z: Z[n]});
    }
    atomList.sort((a,b) => a.depth - b.depth); // back to front

    noStroke();
    for (const at of atomList) {
      const col = at.z === 0 ? [255,80,80] : (elCol[at.z] || [200,200,200]);
      const sz = at.z <= 1 ? 5 : 10;
      fill(col[0], col[1], col[2], 200);
      circle(at.sx, at.sy, sz);
    }

    // Draw highlighted bond pair lines in 3D
    if (window.USER_BOND_PAIRS) {
      const bp = window.USER_BOND_PAIRS;
      const _hid = window.USER_HIDDEN_ATOMS || [];
      function proj3D(n) {
        let ax = nucPos[n][0]-cx3, ay = nucPos[n][1]-cy3, az = nucPos[n][2]-cz3;
        let rx = ax*cosT + az*sinT, rz = -ax*sinT + az*cosT;
        let ry = ay*cosT2 - rz*sinT2;
        return { sx: canvMid + rx*scale3, sy: canvMid + ry*scale3 };
      }
      stroke(255, 255, 0, 200); strokeWeight(2);
      const bondMaxAu = window.USER_BOND_MAX_AU || 4.0;  // hide line when distance > this (au)
      for (let b = 0; b < bp.length; b++) {
        const a1 = bp[b][0], a2 = bp[b][1];
        if (_hid.indexOf(a1) >= 0 || _hid.indexOf(a2) >= 0) continue;
        // Measure actual distance — if bond is broken (distance large), skip the line
        const dx = (nucPos[a1][0] - nucPos[a2][0]) * hGrid;
        const dy = (nucPos[a1][1] - nucPos[a2][1]) * hGrid;
        const dz = (nucPos[a1][2] - nucPos[a2][2]) * hGrid;
        const dAu = Math.sqrt(dx*dx + dy*dy + dz*dz);
        if (dAu > bondMaxAu) continue;
        const p1 = proj3D(a1), p2 = proj3D(a2);
        line(p1.sx, p1.sy, p2.sx, p2.sy);
      }
      noStroke();
    }

    // Ribbon overlay: colored backbone trace through Ca atoms
    const cmd3 = window.USER_CLASSICAL_MD;
    if (cmd3 && cmd3.caIndices && cmd3.caIndices.length > 2) {
      const caIdx = cmd3.caIndices;
      const nCa = caIdx.length;

      // Project all Ca positions
      const caProj = [];
      for (let g = 0; g < nCa; g++) {
        const a = caIdx[g];
        let ax = nucPos[a][0]-cx3, ay = nucPos[a][1]-cy3, az = nucPos[a][2]-cz3;
        let rx = ax*cosT + az*sinT, rz = -ax*sinT + az*cosT;
        let ry = ay*cosT2 - rz*sinT2;
        caProj.push({ x: canvMid + rx*scale3, y: canvMid + ry*scale3 });
      }

      // Classify secondary structure from local geometry
      // Helix: Ca[i-1]-Ca[i]-Ca[i+1] angle < 100° AND Ca[i]-Ca[i+2] distance < 6 Å
      // Sheet: angle > 115° AND Ca[i]-Ca[i+2] distance > 6.5 Å
      const ssType = new Array(nCa).fill(0); // 0=coil, 1=helix, 2=sheet
      const bohrToAng = 0.529177;
      for (let g = 1; g < nCa - 1; g++) {
        const a0 = caIdx[g-1], a1 = caIdx[g], a2 = caIdx[g+1];
        const bax = nucPos[a0][0]-nucPos[a1][0], bay = nucPos[a0][1]-nucPos[a1][1], baz = nucPos[a0][2]-nucPos[a1][2];
        const bcx = nucPos[a2][0]-nucPos[a1][0], bcy = nucPos[a2][1]-nucPos[a1][1], bcz = nucPos[a2][2]-nucPos[a1][2];
        const magBA = Math.sqrt(bax*bax+bay*bay+baz*baz);
        const magBC = Math.sqrt(bcx*bcx+bcy*bcy+bcz*bcz);
        const dot = bax*bcx + bay*bcy + baz*bcz;
        const ang = Math.acos(Math.max(-1, Math.min(1, dot / (magBA * magBC + 1e-10)))) * 180 / Math.PI;

        // Ca[i] to Ca[i+2] distance
        let d_i2 = 999;
        if (g + 1 < nCa) {
          const a3 = caIdx[g+1];
          const dx = (nucPos[a1][0]-nucPos[a3][0]) * hGrid * bohrToAng;
          const dy = (nucPos[a1][1]-nucPos[a3][1]) * hGrid * bohrToAng;
          const dz = (nucPos[a1][2]-nucPos[a3][2]) * hGrid * bohrToAng;
          d_i2 = Math.sqrt(dx*dx+dy*dy+dz*dz);
        }

        if (ang < 105 && d_i2 < 6.0) {
          ssType[g] = 1; // helix
        } else if (ang > 115 && d_i2 > 6.0) {
          ssType[g] = 2; // sheet
        }
      }
      // Extend assignments to neighbors (smooth out isolated assignments)
      for (let pass = 0; pass < 2; pass++) {
        for (let g = 1; g < nCa - 1; g++) {
          if (ssType[g] === 0 && ssType[g-1] === ssType[g+1] && ssType[g-1] !== 0) {
            ssType[g] = ssType[g-1];
          }
        }
      }

      // Draw ribbon
      const ssCol = {0: [150,150,150], 1: [255,50,100], 2: [255,220,50]};
      const ssWidth = {0: 1, 1: 4, 2: 3};
      for (let g = 0; g < nCa - 1; g++) {
        const ss = ssType[g] || ssType[g+1] || 0;
        const col = ssCol[ss];
        stroke(col[0], col[1], col[2], 200);
        strokeWeight(ssWidth[ss]);
        line(caProj[g].x, caProj[g].y, caProj[g+1].x, caProj[g+1].y);
      }

      // Helix: add sine wave ribbon for visual effect
      for (let g = 0; g < nCa - 1; g++) {
        if (ssType[g] === 1 && ssType[g+1] === 1) {
          stroke(255, 80, 150, 120);
          strokeWeight(1);
          const steps = 8;
          for (let s = 0; s < steps; s++) {
            const t1 = s / steps, t2 = (s+1) / steps;
            const x1 = caProj[g].x + (caProj[g+1].x - caProj[g].x) * t1;
            const y1 = caProj[g].y + (caProj[g+1].y - caProj[g].y) * t1 + Math.sin(t1 * Math.PI * 2 + g * 1.7) * 4;
            const x2 = caProj[g].x + (caProj[g+1].x - caProj[g].x) * t2;
            const y2 = caProj[g].y + (caProj[g+1].y - caProj[g].y) * t2 + Math.sin(t2 * Math.PI * 2 + g * 1.7) * 4;
            line(x1, y1, x2, y2);
          }
        }
      }

      // Legend
      noStroke(); textSize(10);
      fill(255, 50, 100); text("\u2588 helix", CANVAS_SIZE - 120, 15);
      fill(255, 220, 50); text("\u2588 sheet", CANVAS_SIZE - 120, 27);
      fill(150, 150, 150); text("\u2588 coil", CANVAS_SIZE - 120, 39);
    }

    // Label
    fill(255, 255, 0);
    noStroke();
    textSize(11);
    text("3D VIEW (3=toggle) H=white C=green N=blue O=red  R=record D=dynamics", 5, 15);

    // Fold angle in 3D view
    if (window.USER_FOLD_ATOMS) {
      const fa = window.USER_FOLD_ATOMS;
      const a3 = nucPos[fa[0]], b3 = nucPos[fa[1]], c3 = nucPos[fa[2]];
      const ba3 = [a3[0]-b3[0], a3[1]-b3[1], a3[2]-b3[2]];
      const bc3 = [c3[0]-b3[0], c3[1]-b3[1], c3[2]-b3[2]];
      const dot3 = ba3[0]*bc3[0] + ba3[1]*bc3[1] + ba3[2]*bc3[2];
      const magBA3 = Math.sqrt(ba3[0]*ba3[0]+ba3[1]*ba3[1]+ba3[2]*ba3[2]);
      const magBC3 = Math.sqrt(bc3[0]*bc3[0]+bc3[1]*bc3[1]+bc3[2]*bc3[2]);
      const foldAngle3 = Math.acos(Math.max(-1, Math.min(1, dot3 / (magBA3 * magBC3)))) * 180 / Math.PI;
      const foldPct3 = (1 - foldAngle3 / 180) * 100;
      fill(255, 255, 0);
      text("Fold: " + foldAngle3.toFixed(1) + "\u00B0 (" + foldPct3.toFixed(0) + "%)  [180\u00B0=open, 0\u00B0=folded]", 5, 30);
    }

    // Bond distances in 3D view
    if (window.USER_BOND_PAIRS) {
      fill(200, 200, 255);
      const bp3 = window.USER_BOND_PAIRS;
      const bohrToAng3 = 0.529177;
      let bondY3 = 45;
      for (let b = 0; b < bp3.length; b++) {
        const a1 = bp3[b][0], a2 = bp3[b][1], label = bp3[b][2] || "";
        const dx = nucPos[a1][0] - nucPos[a2][0];
        const dy = nucPos[a1][1] - nucPos[a2][1];
        const dz = nucPos[a1][2] - nucPos[a2][2];
        const dAu = Math.sqrt(dx*dx + dy*dy + dz*dz) * hGrid;
        const dAng = dAu * bohrToAng3;
        text(label + " " + dAng.toFixed(2) + "\u00C5", 5, bondY3);
        bondY3 += 13;
      }
    }

    // Helix progress in 3D view
    if (window.USER_HELIX_PROGRESS) {
      fill(0, 255, 200);
      text("Helix: " + (window._helixPct || 0).toFixed(1) + "%  R=" + ((window._helixRadius || 0)*0.529177).toFixed(2) +
        "\u00C5  rise=" + ((window._helixRise || 0)*0.529177).toFixed(2) + "\u00C5  H-bonds=" + (window._helixHbonds || 0), 5, CANVAS_SIZE - 20);
    }

    // Energy display
    fill(255, 255, 255);
    text("E=" + E.toFixed(2) + " Ha  T=" + E_T.toFixed(2) + "  V_eK=" + E_eK.toFixed(2) + "  V_ee=" + E_ee.toFixed(2) + "  V_KK=" + E_KK.toFixed(2), 5, CANVAS_SIZE - 20);

    // Binding energy display (if atom reference energies available)
    const atomERef = window._atomEnergies || (function() {
      try {
        const s = localStorage.getItem('realqm_atom_energies');
        if (s) { window._atomEnergies = JSON.parse(s); return window._atomEnergies; }
      } catch(e) {}
      return null;
    })();
    if (atomERef) {
      let Edis = 0, countMissing = 0;
      for (let n = 0; n < NELEC; n++) {
        if (Z_orig[n] === 0 && (Z_nuc[n] === undefined || Z_nuc[n] === 0)) continue;
        const el = (atomLabels[n] || '').charAt(0).toUpperCase();
        if (atomERef[el] !== undefined) {
          Edis += atomERef[el];
        } else {
          countMissing++;
        }
      }
      if (countMissing === 0) {
        const eBind = E - Edis;
        fill(100, 255, 200);
        text("E_disassembled=" + Edis.toFixed(2) + "  E_bind=" + eBind.toFixed(3) + " Ha = " + (eBind*27.211).toFixed(2) + " eV",
             5, CANVAS_SIZE - 50);
      }
    }
    if (langevinKT > 0 && window._thermoT !== undefined) {
      text("T=" + window._thermoT.toFixed(0) + " K (target " + (langevinKT*315775).toFixed(0) + " K)  kick=" + (window.USER_THERMAL_KICK || 0), 5, CANVAS_SIZE - 35);
    }

    // Dynamics status
    fill(dynamicsEnabled ? [0,255,255] : [150,150,150]);
    text("Dynamics: " + (dynamicsEnabled ? "ON" : "OFF") + "  nucStep=" + nucStepCount + "  Force=" + forceScale.toFixed(1) + "x", 5, CANVAS_SIZE - 5);

    // Rg display (above dynamics line)
    if (window._compactRg !== undefined) {
      fill(255, 200, 50);
      text("Rg=" + (window._compactRg * 0.529177).toFixed(1) + "\u00C5", 5, CANVAS_SIZE - 35);
    }

    return; // skip normal 2D draw
  }

  // Draw nuclear positions with force arrows
  fill(255); stroke(255); strokeWeight(1);
  for (let n = 0; n < NELEC; n++) {
    // Draw bare protons (Z=0, Z_nuc>0) as white dots like other nuclei
    if (Z[n] === 0 && Z_nuc[n] > 0) {
      const nx = nucPos[n][0] * PX, ny = nucPos[n][1] * PX;
      fill(255); noStroke();
      circle(nx, ny, 6);
      stroke(255); strokeWeight(1);
    }
    if (Z_nuc[n] > 0) {  // draw all kernels (including bare nuclei)
      const nx = nucPos[n][0] * PX, ny = nucPos[n][1] * PX;
      circle(nx, ny, 6);
      const _frozen = window.USER_FROZEN_ATOMS || [];
      if (nucForceTotal[n] && _frozen.indexOf(n) < 0) {
        const arrowScale = (window.USER_ARROW_SCALE || 250) * forceScale;
        const ft = nucForceTotal[n];
        const fx = ft[0], fy = ft[1];
        if (fx*fx + fy*fy > 1e-16) {
          stroke(255, 255, 0); strokeWeight(2);
          line(nx, ny, nx + fx * arrowScale, ny + fy * arrowScale);
        }
        stroke(255); strokeWeight(1);
      }
    }
  }


  // Screen boundary
  noFill(); stroke(100); strokeWeight(1);
  rect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  noStroke();

  fill(255);
  const labels = atomLabels.map((el, i) => [el, Z_orig[i]]).filter(x => x[1] > 0).map(x => x[0] + "(Z=" + x[1] + ")").join(" ");
  const pLabel = phase === 0 ? "running" : "DONE";
  text("Molecule: " + labels + " | " + screenAu + " au | " + pLabel + " | " + NN + "^3", 5, 20);
  text("step " + tStep + " (" + phaseSteps + "/" + TOTAL_STEPS + ")  E=" + (E_T + E_eK + E_ee + E_KK).toFixed(6) + "  E_min=" + E_min.toFixed(6), 5, 35);
  if (lastMs > 0) text((lastMs / STEPS_PER_FRAME).toFixed(1) + "ms/step", CANVAS_SIZE - 120, 35);

  fill(200);
  text("T=" + E_T.toFixed(4) + " V_eK=" + E_eK.toFixed(4) + " V_ee=" + E_ee.toFixed(4) + " V_KK=" + E_KK.toFixed(4) + "  Dipole=" + dipole_D.toFixed(3) + " D  E_bind=" + E_bind.toFixed(4) + " Ha", 5, 50);
  fill(vcycleEnabled ? [0,255,0] : [255,100,0]);
  const solverName = useLOBPCG ? "LOBPCG" : useCheb ? "Chebyshev" : "ITP";
  text("V-cycle: " + (vcycleEnabled ? "ON" : "OFF") + " (" + vcycleCount + ")  Solver: " + solverName + " [L=LOBPCG C=Cheb]", 5, 65);
  if (window._hohStats) {
    fill(180, 255, 180); textSize(13);
    text(window._hohStats, 5, 95);
  }
  if (window._hbStats) {
    fill(255, 200, 100); textSize(13);
    text(window._hbStats, 5, 140);
  }
  if (window._ooDist) {
    fill(255, 200, 100); textSize(13);
    text(window._ooDist, 5, 320);
  }
  if (window.USER_SHEAR_RATE) {
    fill(0, 230, 255); textSize(14);
    text("W_total = " + (window._shearWork || 0).toExponential(3) + " Ha  (shear rate=" + window.USER_SHEAR_RATE.toExponential(2) + ")", 5, 80);
  }

  // Fold angle measurement
  if (window.USER_FOLD_ATOMS) {
    const fa = window.USER_FOLD_ATOMS;  // [strand1_idx, hinge_idx, strand2_idx]
    const a = nucPos[fa[0]], b = nucPos[fa[1]], c = nucPos[fa[2]];
    const ba = [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
    const bc = [c[0]-b[0], c[1]-b[1], c[2]-b[2]];
    const dot = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2];
    const magBA = Math.sqrt(ba[0]*ba[0]+ba[1]*ba[1]+ba[2]*ba[2]);
    const magBC = Math.sqrt(bc[0]*bc[0]+bc[1]*bc[1]+bc[2]*bc[2]);
    const foldAngle = Math.acos(Math.max(-1, Math.min(1, dot / (magBA * magBC)))) * 180 / Math.PI;
    const foldPct = (1 - foldAngle / 180) * 100;  // 0%=straight(180°), 100%=folded(0°)
    fill(255, 255, 0);
    text("Fold: " + foldAngle.toFixed(1) + "\u00B0  (" + foldPct.toFixed(1) + "%)  [180\u00B0=open, 0\u00B0=folded]", 5, 80);
    window._foldAngle = foldAngle;
    window._foldPct = foldPct;
  }

  // Helix progress measurement
  if (window.USER_HELIX_PROGRESS) {
    const hp = window.USER_HELIX_PROGRESS;
    const caIdx = hp.caIndices;        // Ca atom indices
    const oIdx = hp.oIndices;          // carbonyl O indices
    const hIdx = hp.hIndices;          // amide H indices
    const idealRadius = hp.idealRadius || 4.35; // 2.3 A in au
    const idealRise = hp.idealRise || 2.83;     // 1.5 A in au
    const hbondCutoff = hp.hbondCutoff || 4.72; // 2.5 A in au
    const bohrToAng = 0.529177;

    // 1. Average Ca radius from helix axis
    // Convert all Ca positions to au first
    let caAu = [];
    for (let i = 0; i < caIdx.length; i++) {
      caAu.push([
        nucPos[caIdx[i]][0] * hGrid,
        nucPos[caIdx[i]][1] * hGrid,
        nucPos[caIdx[i]][2] * hGrid
      ]);
    }
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < caAu.length; i++) {
      cx += caAu[i][0]; cy += caAu[i][1]; cz += caAu[i][2];
    }
    cx /= caAu.length; cy /= caAu.length; cz /= caAu.length;
    // Helix axis ~ line from first Ca to last Ca (in au)
    let ax = caAu[caAu.length-1][0] - caAu[0][0];
    let ay = caAu[caAu.length-1][1] - caAu[0][1];
    let az = caAu[caAu.length-1][2] - caAu[0][2];
    let aLen = Math.sqrt(ax*ax + ay*ay + az*az) + 1e-10;
    ax /= aLen; ay /= aLen; az /= aLen;
    // Average perpendicular distance from axis
    let avgRadius = 0;
    for (let i = 0; i < caAu.length; i++) {
      let dx = caAu[i][0] - cx;
      let dy = caAu[i][1] - cy;
      let dz = caAu[i][2] - cz;
      let proj = dx*ax + dy*ay + dz*az;
      let perpSq = dx*dx + dy*dy + dz*dz - proj*proj;
      avgRadius += Math.sqrt(Math.max(0, perpSq));
    }
    avgRadius /= caAu.length;
    let radiusScore = Math.min(1, avgRadius / idealRadius);

    // 2. Average rise per residue (end-to-end along axis / nRes)
    let risePerRes = aLen / (caAu.length - 1);
    // Score: 1.0 when rise = ideal, decreasing as it deviates
    let riseScore = 1 - Math.min(1, Math.abs(risePerRes - idealRise) / idealRise);

    // 3. i→i+4 H-bond formation (O[i]···H[i+4] distance)
    let nHbonds = 0, totalHbonds = 0, avgHbondDist = 0;
    if (oIdx && hIdx) {
      for (let i = 0; i + 4 < oIdx.length && i + 4 < hIdx.length; i++) {
        let dx = (nucPos[oIdx[i]][0] - nucPos[hIdx[i+4]][0]) * hGrid;
        let dy = (nucPos[oIdx[i]][1] - nucPos[hIdx[i+4]][1]) * hGrid;
        let dz = (nucPos[oIdx[i]][2] - nucPos[hIdx[i+4]][2]) * hGrid;
        let d = Math.sqrt(dx*dx + dy*dy + dz*dz);
        avgHbondDist += d;
        if (d < hbondCutoff) nHbonds++;
        totalHbonds++;
      }
      avgHbondDist /= Math.max(1, totalHbonds);
    }
    let hbondScore = totalHbonds > 0 ? nHbonds / totalHbonds : 0;

    // Composite score (weighted)
    let helixPct = (radiusScore * 0.3 + riseScore * 0.3 + hbondScore * 0.4) * 100;

    fill(0, 255, 200);
    text("Helix: " + helixPct.toFixed(1) + "%  R=" + (avgRadius*bohrToAng).toFixed(2) +
      "\u00C5/" + (idealRadius*bohrToAng).toFixed(1) + "  rise=" + (risePerRes*bohrToAng).toFixed(2) +
      "\u00C5/" + (idealRise*bohrToAng).toFixed(1) + "  H-bonds=" + nHbonds + "/" + totalHbonds +
      " (avg " + (avgHbondDist*bohrToAng).toFixed(2) + "\u00C5)", 5, 80);
    window._helixPct = helixPct;
    window._helixRadius = avgRadius;
    window._helixRise = risePerRes;
    window._helixHbonds = nHbonds;
  }

  // Rg display
  if (window._compactRg !== undefined) {
    fill(255, 200, 50);
    text("Rg=" + (window._compactRg * 0.529177).toFixed(1) + "\u00C5 (" + window._compactRg.toFixed(0) + "au)  target=" + (window._compactTargetRg * 0.529177).toFixed(1) + "\u00C5", 5, 95);
  }

  // Dynamics status
  fill(dynamicsEnabled ? [0,255,255] : [150,150,150]);
  text("Dynamics: " + (dynamicsEnabled ? "ON" : "OFF") + " (nucStep=" + nucStepCount + ")  Force=" + forceScale.toFixed(1) + "x  [D toggle, +/- force]", 5, 110);
  fill(255, 255, 0); text("yellow=net force", 5, 125);


  // Display net fold forces on screen
  if (window.USER_FOLD_ATOMS && window._foldNetUnfold !== undefined) {
    fill(window._foldNetUnfold > 0 ? [255, 100, 100] : [100, 255, 100]);
    text("Net fold force: " + window._foldNetUnfold.toExponential(2) +
      (window._foldNetUnfold > 0 ? " UNFOLDING" : " FOLDING") +
      "  s1=" + window._foldS1proj.toExponential(2) +
      " s2=" + window._foldS2proj.toExponential(2), 5, 155);
  }

  // r_c values
  fill(200, 180, 255);
  var rcInfo = atomLabels.map((el, i) => [el, Z_orig[i], r_cut[i]]).filter(x => x[1] > 0);
  var rcSeen = {};
  rcInfo.forEach(x => { rcSeen[x[0]] = x[2]; });
  text("r_c: " + Object.keys(rcSeen).map(k => k + "=" + rcSeen[k]).join(" "), 5, 170);

  // Bond length display (after r_c to avoid overlap)
  if (window.USER_BOND_PAIRS) {
    fill(200, 200, 255);
    const bp = window.USER_BOND_PAIRS;
    const bohrToAng = 0.529177;
    var bondY = 215;
    let bondStr = "Calc:  ";
    var ref = window.USER_REFERENCE;
    var refStr = (ref && ref.bonds) ? "Ref:   " : null;
    for (let b = 0; b < bp.length; b++) {
      const a1 = bp[b][0], a2 = bp[b][1], label = bp[b][2] || "";
      const dx = nucPos[a1][0] - nucPos[a2][0];
      const dy = nucPos[a1][1] - nucPos[a2][1];
      const dz = nucPos[a1][2] - nucPos[a2][2];
      const dAu = Math.sqrt(dx*dx + dy*dy + dz*dz) * hGrid;
      const dAng = dAu * bohrToAng;
      bondStr += label + dAng.toFixed(2) + "\u00C5 ";
      if (refStr !== null) { var rv = ref.bonds[label.replace(":", "")]; refStr += label + (rv || "?") + "\u00C5 "; }
      if (b === 6) {
        text(bondStr, 5, bondY); bondY += 15; bondStr = "      ";
        if (refStr !== null) { fill(150, 255, 150); text(refStr, 5, bondY); bondY += 15; refStr = "      "; fill(200, 200, 255); }
      }
    }
    if (bondStr.trim()) { text(bondStr, 5, bondY); bondY += 15; }
    if (refStr !== null && refStr.trim()) { fill(150, 255, 150); text(refStr, 5, bondY); bondY += 15; }

    // Show reference energy and binding
    if (window.USER_REFERENCE) {
      fill(150, 255, 150);
      if (window.USER_REFERENCE.energy) {
        text("Ref: " + window.USER_REFERENCE.energy, 5, bondY); bondY += 15;
      }
      if (window.USER_REFERENCE.binding) {
        text("Binding: " + window.USER_REFERENCE.binding, 5, bondY);
      }
    }
  }

  // Diagnostics
  fill(255, 255, 0);
  text("maxDens=" + (window._diagMaxD || 0).toExponential(2) +
    " pos=" + (window._diagPos || 0) + " nan=" + (window._diagNan || 0) +
    " neg=" + (window._diagNeg || 0), 5, 195);
  if (window._initDebug) {
    fill(0, 255, 255);
    text(window._initDebug, 300, 195);
  }
  if (window._gpuValErr) {
    fill(255, 0, 0);
    var errLines = (window._gpuValErr).match(/.{1,80}/g) || [window._gpuValErr];
    for (var ei = 0; ei < Math.min(errLines.length, 5); ei++) {
      text(errLines[ei], 5, 128 + ei * 13);
    }
  }

  // Slice position
  fill(180, 180, 255);
  text("Slice k=" + sliceK + "/" + NN + "  [Up/Down to scroll]", 5, 185);

  // Capture video frame after canvas is fully rendered with updated atom positions
  if (window._nucUpdated && window._recStreamTrack) {
    try { window._recStreamTrack.requestFrame(); } catch(e) {}
    window._nucUpdated = false;
  }
}

// Auto-video: set window.USER_AUTO_VIDEO = { steps: 5000, interval: 200, fps: 15, filename: 'fold.webm' }
// Captures snapshots during dynamics, stitches video when step limit reached
(function() {
  var av = window.USER_AUTO_VIDEO;
  if (!av) return;
  var interval = av.interval || 200;
  var fps = av.fps || 15;
  var maxSteps = av.steps || 5000;
  var filename = av.filename || 'fold.webm';
  var frames = [];
  var done = false;
  var dynamicsStartStep = -1;

  setInterval(function() {
    if (done) return;
    var canvas = document.querySelector('canvas');
    if (!canvas || typeof phaseSteps === 'undefined' || typeof dynamicsEnabled === 'undefined') return;
    if (!dynamicsEnabled) return;
    if (dynamicsStartStep < 0) {
      dynamicsStartStep = phaseSteps;
      console.log('AUTO-VIDEO: recording started at step ' + phaseSteps);
    }
    var dynSteps = phaseSteps - dynamicsStartStep;
    if (dynSteps > 0 && dynSteps % interval === 0) {
      canvas.toBlob(function(blob) {
        if (blob) frames.push(blob);
      }, 'image/webp', 0.85);
    }
    if (dynSteps >= maxSteps && frames.length > 0) {
      done = true;
      console.log('AUTO-VIDEO: stitching ' + frames.length + ' frames...');
      var w = canvas.width, h = canvas.height;
      var off = document.createElement('canvas');
      off.width = w; off.height = h;
      var ctx = off.getContext('2d');
      var stream = off.captureStream(0);
      var track = stream.getVideoTracks()[0];
      var mime = MediaRecorder.isTypeSupported('video/webm; codecs=vp9')
        ? 'video/webm; codecs=vp9' : 'video/webm';
      var rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 5000000 });
      var chunks = [];
      rec.ondataavailable = function(ev) { if (ev.data.size > 0) chunks.push(ev.data); };
      rec.onstop = function() {
        var blob = new Blob(chunks, { type: mime });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url; a.download = filename; a.click();
        URL.revokeObjectURL(url);
        console.log('AUTO-VIDEO: saved ' + filename + ' (' + frames.length + ' frames)');
      };
      rec.start();
      var idx = 0;
      var frameDur = 1000 / fps;
      (function drawNext() {
        if (idx >= frames.length) { rec.stop(); return; }
        var img = new Image();
        img.onload = function() {
          ctx.drawImage(img, 0, 0, w, h);
          if (track.requestFrame) track.requestFrame();
          idx++;
          setTimeout(drawNext, frameDur);
        };
        img.src = URL.createObjectURL(frames[idx]);
      })();
    }
  }, 100);
})();

function keyPressed() {
  if (key === '3') {
    window._view3D = !window._view3D;
    console.log("3D view " + (window._view3D ? "ON" : "OFF"));
  }
  if (key === 'f' || key === 'F') {
    window._view3D_fixed = !window._view3D_fixed;
    console.log("3D rotation " + (window._view3D_fixed ? "FIXED" : "rotating"));
  }
  if (key === 'v' || key === 'V') {
    vcycleEnabled = !vcycleEnabled;
    vcycleCount = 0;
    console.log("V-cycle " + (vcycleEnabled ? "ENABLED" : "DISABLED"));
  }
  if (key === 'r' || key === 'R') {
    if (window._toggleRecording) window._toggleRecording();
  }
  if (key === 'd' || key === 'D') {
    dynamicsEnabled = !dynamicsEnabled;
    console.log("Dynamics " + (dynamicsEnabled ? "ENABLED" : "DISABLED"));
  }
  if (key === 'l' || key === 'L') {
    useLOBPCG = !useLOBPCG;
    if (useLOBPCG) useCheb = false;
    console.log("LOBPCG " + (useLOBPCG ? "ENABLED" : "DISABLED"));
  }
  if (key === 'c' || key === 'C') {
    useCheb = !useCheb;
    if (useCheb) useLOBPCG = false;
    console.log("Chebyshev " + (useCheb ? "ENABLED" : "DISABLED"));
    if (useCheb) chebOmega = 1.0;  // reset recurrence
  }
  if (key === '+' || key === '=') {
    forceScale = Math.min(5.0, forceScale + 0.1);
    var sl = document.getElementById('forceSlider');
    if (sl) sl.value = Math.round(forceScale * 100);
    var el = document.getElementById('forceScaleVal');
    if (el) el.textContent = ' ' + forceScale.toFixed(1) + 'x';
  }
  if (key === '-' || key === '_') {
    forceScale = Math.max(0.0, forceScale - 0.1);
    var sl = document.getElementById('forceSlider');
    if (sl) sl.value = Math.round(forceScale * 100);
    var el = document.getElementById('forceScaleVal');
    if (el) el.textContent = ' ' + forceScale.toFixed(1) + 'x';
  }
  // In 3D fixed mode, arrow keys rotate the view; otherwise scroll slice (UP/DOWN only)
  if (window._view3D && window._view3D_fixed) {
    const rotStep = Math.PI / 36;  // 5 degrees
    if (keyCode === LEFT_ARROW) {
      window._view3D_yrot = (window._view3D_yrot || Math.PI/2) - rotStep;
    }
    if (keyCode === RIGHT_ARROW) {
      window._view3D_yrot = (window._view3D_yrot || Math.PI/2) + rotStep;
    }
    if (keyCode === UP_ARROW) {
      window._view3D_tilt = (window._view3D_tilt || Math.PI/4) - rotStep;
    }
    if (keyCode === DOWN_ARROW) {
      window._view3D_tilt = (window._view3D_tilt || Math.PI/4) + rotStep;
    }
  } else {
    const scrollStep = Math.max(1, Math.round(NN / 60));
    if (keyCode === UP_ARROW) {
      sliceK = Math.min(NN - 1, sliceK + scrollStep);
      updateParamsBuf();
    }
    if (keyCode === DOWN_ARROW) {
      sliceK = Math.max(1, sliceK - scrollStep);
      updateParamsBuf();
    }
  }
}
