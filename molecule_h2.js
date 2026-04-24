// Molecule Quantum Simulation — WebGPU Compute Shaders + Nuclear Dynamics
// Up to 10 atoms placed interactively, 3D geometry
const NN = window.USER_NN || 200;
const S = NN + 1;
const S2 = S * S;
const S3 = S * S * S;
const N2 = Math.round(NN / 2);
const MAX_ATOMS = 1024;
const _uz = window.USER_Z || [2, 3, 1, 0, 0];
while (_uz.length < MAX_ATOMS) _uz.push(0);
const NELEC = _uz.filter(z => z > 0).length || 3;
const N_ELECTRONS = _uz.reduce((s, z) => s + z, 0);  // total valence electrons
const NRED_E = 6;  // Energy reduce: T + V_eK + V_ee + dipole(x,y,z)
const r_cut = window.USER_RC || [0, 0, 0, 0, 0];
while (r_cut.length < MAX_ATOMS) r_cut.push(0);
let R_out = 0.5;   // au, unused legacy
let curvReg = 0.15;  // curvature regularization for free boundary
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
// True nuclear charges (Z) for V_KK — distinct from Z_eff used for electron energies
const _elZ = { H:1, He:2, Li:3, Be:4, B:5, C:6, N:7, O:8, F:9, Ne:10, Na:11, S:16 };
const Z_nuc = _atoms.map(a => _elZ[a.el] || a.Z);
let nucPos = _atoms.map(a => [a.i, a.j, a.k !== undefined ? a.k : N2]);
const molNucPos = nucPos.map(p => [...p]);

let E_min = Infinity;
let screenAu = window.USER_SCREEN || 10;
let hGrid = screenAu / NN, h2v = hGrid * hGrid, h3v = hGrid * hGrid * hGrid;
const dv = NELEC > 100 ? 0.03 : 0.12;  // smaller timestep for large systems
let dtv = dv * h2v, half_dv = 0.5 * dv;
const PX = 700 / NN;
const INTERIOR = (NN - 1) * (NN - 1) * (NN - 1);
const R_SING = 2 * hGrid;  // exclude 2 grid spacings from nucleus

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

const STEPS_PER_FRAME = NELEC <= 5 ? 500 : NELEC <= 15 ? 100 : NELEC <= 30 ? 50 : NELEC <= 100 ? 5 : 2;
const W_STEPS_PER_FRAME = Math.max(1, STEPS_PER_FRAME);  // match U steps per frame
const BOUNDARY_INTERVAL = 20;
const NORM_INTERVAL = 20;
const POISSON_INTERVAL = 50;
const SIC_INTERVAL = NELEC <= 15 ? 1 : NELEC <= 30 ? 5 : 999999;  // SIC in dynamics to remove self-interaction from wavefunction evolution
const SIC_JACOBI = NELEC <= 15 ? 50 : 10;

// === Nuclear dynamics state ===
const N_MOVE = 200;         // electronic steps between nuclear moves
const DT_NUC = 0.8;         // au (~0.02 fs)
const NUC_SUBSTEPS = 1;     // single step (forces recomputed each move)
const DAMPING = 0.98;       // light damping
const MAX_VEL = 0.1;        // au/au_time
let forceScale = 1.0;       // adjustable via slider/keys
let boundarySpeed = 0.5;    // dt_w for free boundary evolution
let nucVel = Array.from({length: MAX_ATOMS}, () => [0, 0, 0]);
let nucForce = Array.from({length: MAX_ATOMS}, () => [0, 0, 0]);
let nucStepCount = 0, dynamicsEnabled = false;
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

const ATOM_STRIDE = 8; // 8 f32s per atom (posI, posJ, posK, Z, rc, pad, pad, pad)
const ATOM_BUF_BYTES = MAX_ATOMS * ATOM_STRIDE * 4;
const atomStructWGSL = `
struct Atom {
  posI: u32, posJ: u32, posK: u32, Z: f32,
  rc: f32, _p0: f32, _p1: f32, _p2: f32,
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
  if (rc <= 0.0) { return false; }
  return distToAtom(ci, cj, ck, lbl) < rc;
}

// Check if inside analytical region for bare atoms (H: r_c=0, use R_SING sphere)
fn isInsideAnalytical(ci: u32, cj: u32, ck: u32, lbl: u32) -> bool {
  let _keep = atoms[lbl].rc;  // keep atoms binding alive
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

  // Inside r_c (pseudopotential): U = 0, Neumann BC at r_c surface
  if (isInsideRc(i, j, k, myL)) {
    Uo[id] = 0.0;
    return;
  }

  // Inside analytical region for bare atoms (H): set U from analytical shape
  // scaled by average U from outside the R_SING boundary
  if (isInsideAnalytical(i, j, k, myL)) {
    let Za = atoms[myL].Z;
    let r = distToAtom(i, j, k, myL);
    let ai = atoms[myL].posI; let aj = atoms[myL].posJ; let ak = atoms[myL].posK;
    // Average U on the 6 face-neighbor shell just outside R_SING
    let nOff = u32(ceil(${R_SING} / p.h));  // grid points to reach outside R_SING
    let avgOut = (Ui[(ai + nOff) * p.S2 + aj * p.S + ak] +
                  Ui[(ai - nOff) * p.S2 + aj * p.S + ak] +
                  Ui[ai * p.S2 + (aj + nOff) * p.S + ak] +
                  Ui[ai * p.S2 + (aj - nOff) * p.S + ak] +
                  Ui[ai * p.S2 + aj * p.S + (ak + nOff)] +
                  Ui[ai * p.S2 + aj * p.S + (ak - nOff)]) / 6.0;
    let rOut = f32(nOff) * p.h;
    Uo[id] = avgOut * exp(-Za * r) / exp(-Za * rOut);
    return;
  }

  let uc = Ui[id];

  // Outside: Neumann BC at both r_c and analytical boundaries
  let l_ip = label[id + p.S2]; let excl_ip = isInsideRc(i+1u, j, k, l_ip) || isInsideAnalytical(i+1u, j, k, myL);
  let l_im = label[id - p.S2]; let excl_im = isInsideRc(i-1u, j, k, l_im) || isInsideAnalytical(i-1u, j, k, myL);
  let l_jp = label[id + p.S];  let excl_jp = isInsideRc(i, j+1u, k, l_jp) || isInsideAnalytical(i, j+1u, k, myL);
  let l_jm = label[id - p.S];  let excl_jm = isInsideRc(i, j-1u, k, l_jm) || isInsideAnalytical(i, j-1u, k, myL);
  let l_kp = label[id + 1u];   let excl_kp = isInsideRc(i, j, k+1u, l_kp) || isInsideAnalytical(i, j, k+1u, myL);
  let l_km = label[id - 1u];   let excl_km = isInsideRc(i, j, k-1u, l_km) || isInsideAnalytical(i, j, k-1u, myL);

  // Robin BC at Voronoi inter-orbital boundary: ∂u/∂n = -ROBIN_K · u.
  // Finite-difference fallback value for cross-boundary neighbor: uc · (1 - K·h)
  // (reduces to Neumann when ROBIN_K=0, Dirichlet when ROBIN_K·h → 1).
  let robinFac = 1.0 - ${window.ROBIN_K !== undefined ? window.ROBIN_K.toFixed(4) : '1.0'} * p.h;
  let ucRobin = uc * robinFac;
  let u_ip = select(ucRobin, Ui[id + p.S2], l_ip == myL && !excl_ip);
  let u_im = select(ucRobin, Ui[id - p.S2], l_im == myL && !excl_im);
  let u_jp = select(ucRobin, Ui[id + p.S],  l_jp == myL && !excl_jp);
  let u_jm = select(ucRobin, Ui[id - p.S],  l_jm == myL && !excl_jm);
  let u_kp = select(ucRobin, Ui[id + 1u],   l_kp == myL && !excl_kp);
  let u_km = select(ucRobin, Ui[id - 1u],   l_km == myL && !excl_km);

  let lap = u_ip + u_im + u_jp + u_jm + u_kp + u_km - 6.0 * uc;

  Uo[id] = uc + p.half_d * lap + p.dt * (K[id] - 2.0 * Pi[id]) * uc;
}
`;

// Chebyshev semi-iterative acceleration of ITP
// ψ_{n+1} = ω * (standard ITP step) + (1-ω) * ψ_{n-1}
// Same stencil as updateU, but with momentum term from previous iterate
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
  if (atoms[lbl].rc > 0.0) { return false; }
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

  if (isInsideRc(i, j, k, myL)) {
    Uo[id] = 0.0;
    return;
  }

  if (isInsideAnalytical(i, j, k, myL)) {
    let Za = atoms[myL].Z;
    let r = distToAtom(i, j, k, myL);
    let ai = atoms[myL].posI; let aj = atoms[myL].posJ; let ak = atoms[myL].posK;
    let nOff = u32(ceil(${R_SING} / p.h));
    let avgOut = (Ui[(ai + nOff) * p.S2 + aj * p.S + ak] +
                  Ui[(ai - nOff) * p.S2 + aj * p.S + ak] +
                  Ui[ai * p.S2 + (aj + nOff) * p.S + ak] +
                  Ui[ai * p.S2 + (aj - nOff) * p.S + ak] +
                  Ui[ai * p.S2 + aj * p.S + (ak + nOff)] +
                  Ui[ai * p.S2 + aj * p.S + (ak - nOff)]) / 6.0;
    let rOut = f32(nOff) * p.h;
    Uo[id] = avgOut * exp(-Za * r) / exp(-Za * rOut);
    return;
  }

  let uc = Ui[id];

  let l_ip = label[id + p.S2]; let excl_ip = isInsideRc(i+1u, j, k, l_ip) || isInsideAnalytical(i+1u, j, k, myL);
  let l_im = label[id - p.S2]; let excl_im = isInsideRc(i-1u, j, k, l_im) || isInsideAnalytical(i-1u, j, k, myL);
  let l_jp = label[id + p.S];  let excl_jp = isInsideRc(i, j+1u, k, l_jp) || isInsideAnalytical(i, j+1u, k, myL);
  let l_jm = label[id - p.S];  let excl_jm = isInsideRc(i, j-1u, k, l_jm) || isInsideAnalytical(i, j-1u, k, myL);
  let l_kp = label[id + 1u];   let excl_kp = isInsideRc(i, j, k+1u, l_kp) || isInsideAnalytical(i, j, k+1u, myL);
  let l_km = label[id - 1u];   let excl_km = isInsideRc(i, j, k-1u, l_km) || isInsideAnalytical(i, j, k-1u, myL);

  // Robin BC at Voronoi inter-orbital boundary (same convention as updateU)
  let robinFac = 1.0 - ${window.ROBIN_K !== undefined ? window.ROBIN_K.toFixed(4) : '1.0'} * p.h;
  let ucRobin = uc * robinFac;
  let u_ip = select(ucRobin, Ui[id + p.S2], l_ip == myL && !excl_ip);
  let u_im = select(ucRobin, Ui[id - p.S2], l_im == myL && !excl_im);
  let u_jp = select(ucRobin, Ui[id + p.S],  l_jp == myL && !excl_jp);
  let u_jm = select(ucRobin, Ui[id - p.S],  l_jm == myL && !excl_jm);
  let u_kp = select(ucRobin, Ui[id + 1u],   l_kp == myL && !excl_kp);
  let u_km = select(ucRobin, Ui[id - 1u],   l_km == myL && !excl_km);

  let lap = u_ip + u_im + u_jp + u_jm + u_kp + u_km - 6.0 * uc;

  // Standard ITP step: ψ + dt*(½∇²ψ + V·ψ)
  let itpStep = uc + p.half_d * lap + p.dt * (K[id] - 2.0 * Pi[id]) * uc;

  // Chebyshev momentum: ψ_{n+1} = ω·(ITP step) + (1-ω)·ψ_{n-1}
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

// Subtract per-domain self-potential from Pother at points in that domain.
// For orbitals with occupation n=Zeff>1 we only remove the 1/n "fake" self-interaction
// piece; the (n-1)/n piece is kept as the real intra-orbital pair repulsion.
// Zeff is packed as the float bits in slot 1 of the DomIdx uniform.
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
  // Don't skip cells — u=0 inside r_c naturally contributes 0 to V terms
  return false;
}
fn isInsideRcEnergy(ci: u32, cj: u32, ck: u32, lbl: u32) -> bool {
  // For gradient zeroing at r_c boundary (Neumann BC in energy, matching ITP)
  let rc = atoms[lbl].rc;
  if (rc <= 0.0) { return false; }
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

    // Skip points inside exclusion sphere max(r_c, R_SING) — replaced by analytical correction
    if (isInsideExcl(i, j, k, myL)) { cell = cell + stride; continue; }

    // Gradients: zero if either endpoint (current or neighbor) is inside r_c.
    // U=0 inside r_c is a boundary condition, not a physical value — the forward
    // difference U_out - 0 across the r_c shell would otherwise produce a spurious
    // kinetic-energy cliff all around the exclusion sphere.
    let insideSelf = isInsideRcEnergy(i, j, k, myL);
    let sameL_ip = label[id + p.S2] == myL && !insideSelf && !isInsideRcEnergy(i+1u, j, k, myL);
    let sameL_jp = label[id + p.S]  == myL && !insideSelf && !isInsideRcEnergy(i, j+1u, k, myL);
    let sameL_kp = label[id + 1u]   == myL && !insideSelf && !isInsideRcEnergy(i, j, k+1u, myL);
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

  // K line data along j=sliceK axis
  if (j == 0u) {
    out[3u * SS * SS + i] = K[i * p.S2 + sk * p.S + sk];
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

  // Hide density inside r_c
  let rc = atoms[lbl].rc;
  var dens = u * u;
  if (rc > 0.0) {
    let dx = (f32(i) - f32(atoms[lbl].posI)) * p.h;
    let dy = (f32(j) - f32(atoms[lbl].posJ)) * p.h;
    let dz = (f32(sk) - f32(atoms[lbl].posK)) * p.h;
    if (sqrt(dx*dx + dy*dy + dz*dz) < rc) { dens = 0.0; }
  }
  out[i * SS + j] = dens;
  out[SS * SS + i * SS + j] = Zlbl;

  // Boundary: only where density exists (skip empty Voronoi regions)
  var bnd = 0.0;
  if (dens > 1e-6) {
    if (i > 1u && i < p.NN - 1u) {
      if (lbl != label[idx + p.S2]) { bnd = 1.0; }
    }
    if (j > 1u && j < p.NN - 1u) {
      if (lbl != label[idx + p.S]) { bnd = 1.0; }
    }
  }
  out[2u * SS * SS + i * SS + j] = bnd;
}
`;

// === Nuclear dynamics shaders ===

// Gradient of electron potential P at nuclear positions — reads directly from P_buf[0]
const FORCE_RADIUS = Math.max(3, Math.round(1.0 / hGrid));  // ~1 au sphere in grid cells
const gradPtotal_WGSL = `
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

  let ZA = atoms[atom].Z;
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
  for (var n: u32 = 0u; n < ${NELEC}u; n++) {
    let Za = atoms[n].Z;
    if (Za <= 0.0) { continue; }
    let di = (f32(i) - f32(atoms[n].posI)) * p.h;
    let dj = (f32(j) - f32(atoms[n].posJ)) * p.h;
    let dk = (f32(k) - f32(atoms[n].posK)) * p.h;
    let r = sqrt(di*di + dj*dj + dk*dk + p.h2);
    Kval += Za / r;
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
  let soft = 0.04 * p.h2;
  let end = range.start + range.count;

  for (var n: u32 = range.start; n < end; n++) {
    let Za = atoms[n].Z;
    if (Za <= 0.0) { continue; }
    let dx = xi - f32(atoms[n].posI) * p.h;
    let dy = yj - f32(atoms[n].posJ) * p.h;
    let dz = zk - f32(atoms[n].posK) * p.h;
    let r2 = dx*dx + dy*dy + dz*dz + p.h2;
    let r = sqrt(r2);
    Kval += Za / r;
    // Normalized trial: ∫U²dV = Z_eff analytically (U = Z²/√π · exp(-Z·r))
    // Domains assigned by highest normalized density
    let uTrial = Za * Za * ${(1/Math.sqrt(Math.PI)).toFixed(10)} * exp(-Za * r);
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
  if (atoms[lbl].rc > 0.0) { return false; }
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

  // Inside exclusion: HU = 0
  if (isInsideRc(i, j, k, myL) || isInsideAnalytical(i, j, k, myL)) {
    HU[id] = 0.0;
    return;
  }

  let uc = Ui[id];

  // Neumann BC at domain boundaries and exclusion surfaces (same as updateU)
  let l_ip = label[id + p.S2]; let excl_ip = isInsideRc(i+1u, j, k, l_ip) || isInsideAnalytical(i+1u, j, k, myL);
  let l_im = label[id - p.S2]; let excl_im = isInsideRc(i-1u, j, k, l_im) || isInsideAnalytical(i-1u, j, k, myL);
  let l_jp = label[id + p.S];  let excl_jp = isInsideRc(i, j+1u, k, l_jp) || isInsideAnalytical(i, j+1u, k, myL);
  let l_jm = label[id - p.S];  let excl_jm = isInsideRc(i, j-1u, k, l_jm) || isInsideAnalytical(i, j-1u, k, myL);
  let l_kp = label[id + 1u];   let excl_kp = isInsideRc(i, j, k+1u, l_kp) || isInsideAnalytical(i, j, k+1u, myL);
  let l_km = label[id - 1u];   let excl_km = isInsideRc(i, j, k-1u, l_km) || isInsideAnalytical(i, j, k-1u, myL);

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
let addNucRepulsion = true;
let vcycleEnabled = true;
let vcycleCount = 0;

const SLICE_SIZE = (3 * S * S + S) * 4;  // 3 image slices (density, Z, boundary) + 1 K line
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
  }
  device.queue.writeBuffer(atomBuf, 0, ab);
}

async function uploadInitialData() {
  console.log("GPU Init: " + NELEC + " atoms, NN=" + NN + " S3=" + S3);
  fillAtomBuf();

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
  addNucRepulsion = true;
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
  addNucRepulsion = true;
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
  createCanvas(700, 700);
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
    if (SIC_INTERVAL < 999999) {
      for (let m = 0; m < NELEC; m++) {
        domainBufs[m] = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        // slot 0: domain index (u32). slot 1: Zeff as float-bits (for subtractPself SIC factor).
        const buf = new ArrayBuffer(16);
        new Uint32Array(buf, 0, 1)[0] = m;
        new Float32Array(buf, 4, 1)[0] = Z[m] || 1.0;
        device.queue.writeBuffer(domainBufs[m], 0, buf);
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

    // Nuclear dynamics bind groups
    // gradPtotal: reads P_buf[0] directly (already converged from main V-cycle)
    for (let c = 0; c < 2; c++) {
      gradPtotalBG[c] = device.createBindGroup({ layout: gradPtotalPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: P_buf[0] } },
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
        const buf = new ArrayBuffer(16);
        new Uint32Array(buf, 0, 1)[0] = m;
        new Float32Array(buf, 4, 1)[0] = Z[m] || 1.0;
        device.queue.writeBuffer(domainBufs[m], 0, buf);
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
    // --- Compute rho_total from U[cur] + labels ---
    let vp = enc.beginComputePass();
    vp.setPipeline(computeRhoPL);
    vp.setBindGroup(0, computeRhoBG[cur]);
    dispatchLinear(vp, INTERIOR);
    vp.end();

    // --- Poisson solve (skip for single-electron systems — no V_ee) ---
    if (N_ELECTRONS > 1) {
    // --- Jacobi smooth P every step (2 sweeps: P[0]->P[1]->P[0]) ---
    for (let js = 0; js < 2; js++) {
      vp = enc.beginComputePass();
      vp.setPipeline(jacobiSmoothPL);
      vp.setBindGroup(0, jacobiFineBG[js]);
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

    } // end N_ELECTRONS > 1 Poisson block

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

    // Nuclear force computation at N_MOVE intervals — gradient of P directly (always compute for display)
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
  if (N_ELECTRONS <= 1) {
    enc.clearBuffer(PotherBuf);  // single electron: no V_ee
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
  } // end N_ELECTRONS > 1 else block

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
  for (let s = 0; s < W_STEPS_PER_FRAME; s++) {
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
  // Per-H correction disabled (sqrt(r2+h2) softening, no exclusion sphere)

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
    for (let a = 0; a < NELEC; a++) {
      for (let b = a + 1; b < NELEC; b++) {
        if (Z[a] === 0 || Z[b] === 0) continue;
        const d = Math.sqrt(
          ((nucPos[a][0]-nucPos[b][0])*hGrid)**2 +
          ((nucPos[a][1]-nucPos[b][1])*hGrid)**2 +
          ((nucPos[a][2]-nucPos[b][2])*hGrid)**2 + soft_nuc);
        E_KK += Z_nuc[a]*Z_nuc[b]/d;
      }
    }
  }
  E = E_T + E_eK + E_ee + E_KK;
  E_bind = E - E_atoms_sum;

  if (!isFinite(E)) {
    gpuError = "Numerical instability at step " + tStep;
    return;
  }

  console.log("Step " + tStep + ": E=" + E.toFixed(6) + " (" + lastMs.toFixed(0) + "ms/" + n + "steps)");

  // Force readback and nuclear dynamics
  if (needForceReadback) {
    await forceSumsReadBuf.mapAsync(GPUMapMode.READ);
    const forceData = new Float32Array(forceSumsReadBuf.getMappedRange().slice(0));
    forceSumsReadBuf.unmap();
    await moveNuclei(forceData);
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

    // Compute rho
    let vp = enc.beginComputePass();
    vp.setPipeline(computeRhoPL);
    vp.setBindGroup(0, computeRhoBG[cur]);
    dispatchLinear(vp, INTERIOR);
    vp.end();

    // Poisson solve
    if (N_ELECTRONS > 1) {
      for (let js = 0; js < 4; js++) {
        vp = enc.beginComputePass();
        vp.setPipeline(jacobiSmoothPL);
        vp.setBindGroup(0, jacobiFineBG[js % 2]);
        dispatchLinear(vp, INTERIOR);
        vp.end();
      }
      // V-cycle
      if (vcycleEnabled) {
        vp = enc.beginComputePass(); vp.setPipeline(computeResidualPL); vp.setBindGroup(0, residualBG[0]); dispatchLinear(vp, INTERIOR); vp.end();
        vp = enc.beginComputePass(); vp.setPipeline(restrictPL); vp.setBindGroup(0, restrictBG); vp.dispatchWorkgroups(WG_COARSE); vp.end();
        enc.clearBuffer(Pc_buf[0]);
        for (let cs = 0; cs < 10; cs++) { vp = enc.beginComputePass(); vp.setPipeline(coarseSmoothPL); vp.setBindGroup(0, coarseSmoothBG[cs % 2]); vp.dispatchWorkgroups(WG_COARSE); vp.end(); }
        vp = enc.beginComputePass(); vp.setPipeline(prolongCorrectPL); vp.setBindGroup(0, prolongCorrectBG); dispatchLinear(vp, INTERIOR); vp.end();
      }
    }

    // SIC
    if (N_ELECTRONS <= 1) {
      enc.clearBuffer(PotherBuf);
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
    for (let s = 0; s < W_STEPS_PER_FRAME; s++) {
      let bp = enc.beginComputePass();
      bp.setPipeline(evolveBoundaryPL);
      bp.setBindGroup(0, evolveBoundaryBG[next]);
      dispatchLinear(bp, INTERIOR);
      bp.end();
      enc.copyBufferToBuffer(label2Buf, 0, labelBuf, 0, S3 * 4);
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
  // Per-H correction disabled (sqrt(r2+h2) softening, no exclusion sphere)
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
    for (let a = 0; a < NELEC; a++) {
      for (let b = a + 1; b < NELEC; b++) {
        if (Z[a] === 0 || Z[b] === 0) continue;
        const d = Math.sqrt(
          ((nucPos[a][0]-nucPos[b][0])*hGrid)**2 +
          ((nucPos[a][1]-nucPos[b][1])*hGrid)**2 +
          ((nucPos[a][2]-nucPos[b][2])*hGrid)**2 + soft_nuc);
        E_KK += Z_nuc[a]*Z_nuc[b]/d;
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
    if (Z[a] === 0) { nucForce[a] = [0,0,0]; continue; }
    nucForce[a] = [gpuForces[a*3], gpuForces[a*3+1], gpuForces[a*3+2]];
  }

  // Add nuclear-nuclear (kernel-kernel) Coulomb repulsion forces
  // F_A += sum_{B≠A} Z_A * Z_B * (R_A - R_B) / |R_A - R_B|^3
  for (let a = 0; a < NELEC; a++) {
    if (Z[a] === 0) continue;
    for (let b = 0; b < NELEC; b++) {
      if (b === a || Z[b] === 0) continue;
      const dx = (nucPos[a][0] - nucPos[b][0]) * hGrid;
      const dy = (nucPos[a][1] - nucPos[b][1]) * hGrid;
      const dz = (nucPos[a][2] - nucPos[b][2]) * hGrid;
      const r2 = dx*dx + dy*dy + dz*dz;
      const r = Math.sqrt(r2);
      const inv_r3 = 1.0 / (r * r2);
      nucForce[a][0] += Z[a] * Z[b] * dx * inv_r3;
      nucForce[a][1] += Z[a] * Z[b] * dy * inv_r3;
      nucForce[a][2] += Z[a] * Z[b] * dz * inv_r3;
    }
  }

  console.log("Forces (elec+nuc): " + nucForce.filter((_,i) => Z[i]>0).map((f,i) =>
    atomLabels[i]+"=("+f.map(x=>x.toExponential(3)).join(",")+")").join(" "));

  for (let sub = 0; sub < NUC_SUBSTEPS; sub++) {
    for (let a = 0; a < NELEC; a++) {
      if (Z[a] === 0) continue;
      const m = nucMass(Z[a]);
      for (let d = 0; d < 3; d++) {
        nucVel[a][d] += nucForce[a][d] / m * DT_NUC * forceScale;
        nucVel[a][d] *= DAMPING;
        nucVel[a][d] = Math.max(-MAX_VEL, Math.min(MAX_VEL, nucVel[a][d]));
        nucPos[a][d] += nucVel[a][d] * DT_NUC / hGrid;
        nucPos[a][d] = Math.max(5, Math.min(NN - 5, nucPos[a][d]));
      }
    }
  }

  nucStepCount++;
  console.log("Nuc step " + nucStepCount + ": " +
    nucPos.filter((_, i) => Z[i] > 0).map(p => "(" + p.map(x => x.toFixed(2)).join(",") + ")").join(" "));

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

      // Auto-enable dynamics after convergence (skip if adaptive sweep)
      if (!dynamicsEnabled && !window.CONVERGENCE_THRESHOLD && phaseSteps >= 10000) {
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
            if (window.onSweepDone) window.onSweepDone(E, phaseSteps);
          }
        } else {
          window._convCount = 0;
        }
        window._prevE = E;
      }

      if (!dynamicsEnabled && phaseSteps >= TOTAL_STEPS) {
        console.log("=== DONE: E=" + E.toFixed(6) + " ===");
        phase = 1;  // done
        if (window.onSweepDone) window.onSweepDone(E, phaseSteps);
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
    loadPixels();
    const d = pixelDensity();
    const W = 700 * d, H = 700 * d;
    for (let p = 0; p < W * H * 4; p += 4) {
      pixels[p] = 0; pixels[p+1] = 0; pixels[p+2] = 0; pixels[p+3] = 255;
    }
    // Element colors: H=yellow, O=red, N=blue, C=green
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
        const Zel = sliceData[SS2 + b];         // slot 1: element Z
        const bnd = sliceData[2 * SS2 + b];     // slot 2: boundary
        const norm = Math.min(1.0, dens * invMax);
        const brightness = 255 * Math.sqrt(norm);
        const rgb = zRGB[Math.round(Zel)] || [0.5, 0.5, 0.5];
        let ri = Math.min(255, Math.floor(brightness * rgb[0]));
        let gi = Math.min(255, Math.floor(brightness * rgb[1]));
        let bi = Math.min(255, Math.floor(brightness * rgb[2]));
        // Dim boundary overlay (don't replace density, just brighten slightly)
        if (bnd > 0.5) {
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
      const zRGBplot = {1:[255,255,0], 2:[255,60,60], 3:[60,130,255], 4:[60,255,60]};
      const lineH = 60;
      const sc = lineH / globalMax;
      for (let li = 0; li < nLines; li++) {
        const row = lineRows[li];
        const rowY = PX * row;
        stroke(255, 255, 255, 40); strokeWeight(1);
        line(0, rowY, 700, rowY);
        strokeWeight(2); noFill();
        for (let i = 1; i < NN - 1; i++) {
          const v1 = sliceData[i * SS + row];
          const v2 = sliceData[(i+1) * SS + row];
          if (v1 < 1e-12 && v2 < 1e-12) continue;
          const z = Math.round(sliceData[SS2 + i * SS + row]);
          const c = zRGBplot[z] || [180,180,180];
          stroke(c[0], c[1], c[2], 220);
          line(PX * i, rowY - v1 * sc, PX * (i+1), rowY - v2 * sc);
        }
      }
    }

    // K potential line plot along the slice axis (through nuclei)
    const kBase = 3 * SS * SS;
    // Use median-based scale to avoid 1/r singularity dominating the plot
    let kVals = [];
    for (let i = 1; i < NN; i++) {
      const kv = sliceData[kBase + i];
      if (kv > 0) kVals.push(kv);
    }
    kVals.sort((a, b) => a - b);
    const kMax = kVals.length > 0 ? kVals[kVals.length - 1] : 0;
    const kCap = kVals.length > 0 ? kVals[Math.floor(kVals.length * 0.95)] * 3 : 1;
    if (kMax > 0) {
      const kPlotH = 80;
      const kPlotY = 700 - 10;
      const kSc = kPlotH / kCap;
      stroke(0, 200, 255, 180); strokeWeight(1.5); noFill();
      for (let i = 1; i < NN - 1; i++) {
        const k1 = sliceData[kBase + i];
        const k2 = sliceData[kBase + i + 1];
        line(PX * i, kPlotY - k1 * kSc, PX * (i+1), kPlotY - k2 * kSc);
      }
      fill(0, 200, 255); noStroke();
      text("K(r) max=" + kMax.toFixed(2), 5, kPlotY - kPlotH + 10);
    }
  }

  // Draw nuclear positions with force arrows
  fill(255); stroke(255); strokeWeight(1);
  for (let n = 0; n < NELEC; n++) {
    if (Z[n] > 0) {
      circle(nucPos[n][0] * PX, nucPos[n][1] * PX, 6);
      // Force arrows always shown; dimmer when dynamics off
      if (nucForce[n]) {
        const fx = nucForce[n][0], fy = nucForce[n][1];
        const fmag = Math.sqrt(fx*fx + fy*fy);
        if (fmag > 1e-8) {
          const arrowScale = 100;
          const ax = nucPos[n][0] * PX + fx * arrowScale;
          const ay = nucPos[n][1] * PX + fy * arrowScale;
          stroke(dynamicsEnabled ? 255 : 180, dynamicsEnabled ? 0 : 80, dynamicsEnabled ? 0 : 80);
          strokeWeight(1.5);
          line(nucPos[n][0] * PX, nucPos[n][1] * PX, ax, ay);
        }
      }
    }
  }


  // Screen boundary
  noFill(); stroke(100); strokeWeight(1);
  rect(0, 0, 700, 700);
  noStroke();

  fill(255);
  const labels = atomLabels.map((el, i) => [el, Z_orig[i]]).filter(x => x[1] > 0).map(x => x[0] + "(Z=" + x[1] + ")").join(" ");
  const pLabel = phase === 0 ? "running" : "DONE";
  text("Molecule: " + labels + " | " + screenAu + " au | " + pLabel + " | " + NN + "^3", 5, 20);
  text("step " + tStep + " (" + phaseSteps + "/" + TOTAL_STEPS + ")  E=" + (E_T + E_eK + E_ee + E_KK).toFixed(6) + "  E_min=" + E_min.toFixed(6), 5, 35);
  if (lastMs > 0) text((lastMs / STEPS_PER_FRAME).toFixed(1) + "ms/step", 300, 35);

  fill(200);
  text("T=" + E_T.toFixed(4) + " V_eK=" + E_eK.toFixed(4) + " V_ee=" + E_ee.toFixed(4) + " V_KK=" + E_KK.toFixed(4) + "  Dipole=" + dipole_D.toFixed(3) + " D  E_bind=" + E_bind.toFixed(4) + " Ha", 5, 50);
  fill(vcycleEnabled ? [0,255,0] : [255,100,0]);
  const solverName = useLOBPCG ? "LOBPCG" : useCheb ? "Chebyshev" : "ITP";
  text("V-cycle: " + (vcycleEnabled ? "ON" : "OFF") + " (" + vcycleCount + ")  Solver: " + solverName + " [L=LOBPCG C=Cheb]", 5, 65);

  // Dynamics status
  fill(dynamicsEnabled ? [0,255,255] : [150,150,150]);
  text("Dynamics: " + (dynamicsEnabled ? "ON" : "OFF") + " (nucStep=" + nucStepCount + ")  Force=" + forceScale.toFixed(1) + "x  [D toggle, +/- force]", 5, 80);
  // H-H kernel distance (first 2 atoms with Z>0)
  if (NELEC >= 2 && Z[0] > 0 && Z[1] > 0 && nucPos[0] && nucPos[1]) {
    const dx = (nucPos[0][0] - nucPos[1][0]) * hGrid;
    const dy = (nucPos[0][1] - nucPos[1][1]) * hGrid;
    const dz = (nucPos[0][2] - nucPos[1][2]) * hGrid;
    const dAu = Math.sqrt(dx*dx + dy*dy + dz*dz);
    fill(255, 200, 100);
    text("kernel distance: " + dAu.toFixed(3) + " au (" + (dAu * 0.529177).toFixed(3) + " Å, target 1.4 au)", 5, 95);
  }

  // r_c values
  fill(200, 180, 255);
  var rcInfo = atomLabels.map((el, i) => [el, Z_orig[i], r_cut[i]]).filter(x => x[1] > 0);
  var rcSeen = {};
  rcInfo.forEach(x => { rcSeen[x[0]] = x[2]; });
  text("r_c: " + Object.keys(rcSeen).map(k => k + "=" + rcSeen[k]).join(" "), 5, 95);

  // Diagnostics
  fill(255, 255, 0);
  text("maxDens=" + (window._diagMaxD || 0).toExponential(2) +
    " pos=" + (window._diagPos || 0) + " nan=" + (window._diagNan || 0) +
    " neg=" + (window._diagNeg || 0), 5, 110);
  if (window._initDebug) {
    fill(0, 255, 255);
    text(window._initDebug, 300, 110);
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
  text("Slice k=" + sliceK + "/" + NN + "  [Up/Down to scroll]", 5, 125);
}

function keyPressed() {
  if (key === 'v' || key === 'V') {
    vcycleEnabled = !vcycleEnabled;
    vcycleCount = 0;
    console.log("V-cycle " + (vcycleEnabled ? "ENABLED" : "DISABLED"));
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
