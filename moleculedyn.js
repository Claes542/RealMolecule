// Molecule Quantum Simulation — WebGPU Compute Shaders + Nuclear Dynamics
// Up to 10 atoms placed interactively, 3D geometry
const NN = window.USER_NN || 200;
const S = NN + 1;
const S2 = S * S;
const S3 = S * S * S;
const N2 = Math.round(NN / 2);
const MAX_ATOMS = 300;
const _uz = window.USER_Z || [2, 3, 1, 0, 0];
while (_uz.length < MAX_ATOMS) _uz.push(0);
const NELEC = _uz.filter(z => z > 0).length || 3;
const NRED = NELEC + 3;  // N norms + T + V_eK + V_ee
const r_cut = window.USER_RC || [0, 0, 0, 0, 0];
while (r_cut.length < MAX_ATOMS) r_cut.push(0);
let R_out = 1.0;   // au, outer w cutoff
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
let nucPos = _atoms.map(a => [a.i, a.j, a.k !== undefined ? a.k : N2]);
const molNucPos = nucPos.map(p => [...p]);

let E_min = Infinity;
let screenAu = window.USER_SCREEN || 10;
let hGrid = screenAu / NN, h2v = hGrid * hGrid, h3v = hGrid * hGrid * hGrid;
const dv = 0.12;
let dtv = dv * h2v, half_dv = 0.5 * dv;
const PX = 400 / NN;
const INTERIOR = (NN - 1) * (NN - 1) * (NN - 1);
const STEPS_PER_FRAME = NELEC <= 5 ? 500 : NELEC <= 15 ? 100 : NELEC <= 30 ? 50 : 5;
const W_STEPS_PER_FRAME = 1;
const BOUNDARY_INTERVAL = 10;
const NORM_INTERVAL = 20;
const POISSON_INTERVAL = 50;
const SIC_INTERVAL = NELEC <= 15 ? 1 : NELEC <= 30 ? 5 : 999999;  // skip SIC for large molecules
const SIC_JACOBI = NELEC <= 15 ? 10 : 4;

// === Nuclear dynamics state ===
const N_MOVE = 2000;        // electronic steps between nuclear moves
const DT_NUC = 10.0;        // au (~0.24 fs)
const NUC_SUBSTEPS = 1;     // single step (forces recomputed each move)
const DAMPING = 0.90;       // strong damping for optimization
const MAX_VEL = 0.01;       // au/au_time
let nucVel = Array.from({length: MAX_ATOMS}, () => [0, 0, 0]);
let nucForce = Array.from({length: MAX_ATOMS}, () => [0, 0, 0]);
let nucStepCount = 0, dynamicsEnabled = false;
function nucMass(z) { return ({1:1, 2:16, 3:14, 4:12}[z] || 1) * 1836; }

// Multigrid coarse grid
if (NN % 2 !== 0) throw new Error("NN must be even for multigrid");
const NC = Math.floor(NN / 2);
const SC = NC + 1, SC2 = SC * SC, SC3 = SC * SC * SC;
const INTERIOR_C = (NC - 1) * (NC - 1) * (NC - 1);

// Reduce workgroup size: NRED * REDUCE_WG * 4 must be <= 16384 bytes
const REDUCE_WG = 1 << Math.floor(Math.log2(Math.min(128, Math.floor(4096 / NRED))));

// ===== WGSL SHADERS =====

// Param struct: 16 common fields = 64 bytes. Atom data in separate storage buffer.
const PARAM_BYTES = 64;
const paramStructWGSL = `
struct P {
  NN: u32, S: u32, S2: u32, S3: u32,
  N2: u32, voronoi: f32, R_out: f32, TWO_PI: f32,
  h: f32, h2: f32, inv_h: f32, inv_h2: f32,
  dt: f32, half_d: f32, h3: f32, _pad0: f32,
}`;

const ATOM_STRIDE = 8; // 8 f32s per atom (posI, posJ, posK, Z, rc, pad, pad, pad)
const ATOM_BUF_BYTES = MAX_ATOMS * ATOM_STRIDE * 4;
const atomStructWGSL = `
struct Atom {
  posI: u32, posJ: u32, posK: u32, Z: f32,
  rc: f32, _p0: f32, _p1: f32, _p2: f32,
}`;

// U update — label-based domains, Neumann BC at domain boundaries
const updateU_WGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> Ui: array<f32>;
@group(0) @binding(3) var<storage, read> label: array<u32>;
@group(0) @binding(4) var<storage, read> Pi: array<f32>;
@group(0) @binding(5) var<storage, read_write> Uo: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }

  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let myL = label[id];

  let uc = Ui[id];

  // Neumann BC: no diffusion across domain boundaries (independent u per domain)
  let u_ip = select(uc, Ui[id + p.S2], label[id + p.S2] == myL);
  let u_im = select(uc, Ui[id - p.S2], label[id - p.S2] == myL);
  let u_jp = select(uc, Ui[id + p.S],  label[id + p.S]  == myL);
  let u_jm = select(uc, Ui[id - p.S],  label[id - p.S]  == myL);
  let u_kp = select(uc, Ui[id + 1u],   label[id + 1u]   == myL);
  let u_km = select(uc, Ui[id - 1u],   label[id - 1u]   == myL);

  let lap = u_ip + u_im + u_jp + u_jm + u_kp + u_km - 6.0 * uc;

  // Full nuclear potential (all nuclei) minus other-electron repulsion (no self-repulsion)
  Uo[id] = uc + p.half_d * lap + p.dt * (K[id] - 2.0 * Pi[id]) * uc;
}
`;

// Level set boundary evolution — accumulate density difference in W, flip when W < 0
// W > 0 means "I belong here", W < 0 means "neighbor density dominates, should flip"
const evolveBoundaryWGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> labelIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> labelOut: array<u32>;
@group(0) @binding(3) var<storage, read> U: array<f32>;
@group(0) @binding(4) var<storage, read_write> W: array<f32>;
@group(0) @binding(5) var<storage, read> atoms: array<Atom>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }

  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let myL = labelIn[id];
  let myZ = atoms[myL].Z;
  let myRho = myZ * U[id] * U[id];

  let id_ip = id + p.S2; let id_im = id - p.S2;
  let id_jp = id + p.S;  let id_jm = id - p.S;
  let id_kp = id + 1u;   let id_km = id - 1u;

  let l_ip = labelIn[id_ip]; let l_im = labelIn[id_im];
  let l_jp = labelIn[id_jp]; let l_jm = labelIn[id_jm];
  let l_kp = labelIn[id_kp]; let l_km = labelIn[id_km];

  // Find best cross-boundary neighbor (highest density) + count same-domain
  var bestOtherL: u32 = myL;
  var bestOtherRho: f32 = 0.0;
  var myCnt: f32 = 0.0;

  if (l_ip == myL) { myCnt += 1.0; } else { let r = atoms[l_ip].Z * U[id_ip] * U[id_ip]; if (r > bestOtherRho) { bestOtherRho = r; bestOtherL = l_ip; } }
  if (l_im == myL) { myCnt += 1.0; } else { let r = atoms[l_im].Z * U[id_im] * U[id_im]; if (r > bestOtherRho) { bestOtherRho = r; bestOtherL = l_im; } }
  if (l_jp == myL) { myCnt += 1.0; } else { let r = atoms[l_jp].Z * U[id_jp] * U[id_jp]; if (r > bestOtherRho) { bestOtherRho = r; bestOtherL = l_jp; } }
  if (l_jm == myL) { myCnt += 1.0; } else { let r = atoms[l_jm].Z * U[id_jm] * U[id_jm]; if (r > bestOtherRho) { bestOtherRho = r; bestOtherL = l_jm; } }
  if (l_kp == myL) { myCnt += 1.0; } else { let r = atoms[l_kp].Z * U[id_kp] * U[id_kp]; if (r > bestOtherRho) { bestOtherRho = r; bestOtherL = l_kp; } }
  if (l_km == myL) { myCnt += 1.0; } else { let r = atoms[l_km].Z * U[id_km] * U[id_km]; if (r > bestOtherRho) { bestOtherRho = r; bestOtherL = l_km; } }

  var w = W[id];

  // Interior cells (all 6 neighbors same domain): keep W positive, no evolution
  if (myCnt >= 6.0) {
    w = max(w, 1.0);
    W[id] = w;
    labelOut[id] = myL;
    return;
  }

  // Boundary cell: evolve W continuously based on density difference
  // velocity = myRho - bestOtherRho (positive = I'm stronger, negative = neighbor stronger)
  let velocity = myRho - bestOtherRho;
  let dt_w: f32 = 0.1;
  w += dt_w * velocity;

  // Curvature regularization: penalize low same-domain count (convex protrusions)
  let curv = (myCnt - 3.0) / 3.0;  // [-1, +1], negative = surrounded by others
  w += 0.01 * curv;

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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }
  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let oldL = labelOld[id];
  let newL = labelNew[id];
  if (oldL != newL) {
    let Zold = atoms[oldL].Z;
    let Znew = atoms[newL].Z;
    U[id] *= sqrt(Zold / max(Znew, 0.001));
  }
}
`;

// Compute density for a single domain (for self-potential calculation)
const computeRhoSelfWGSL = `
${paramStructWGSL}
${atomStructWGSL}
struct DomIdx { idx: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read_write> rhoSelf: array<f32>;
@group(0) @binding(3) var<storage, read> atoms: array<Atom>;
@group(0) @binding(4) var<storage, read> label: array<u32>;
@group(0) @binding(5) var<uniform> dom: DomIdx;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }
  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;
  let u = U[id];
  rhoSelf[id] = select(0.0, atoms[label[id]].Z * u * u, label[id] == dom.idx);
}
`;

// Subtract per-domain self-potential from Pother at points in that domain
const subtractPselfWGSL = `
${paramStructWGSL}
struct DomIdx { idx: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Pm: array<f32>;
@group(0) @binding(2) var<storage, read_write> Pother: array<f32>;
@group(0) @binding(3) var<storage, read> label: array<u32>;
@group(0) @binding(4) var<uniform> dom: DomIdx;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }
  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;
  if (label[id] == dom.idx) {
    Pother[id] -= Pm[id];
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }
  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
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
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read_write> rhoTotal: array<f32>;
@group(0) @binding(3) var<storage, read> atoms: array<Atom>;
@group(0) @binding(4) var<storage, read> label: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }
  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;
  let u = U[id];
  rhoTotal[id] = atoms[label[id]].Z * u * u;
}
`;

// Compute residual: r = -2*pi*rho - lap(P)
const computeResidualWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Pin: array<f32>;
@group(0) @binding(2) var<storage, read_write> Rout: array<f32>;
@group(0) @binding(3) var<storage, read> rhoTotal: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }
  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }
  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;

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

// Cap workgroups so partials buffer stays small (grid-stride loop handles all cells)
const MAX_REDUCE_WG = 256;
const reduceWGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> label: array<u32>;
@group(0) @binding(3) var<storage, read> Pv: array<f32>;
@group(0) @binding(4) var<storage, read> K: array<f32>;
@group(0) @binding(5) var<storage, read_write> partials: array<f32>;
@group(0) @binding(6) var<storage, read> atoms: array<Atom>;

var<workgroup> sn: array<f32, ${NRED * REDUCE_WG}>;

@compute @workgroup_size(${REDUCE_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(workgroup_id) wgid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  let stride = ${MAX_REDUCE_WG}u * ${REDUCE_WG}u;

  for (var x: u32 = 0u; x < ${NRED}u; x++) { sn[lid * ${NRED}u + x] = 0.0; }

  // Grid-stride loop: each thread processes multiple cells
  var cell = gid.x;
  loop {
    if (cell >= tot) { break; }
    let k = (cell % NM) + 1u;
    let j = ((cell / NM) % NM) + 1u;
    let i = (cell / (NM * NM)) + 1u;
    let id = i * p.S2 + j * p.S + k;

    let v = U[id];
    let m = label[id];
    let Zm = atoms[m].Z;

    sn[lid * ${NRED}u + m] += v * v * p.h3;

    let v_ip = U[id + p.S2];
    let v_jp = U[id + p.S];
    let v_kp = U[id + 1u];
    let a = v_ip - v;
    let b = v_jp - v;
    let c = v_kp - v;
    sn[lid * ${NRED}u + ${NELEC}u] += Zm * 0.5 * (a * a + b * b + c * c) * p.h;
    sn[lid * ${NRED}u + ${NELEC + 1}u] += -Zm * K[id] * v * v * p.h3;
    sn[lid * ${NRED}u + ${NELEC + 2}u] += Zm * Pv[id] * v * v * p.h3;

    cell = cell + stride;
  }

  workgroupBarrier();

  for (var s: u32 = ${REDUCE_WG >> 1}u; s > 0u; s >>= 1u) {
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

const finalizeWGSL = `
struct NWG { count: u32 }
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> sums: array<f32>;
@group(0) @binding(2) var<uniform> nwg: NWG;

var<workgroup> wg: array<f32, ${NRED * REDUCE_WG}>;

@compute @workgroup_size(${REDUCE_WG})
fn main(@builtin(local_invocation_index) lid: u32) {
  for (var x: u32 = 0u; x < ${NRED}u; x++) { wg[lid * ${NRED}u + x] = 0.0; }

  for (var i: u32 = lid; i < nwg.count; i += ${REDUCE_WG}u) {
    for (var x: u32 = 0u; x < ${NRED}u; x++) {
      wg[lid * ${NRED}u + x] += partials[i * ${NRED}u + x];
    }
  }

  workgroupBarrier();

  for (var s: u32 = ${REDUCE_WG >> 1}u; s > 0u; s >>= 1u) {
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

const normalizeWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> U: array<f32>;
@group(0) @binding(2) var<storage, read> sums: array<f32>;
@group(0) @binding(3) var<storage, read> label: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (g.x >= tot) { return; }

  let k = (g.x % NM) + 1u;
  let j = ((g.x / NM) % NM) + 1u;
  let i = (g.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let n = sums[label[id]];
  if (n > 0.0) { U[id] *= inverseSqrt(n); }
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

  let idx = i * p.S2 + j * p.S + p.N2;

  // K line data along j=N2 axis (before boundary return so it's reachable for j==0)
  if (j == 0u) {
    out[3u * SS * SS + i] = K[i * p.S2 + p.N2 * p.S + p.N2];
  }

  if (i < 1u || i >= p.NN || j < 1u || j >= p.NN) {
    out[i * SS + j] = 0.0;
    out[SS * SS + i * SS + j] = 0.0;
    out[2u * SS * SS + i * SS + j] = 0.0;
    return;
  }

  let u = U[idx];
  let lbl = label[idx];
  let Zlbl = atoms[lbl].Z;
  out[i * SS + j] = Zlbl * u * u;
  out[SS * SS + i * SS + j] = Zlbl;

  // Boundary: only domain boundaries (label changes), skip density edges
  var bnd = 0.0;
  if (i > 1u && i < p.NN - 1u) {
    if (lbl != label[idx + p.S2]) { bnd = 1.0; }
  }
  if (j > 1u && j < p.NN - 1u) {
    if (lbl != label[idx + p.S]) { bnd = 1.0; }
  }
  out[2u * SS * SS + i * SS + j] = bnd;
}
`;

// === Nuclear dynamics shaders ===

// Gradient of electron potential P at nuclear positions — reads directly from P_buf[0]
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

  let ii = atoms[atom].posI;
  let jj = atoms[atom].posJ;
  let kk = atoms[atom].posK;

  let inv2h = 0.5 * p.inv_h;
  let dPdi = (Pt[(ii+1u)*p.S2 + jj*p.S + kk] - Pt[(ii-1u)*p.S2 + jj*p.S + kk]) * inv2h;
  let dPdj = (Pt[ii*p.S2 + (jj+1u)*p.S + kk] - Pt[ii*p.S2 + (jj-1u)*p.S + kk]) * inv2h;
  let dPdk = (Pt[ii*p.S2 + jj*p.S + (kk+1u)] - Pt[ii*p.S2 + jj*p.S + (kk-1u)]) * inv2h;

  forceSums[atom * 3u]      = 2.0 * ZA * dPdi;
  forceSums[atom * 3u + 1u] = 2.0 * ZA * dPdj;
  forceSums[atom * 3u + 2u] = 2.0 * ZA * dPdk;
}
`;

// Recompute nuclear potential K on GPU after nuclei move — adapted for atomBuf (loop)
const R_SING = 0.1;  // fixed singularity limit for 1/r potential (au)
const recomputeK_WGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> K: array<f32>;
@group(0) @binding(2) var<storage, read> atoms: array<Atom>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
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
    let r = sqrt(di*di + dj*dj + dk*dk);
    Kval += Za / max(r, ${R_SING});
  }
  K[id] = Kval;
}
`;
const WG_RECOMPUTE_K = Math.ceil(S3 / 256);

// ===== GPU STATE =====
let device, paramsBuf, atomBuf, K_buf, sumsBuf, sumsReadBuf, sliceBuf, sliceReadBuf, partialsBuf, numWGBuf;
let U_buf = [], P_buf = [], labelBuf, label2Buf, W_buf;
let rhoTotalBuf, residualBuf, Pc_buf = [], coarseRhsBuf;
let PotherBuf, PselfScratchBuf, domainBufs = [];
let updatePL, evolveBoundaryPL, fixBoundaryUPL, jacobiSmoothPL, reducePL, finalizePL, normalizePL, extractPL;
let computeRhoPL, computeResidualPL, restrictPL, coarseSmoothPL, prolongCorrectPL;
let computeRhoSelfPL, subtractPselfPL;
let updateBG = [], evolveBoundaryBG = [], fixBoundaryUBG = [], jacobiFineBG = [], reduceBG = [], finalizeBG, normalizeBG = [], extractBG = [];
let computeRhoBG = [], residualBG = [], prolongCorrectBG;
let restrictBG, coarseSmoothBG = [];
let computeRhoSelfBG = [], jacobiSelfBG = [], subtractPselfBG = [];
// Nuclear dynamics GPU state
let forceSumsBuf, forceSumsReadBuf;
let gradPtotalPL, recomputeK_PL;
let gradPtotalBG = [], recomputeK_BG;

let cur = 0, gpuReady = false, computing = false, initProgress = 0;
let tStep = 0, E = 0, lastMs = 0;
let E_T = 0, E_eK = 0, E_ee = 0, E_KK = 0;
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
const SUMS_BYTES = NRED * 4;

let sliceData = null;

function fillParamsBuf(pb) {
  const pu = new Uint32Array(pb);
  const pf = new Float32Array(pb);
  pu[0] = NN; pu[1] = S; pu[2] = S2; pu[3] = S3;
  pu[4] = N2; pf[5] = 1.0; pf[6] = R_out; pf[7] = 2 * Math.PI;
  pf[8] = hGrid; pf[9] = h2v; pf[10] = 1 / hGrid; pf[11] = 1 / h2v;
  pf[12] = dtv; pf[13] = half_dv; pf[14] = h3v; pf[15] = 0;
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
  console.log("Init: " + NELEC + " atoms, NN=" + NN);

  const Kd = new Float32Array(S3);
  const Ud = new Float32Array(S3);
  const Ld = new Uint32Array(S3);
  const Pd = new Float32Array(S3);
  const soft = 0.04 * h2v;
  const NA = NELEC;

  // Pre-compute nucleus positions in au for speed
  const nxAu = new Float64Array(NA);
  const nyAu = new Float64Array(NA);
  const nzAu = new Float64Array(NA);
  for (let n = 0; n < NA; n++) {
    nxAu[n] = nucPos[n][0] * hGrid;
    nyAu[n] = nucPos[n][1] * hGrid;
    nzAu[n] = nucPos[n][2] * hGrid;
  }

  // Chunked init — yield to browser every CHUNK_I slices to avoid freezing
  const CHUNK_I = Math.max(1, Math.floor(40 / Math.max(1, NA / 10)));
  const t0 = performance.now();
  for (let iStart = 0; iStart <= NN; iStart += CHUNK_I) {
    const iEnd = Math.min(iStart + CHUNK_I, NN + 1);
    for (let i = iStart; i < iEnd; i++) {
      const xi = i * hGrid;
      for (let j = 0; j <= NN; j++) {
        const yj = j * hGrid;
        for (let k = 0; k <= NN; k++) {
          const zk = k * hGrid;
          const id = i * S2 + j * S + k;

          // Find nearest atom + accumulate K potential
          let best = -1, bestR2 = Infinity, Kval = 0;
          for (let n = 0; n < NA; n++) {
            if (Z[n] <= 0) continue;
            const dx = xi - nxAu[n], dy = yj - nyAu[n], dz = zk - nzAu[n];
            const r2 = dx*dx + dy*dy + dz*dz + soft;
            const r = Math.sqrt(r2);
            Kval += Z[n] / Math.max(r, R_SING);
            if (r2 < bestR2) { bestR2 = r2; best = n; }
          }
          Kd[id] = Kval;
          if (best >= 0) {
            const r = Math.sqrt(bestR2);
            Ld[id] = best;
            Ud[id] = (r < R_out) ? Math.exp(-r) : 0.0;
            Pd[id] = Kval / NA;
          }
        }
      }
    }
    // Yield to browser to prevent freezing
    if (iStart + CHUNK_I <= NN) {
      initProgress = (iStart + CHUNK_I) / (NN + 1);
      await new Promise(r => setTimeout(r, 0));
    }
  }
  console.log("Init computed in " + ((performance.now() - t0)/1000).toFixed(1) + "s");

  console.log("Uploading to GPU...");
  device.queue.writeBuffer(K_buf, 0, Kd);
  device.queue.writeBuffer(labelBuf, 0, Ld);
  device.queue.writeBuffer(label2Buf, 0, Ld);
  const Wd = new Float32Array(S3); Wd.fill(1.0);
  device.queue.writeBuffer(W_buf, 0, Wd);
  for (let i = 0; i < 2; i++) {
    device.queue.writeBuffer(U_buf[i], 0, Ud);
    device.queue.writeBuffer(P_buf[i], 0, Pd);
  }
  device.queue.writeBuffer(PotherBuf, 0, Pd);
  fillAtomBuf();
  cur = 0;
}

function updateParamsBuf() {
  const pb = new ArrayBuffer(PARAM_BYTES);
  fillParamsBuf(pb);
  device.queue.writeBuffer(paramsBuf, 0, pb);
  fillAtomBuf();
}

function startMolPhase() {
  nucPos = molNucPos.map(p => [...p]);
  Z = [...Z_orig]; Ne = [...Ne_orig];
  R_out = 1.0;
  addNucRepulsion = true;
  updateParamsBuf();
  uploadInitialData();
  tStep = 0;
  phaseSteps = 0;
  frameCount = 0;
  E_min = Infinity;
  cur = 0;
  phase = 0;
  console.log("=== Molecule: " + atomLabels.join("-") + " ===");
}

// External API: update atom positions and restart
window.restartWithPositions = function(newPositions) {
  // newPositions: array of [i, j, k] per atom
  for (let n = 0; n < newPositions.length && n < NELEC; n++) {
    nucPos[n] = [...newPositions[n]];
    molNucPos[n] = [...newPositions[n]];
  }
  R_out = 1.0;
  addNucRepulsion = true;
  updateParamsBuf();
  uploadInitialData();
  tStep = 0;
  phaseSteps = 0;
  frameCount = 0;
  E_min = Infinity;
  cur = 0;
  phase = 0;
  computing = false;
  gpuError = null;
};

function setup() {
  console.log("setup: NELEC=" + NELEC + " NRED=" + NRED + " REDUCE_WG=" + REDUCE_WG);
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

    K_buf = device.createBuffer({ size: bs, usage });
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
    for (let m = 0; m < NELEC; m++) {
      domainBufs[m] = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(domainBufs[m], 0, new Uint32Array([m, 0, 0, 0]));
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

    const partialSize = WG_REDUCE * NRED * 4;
    partialsBuf = device.createBuffer({ size: partialSize, usage: GPUBufferUsage.STORAGE });
    sumsBuf = device.createBuffer({ size: SUMS_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    sumsReadBuf = device.createBuffer({ size: SUMS_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    sliceBuf = device.createBuffer({ size: SLICE_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    sliceReadBuf = device.createBuffer({ size: SLICE_SIZE, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    numWGBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(numWGBuf, 0, new Uint32Array([WG_REDUCE, 0, 0, 0]));

    // Force buffers for nuclear dynamics
    const forceSumsSize = NELEC * 3 * 4;
    forceSumsBuf = device.createBuffer({ size: forceSumsSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    forceSumsReadBuf = device.createBuffer({ size: forceSumsSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    const pb = new ArrayBuffer(PARAM_BYTES);
    fillParamsBuf(pb);
    paramsBuf = device.createBuffer({ size: PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuf, 0, pb);

    await uploadInitialData();

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
    const reduceMod = await compileShader('reduce', reduceWGSL);
    const finalizeMod = await compileShader('finalize', finalizeWGSL);
    const normalizeMod = await compileShader('normalize', normalizeWGSL);
    const extractMod = await compileShader('extract', extractWGSL);
    const computeRhoSelfMod = await compileShader('computeRhoSelf', computeRhoSelfWGSL);
    const subtractPselfMod = await compileShader('subtractPself', subtractPselfWGSL);
    // Nuclear dynamics shaders
    const gradPtotalMod = await compileShader('gradPtotal', gradPtotal_WGSL);
    const recomputeK_Mod = await compileShader('recomputeK', recomputeK_WGSL);

    updatePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: updateMod, entryPoint: 'main' } });
    evolveBoundaryPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: evolveBoundaryMod, entryPoint: 'main' } });
    const fixBoundaryUMod = await compileShader('fixBoundaryU', fixBoundaryU_WGSL);
    fixBoundaryUPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: fixBoundaryUMod, entryPoint: 'main' } });
    computeRhoSelfPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: computeRhoSelfMod, entryPoint: 'main' } });
    subtractPselfPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: subtractPselfMod, entryPoint: 'main' } });
    jacobiSmoothPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: jacobiSmoothMod, entryPoint: 'main' } });
    computeRhoPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: computeRhoMod, entryPoint: 'main' } });
    computeResidualPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: computeResidualMod, entryPoint: 'main' } });
    restrictPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: restrictMod, entryPoint: 'main' } });
    coarseSmoothPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: coarseSmoothMod, entryPoint: 'main' } });
    prolongCorrectPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: prolongCorrectMod, entryPoint: 'main' } });
    reducePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: reduceMod, entryPoint: 'main' } });
    finalizePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: finalizeMod, entryPoint: 'main' } });
    normalizePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: normalizeMod, entryPoint: 'main' } });
    extractPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: extractMod, entryPoint: 'main' } });
    // Nuclear dynamics pipelines
    gradPtotalPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: gradPtotalMod, entryPoint: 'main' } });
    recomputeK_PL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: recomputeK_Mod, entryPoint: 'main' } });

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
      ]});
      reduceBG[c] = device.createBindGroup({ layout: reducePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: labelBuf } },
        { binding: 3, resource: { buffer: PotherBuf } },
        { binding: 4, resource: { buffer: K_buf } },
        { binding: 5, resource: { buffer: partialsBuf } },
        { binding: 6, resource: { buffer: atomBuf } },
      ]});
      normalizeBG[c] = device.createBindGroup({ layout: normalizePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: sumsBuf } },
        { binding: 3, resource: { buffer: labelBuf } },
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
        { binding: 3, resource: { buffer: atomBuf } },
        { binding: 4, resource: { buffer: labelBuf } },
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
        { binding: 5, resource: { buffer: atomBuf } },
      ]});
      // Fix U at flipped cells: labelBuf=old, label2Buf=new, U[cur]=read_write
      fixBoundaryUBG[u] = device.createBindGroup({ layout: fixBoundaryUPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: labelBuf } },
        { binding: 2, resource: { buffer: label2Buf } },
        { binding: 3, resource: { buffer: U_buf[u] } },
        { binding: 4, resource: { buffer: atomBuf } },
      ]});
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

    finalizeBG = device.createBindGroup({ layout: finalizePL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: partialsBuf } },
      { binding: 1, resource: { buffer: sumsBuf } },
      { binding: 2, resource: { buffer: numWGBuf } },
    ]});

    // Per-domain self-potential bind groups
    for (let m = 0; m < NELEC; m++) {
      // computeRhoSelf: one per domain x 2 for cur
      computeRhoSelfBG[m] = [];
      for (let c = 0; c < 2; c++) {
        computeRhoSelfBG[m][c] = device.createBindGroup({ layout: computeRhoSelfPL.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: paramsBuf } },
          { binding: 1, resource: { buffer: U_buf[c] } },
          { binding: 2, resource: { buffer: residualBuf } },
          { binding: 3, resource: { buffer: atomBuf } },
          { binding: 4, resource: { buffer: labelBuf } },
          { binding: 5, resource: { buffer: domainBufs[m] } },
        ]});
      }
      // subtractPself: one per domain
      subtractPselfBG[m] = device.createBindGroup({ layout: subtractPselfPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: P_buf[1] } },
        { binding: 2, resource: { buffer: PotherBuf } },
        { binding: 3, resource: { buffer: labelBuf } },
        { binding: 4, resource: { buffer: domainBufs[m] } },
      ]});
    }
    // Jacobi for self-potential: P_buf[1] <-> PselfScratchBuf, rhs=residualBuf
    jacobiSelfBG[0] = device.createBindGroup({ layout: jacobiSmoothPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: P_buf[1] } },
      { binding: 2, resource: { buffer: PselfScratchBuf } },
      { binding: 3, resource: { buffer: residualBuf } },
    ]});
    jacobiSelfBG[1] = device.createBindGroup({ layout: jacobiSmoothPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: PselfScratchBuf } },
      { binding: 2, resource: { buffer: P_buf[1] } },
      { binding: 3, resource: { buffer: residualBuf } },
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

    console.log("Ready! dispatch(" + WG_UPDATE + ") " + NELEC + " domains + multigrid V-cycle + SIC + dynamics");
    gpuReady = true;
    startMolPhase();

  } catch (e) {
    gpuError = e.message || String(e);
    console.error("GPU init failed:", e);
  }
}

async function doSteps(n) {
  const t0 = performance.now();
  const enc = device.createCommandEncoder();
  let needForceReadback = false;

  for (let s = 0; s < n; s++) {
    const next = 1 - cur;
    // --- Compute rho_total from U[cur] + labels ---
    let vp = enc.beginComputePass();
    vp.setPipeline(computeRhoPL);
    vp.setBindGroup(0, computeRhoBG[cur]);
    vp.dispatchWorkgroups(WG_UPDATE);
    vp.end();

    // --- Jacobi smooth P every step (2 sweeps: P[0]->P[1]->P[0]) ---
    for (let js = 0; js < 2; js++) {
      vp = enc.beginComputePass();
      vp.setPipeline(jacobiSmoothPL);
      vp.setBindGroup(0, jacobiFineBG[js]);
      vp.dispatchWorkgroups(WG_UPDATE);
      vp.end();
    }

    // --- V-cycle coarse correction every POISSON_INTERVAL steps ---
    if (vcycleEnabled && s > 0 && s % POISSON_INTERVAL === 0) {
      vcycleCount++;
      // Residual from P[0]
      vp = enc.beginComputePass();
      vp.setPipeline(computeResidualPL);
      vp.setBindGroup(0, residualBG[0]);
      vp.dispatchWorkgroups(WG_UPDATE);
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
      vp.dispatchWorkgroups(WG_UPDATE);
      vp.end();
    }

    // --- U update (Neumann BC from labels) ---
    let cp = enc.beginComputePass();
    cp.setPipeline(updatePL);
    cp.setBindGroup(0, updateBG[cur]);
    cp.dispatchWorkgroups(WG_UPDATE);
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

    // Nuclear force computation at N_MOVE intervals — gradient of P directly
    if (dynamicsEnabled && (tStep + s + 1) % N_MOVE === 0) {
      cp = enc.beginComputePass();
      cp.setPipeline(gradPtotalPL);
      cp.setBindGroup(0, gradPtotalBG[cur]);
      cp.dispatchWorkgroups(NELEC);
      cp.end();
      needForceReadback = true;
    }
  }

  // --- Compute Pother = P_total - P_self (remove self-repulsion) ---
  enc.copyBufferToBuffer(P_buf[0], 0, PotherBuf, 0, S3 * 4);
  // Only run SIC periodically to avoid GPU timeout with many atoms
  if (frameCount > 0 && frameCount % SIC_INTERVAL === 0) {
    for (let m = 0; m < NELEC; m++) {
      if (Z[m] === 0) continue;
      let sp = enc.beginComputePass();
      sp.setPipeline(computeRhoSelfPL);
      sp.setBindGroup(0, computeRhoSelfBG[m][cur]);
      sp.dispatchWorkgroups(WG_UPDATE);
      sp.end();
      enc.copyBufferToBuffer(P_buf[0], 0, P_buf[1], 0, S3 * 4);
      for (let js = 0; js < SIC_JACOBI; js++) {
        sp = enc.beginComputePass();
        sp.setPipeline(jacobiSmoothPL);
        sp.setBindGroup(0, jacobiSelfBG[js % 2]);
        sp.dispatchWorkgroups(WG_UPDATE);
        sp.end();
      }
      sp = enc.beginComputePass();
      sp.setPipeline(subtractPselfPL);
      sp.setBindGroup(0, subtractPselfBG[m]);
      sp.dispatchWorkgroups(WG_UPDATE);
      sp.end();
    }
  }

  // --- Evolve level set W + flip labels where W < 0 ---
  frameCount++;
  for (let s = 0; s < W_STEPS_PER_FRAME; s++) {
    let bp = enc.beginComputePass();
    bp.setPipeline(evolveBoundaryPL);
    bp.setBindGroup(0, evolveBoundaryBG[cur]);
    bp.dispatchWorkgroups(WG_UPDATE);
    bp.end();
    // Fix U at flipped cells for density continuity (before copying new labels)
    bp = enc.beginComputePass();
    bp.setPipeline(fixBoundaryUPL);
    bp.setBindGroup(0, fixBoundaryUBG[cur]);
    bp.dispatchWorkgroups(WG_UPDATE);
    bp.end();
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

  await sumsReadBuf.mapAsync(GPUMapMode.READ);
  const sumsData = new Float32Array(sumsReadBuf.getMappedRange().slice(0));
  sumsReadBuf.unmap();
  E_T = sumsData[NELEC];
  E_eK = sumsData[NELEC + 1];
  E_ee = sumsData[NELEC + 2];

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

  // Force readback and nuclear dynamics
  if (needForceReadback) {
    await forceSumsReadBuf.mapAsync(GPUMapMode.READ);
    const forceData = new Float32Array(forceSumsReadBuf.getMappedRange().slice(0));
    forceSumsReadBuf.unmap();
    await moveNuclei(forceData);
  }
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
        nucVel[a][d] += nucForce[a][d] / m * DT_NUC;
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

  // Update atomBuf with new positions, recompute K on GPU (keep converged U, labels, P)
  fillAtomBuf();
  const enc = device.createCommandEncoder();
  const cp = enc.beginComputePass();
  cp.setPipeline(recomputeK_PL);
  cp.setBindGroup(0, recomputeK_BG);
  cp.dispatchWorkgroups(WG_RECOMPUTE_K);
  cp.end();
  device.queue.submit([enc.finish()]);
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
    doSteps(STEPS_PER_FRAME).then(() => {
      computing = false;
      phaseSteps += STEPS_PER_FRAME;
      if (isFinite(E) && E < E_min) E_min = E;

      // Dynamics off by default — press D to enable

      if (phaseSteps >= TOTAL_STEPS) {
        console.log("=== DONE: E=" + E.toFixed(6) + " ===");
        phase = 1;  // done
        if (window.onSweepDone) window.onSweepDone(E_min);
      }
    }).catch((e) => {
      gpuError = e.message || String(e);
      console.error("GPU step failed:", e);
      computing = false;
    });
  }

  if (sliceData) {
    const SS = S;
    const SS2 = SS * SS;
    loadPixels();
    const d = pixelDensity();
    const W = 400 * d, H = 400 * d;
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

    // Line density plot: total density along center row + K potential
    const row = N2;
    const plotH = 60;
    const yBase = 390;
    let maxV = 0;
    for (let i = 1; i < NN; i++) {
      const v = sliceData[i * SS + row];
      if (v > maxV) maxV = v;
    }
    if (maxV > 0) {
      const sc = plotH / maxV;
      stroke(60); strokeWeight(1);
      line(0, yBase, 400, yBase);
      stroke(0, 255, 255, 180); noFill();
      beginShape();
      for (let i = 1; i < NN; i++) {
        vertex(PX * i, yBase - sliceData[i * SS + row] * sc);
      }
      endShape();
      noStroke(); fill(255);
      text("density j=" + row, 5, yBase - plotH - 2);
    }
    // K potential line
    let maxK = 0;
    for (let i = 1; i < NN; i++) {
      const v = sliceData[3 * SS2 + i];
      if (v > maxK) maxK = v;
    }
    if (maxK > 0) {
      const yBase2 = yBase - plotH - 20;
      const sc2 = 40 / maxK;
      stroke(60); strokeWeight(1);
      line(0, yBase2, 400, yBase2);
      stroke(255, 128, 0, 180); noFill();
      beginShape();
      for (let i = 1; i < NN; i++) {
        vertex(PX * i, yBase2 - sliceData[3 * SS2 + i] * sc2);
      }
      endShape();
      noStroke(); fill(255);
      text("K potential", 5, yBase2 - 42);
    }
  }

  // Draw nuclear positions with force arrows
  fill(255); stroke(255); strokeWeight(1);
  for (let n = 0; n < NELEC; n++) {
    if (Z[n] > 0) {
      circle(nucPos[n][0] * PX, nucPos[n][1] * PX, 6);
      // Draw force arrows when dynamics enabled
      if (dynamicsEnabled && nucForce[n]) {
        const fx = nucForce[n][0], fy = nucForce[n][1];
        const fmag = Math.sqrt(fx*fx + fy*fy);
        if (fmag > 1e-8) {
          const arrowScale = 5000;
          const ax = nucPos[n][0] * PX + fx * arrowScale;
          const ay = nucPos[n][1] * PX + fy * arrowScale;
          stroke(0, 255, 0); strokeWeight(1);
          line(nucPos[n][0] * PX, nucPos[n][1] * PX, ax, ay);
        }
      }
    }
  }

  // Bond lengths
  if (dynamicsEnabled) {
    noStroke(); fill(180, 255, 180);
    for (let a = 0; a < NELEC; a++) {
      for (let b = a + 1; b < NELEC; b++) {
        if (Z[a] === 0 || Z[b] === 0) continue;
        const dx = (nucPos[a][0] - nucPos[b][0]) * hGrid;
        const dy = (nucPos[a][1] - nucPos[b][1]) * hGrid;
        const dz = (nucPos[a][2] - nucPos[b][2]) * hGrid;
        const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
        const mx = (nucPos[a][0] + nucPos[b][0]) * 0.5 * PX;
        const my = (nucPos[a][1] + nucPos[b][1]) * 0.5 * PX;
        text(dist.toFixed(2), mx, my);
      }
    }
  }

  // Screen boundary
  noFill(); stroke(100); strokeWeight(1);
  rect(0, 0, 400, 400);
  noStroke();

  fill(255);
  const labels = atomLabels.map((el, i) => [el, Z_orig[i]]).filter(x => x[1] > 0).map(x => x[0] + "(Z=" + x[1] + ")").join(" ");
  const pLabel = phase === 0 ? "running" : "DONE";
  text("Molecule: " + labels + " | " + screenAu + " au | " + pLabel + " | " + NN + "^3", 5, 20);
  text("step " + tStep + " (" + phaseSteps + "/" + TOTAL_STEPS + ")  E=" + E.toFixed(6) + "  E_min=" + E_min.toFixed(6), 5, 35);
  if (lastMs > 0) text((lastMs / STEPS_PER_FRAME).toFixed(1) + "ms/step", 300, 35);

  fill(200);
  text("T=" + E_T.toFixed(4) + " V_eK=" + E_eK.toFixed(4) + " V_ee=" + E_ee.toFixed(4) + " V_KK=" + E_KK.toFixed(4), 5, 50);
  fill(vcycleEnabled ? [0,255,0] : [255,100,0]);
  text("V-cycle: " + (vcycleEnabled ? "ON" : "OFF") + " (" + vcycleCount + " cycles)  [press V to toggle]", 5, 65);

  // Dynamics status
  fill(dynamicsEnabled ? [0,255,255] : [150,150,150]);
  text("Dynamics: " + (dynamicsEnabled ? "ON" : "OFF") + " (nucStep=" + nucStepCount + ")  [press D to toggle]", 5, 80);
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
}
