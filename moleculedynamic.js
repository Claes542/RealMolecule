// Molecule Quantum Simulation — WebGPU Compute Shaders
// Up to 10 atoms placed interactively, 3D geometry

const NN = window.USER_NN || 200;
const S = NN + 1;
const S2 = S * S;
const S3 = S * S * S;
const N2 = Math.round(NN / 2);
const MAX_ATOMS = 400;
const _uz = window.USER_Z || [2, 3, 1, 0, 0];
while (_uz.length < MAX_ATOMS) _uz.push(0);
const NELEC = _uz.filter(z => z > 0).length || 3;
const NRED = 3;  // T + V_eK + V_ee
const r_cut = window.USER_RC || [0.5, 0.3, 0.1, 0, 0];
while (r_cut.length < MAX_ATOMS) r_cut.push(0);
const _uzn = window.USER_ZN || [..._uz];  // nuclear charges (default = electron Z)
while (_uzn.length < MAX_ATOMS) _uzn.push(0);
let R_out = window.USER_ROUT || 2.0;   // au, outer w cutoff
let Z = [..._uz];
let Zn = [..._uzn];
let Ne = [..._uz];
const Z_orig = [..._uz];
const Zn_orig = [..._uzn];
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
// Nuclear velocities (grid units per frame) and dynamics
let nucVel = _atoms.map(() => [0, 0, 0]);
const NUC_DT = 0.002;     // nuclear timestep (au)
const NUC_FRICTION = 0.95; // damping to find equilibrium
const FORCES_BYTES = MAX_ATOMS * 3 * 4;
let screenAu = window.USER_SCREEN || 10;
let hv = screenAu / NN, h2v = hv * hv, h3v = hv * hv * hv;
const dv = 0.12;
let dtv = dv * h2v, half_dv = 0.5 * dv;
const PX = 400 / NN;
const INTERIOR = (NN - 1) * (NN - 1) * (NN - 1);
const STEPS_PER_FRAME = 10;
const W_STEPS_PER_FRAME = 10;
const NORM_INTERVAL = 20;

// Grid constants

// Reduce workgroup size: NRED * REDUCE_WG * 4 must be <= 16384 bytes
const REDUCE_WG = 1 << Math.floor(Math.log2(Math.min(128, Math.floor(4096 / NRED))));

// ===== WGSL SHADERS =====

// Param struct: 16 common fields = 64 bytes. Atom data in separate storage buffer.
const PARAM_BYTES = 64;
const paramStructWGSL = `
struct P {
  NN: u32, S: u32, S2: u32, S3: u32,
  N2: u32, NATOM: u32, R_out: f32, TWO_PI: f32,
  h: f32, h2: f32, inv_h: f32, inv_h2: f32,
  dt: f32, half_d: f32, h3: f32, _pad0: f32,
}`;

const ATOM_STRIDE = 8; // 8 f32s per atom (posI, posJ, posK, Z, rc, pad, pad, pad)
const ATOM_BUF_BYTES = MAX_ATOMS * ATOM_STRIDE * 4;
const atomStructWGSL = `
struct Atom {
  posI: u32, posJ: u32, posK: u32, Z: f32,
  rc: f32, Zn: f32, _p1: f32, _p2: f32,
}`;

// U update — Ws (smooth W) for face weights, Wd (sharp W) for support
const updateU_WGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> Ui: array<f32>;
@group(0) @binding(3) var<storage, read> Ws: array<f32>;
@group(0) @binding(4) var<storage, read> Pi: array<f32>;
@group(0) @binding(5) var<storage, read_write> Uo: array<f32>;
@group(0) @binding(6) var<storage, read> atoms: array<Atom>;
@group(0) @binding(7) var<storage, read> Wd: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }

  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  // Sharp W defines support — U = 0 outside support
  if (Wd[id] < 0.5) { Uo[id] = 0.0; return; }

  let uc = Ui[id];

  // Smooth W for face weights
  let wc = Ws[id];

  let w_ip = Ws[id + p.S2]; let w_im = Ws[id - p.S2];
  let w_jp = Ws[id + p.S];  let w_jm = Ws[id - p.S];
  let w_kp = Ws[id + 1u];   let w_km = Ws[id - 1u];

  let u_ip = Ui[id + p.S2];
  let u_im = Ui[id - p.S2];
  let u_jp = Ui[id + p.S];
  let u_jm = Ui[id - p.S];
  let u_kp = Ui[id + 1u];
  let u_km = Ui[id - 1u];

  // Arithmetic mean face weights from smooth W
  let lap = (u_ip - uc) * (w_ip + wc) * 0.5 - (uc - u_im) * (wc + w_im) * 0.5
          + (u_jp - uc) * (w_jp + wc) * 0.5 - (uc - u_jm) * (wc + w_jm) * 0.5
          + (u_kp - uc) * (w_kp + wc) * 0.5 - (uc - u_km) * (wc + w_km) * 0.5;

  Uo[id] = uc + p.half_d * lap + p.dt * (K[id] - 2.0 * Pi[id]) * uc * wc;
}
`;

// W update — level set evolution with density-driven advection (no Voronoi labels)
// cm > 0 where electron density present (W expands), cm < 0 where absent (W contracts)
const updateW_WGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Wi: array<f32>;
@group(0) @binding(2) var<storage, read_write> Wo: array<f32>;
@group(0) @binding(3) var<storage, read> Ui: array<f32>;
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

  // Density-driven charge concentration: expand where electron present, contract where absent
  let myU = Ui[id];
  let cm = clamp(sign(myU * myU - 1e-6), -1.0, 1.0);

  // W neighbors
  let wc  = Wi[id];
  let wip = Wi[id + p.S2]; let wim = Wi[id - p.S2];
  let wjp = Wi[id + p.S];  let wjm = Wi[id - p.S];
  let wkp = Wi[id + 1u];   let wkm = Wi[id - 1u];

  // Level set evolution: curvature flow + charge-driven advection
  let gx = (wip - wim) * p.inv_h;
  let gy = (wjp - wjm) * p.inv_h;
  let gz = (wkp - wkm) * p.inv_h;
  let gradMag = sqrt(gx * gx + gy * gy + gz * gz);

  let lw = (wip + wim + wjp + wjm + wkp + wkm - 6.0 * wc);
  var nw = wc + 5.0 * p.dt * cm * gradMag + 0.1 * lw;
  nw = clamp(nw, 0.0, 1.0);

  // Smooth cutoff near nuclear cores
  for (var b: u32 = 0u; b < p.NATOM; b++) {
    let ab = atoms[b];
    if (ab.Zn < 0.5) { continue; }
    let di = f32(i) - f32(ab.posI);
    let dj = f32(j) - f32(ab.posJ);
    let dk = f32(k) - f32(ab.posK);
    let r = sqrt(di*di + dj*dj + dk*dk) * p.h;
    let rc = ab.rc;
    if (r < rc) {
      let edge = rc - 3.0 * p.h;
      let t = clamp((r - edge) / (rc - edge), 0.0, 1.0);
      nw = min(nw, t * t * (3.0 - 2.0 * t));
    }
  }

  Wo[id] = nw;
}
`;

// Compute sharp W (Wd) from smooth W (Ws): defines support of U
// Interior (Ws > 0.5): Wd = 1. Exterior (Ws < threshold): Wd = 0.
// Transition: assign to 1 if this cell has significant U, else 0.
const computeSharpWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Ws: array<f32>;
@group(0) @binding(2) var<storage, read> U: array<f32>;
@group(0) @binding(3) var<storage, read_write> Wd: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }
  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  let ws = Ws[id];

  // Clear interior/exterior
  if (ws > 0.99) { Wd[id] = 1.0; return; }
  if (ws < 0.01) { Wd[id] = 0.0; return; }

  // Transition zone: assign to support if U is present here or in nearby same-side cells
  let u2 = U[id] * U[id];
  // Check if any neighbor with higher Ws has significant U
  var maxU2: f32 = u2;
  if (Ws[id + p.S2] > ws) { maxU2 = max(maxU2, U[id + p.S2] * U[id + p.S2]); }
  if (Ws[id - p.S2] > ws) { maxU2 = max(maxU2, U[id - p.S2] * U[id - p.S2]); }
  if (Ws[id + p.S]  > ws) { maxU2 = max(maxU2, U[id + p.S]  * U[id + p.S]); }
  if (Ws[id - p.S]  > ws) { maxU2 = max(maxU2, U[id - p.S]  * U[id - p.S]); }
  if (Ws[id + 1u]   > ws) { maxU2 = max(maxU2, U[id + 1u]   * U[id + 1u]); }
  if (Ws[id - 1u]   > ws) { maxU2 = max(maxU2, U[id - 1u]   * U[id - 1u]); }

  Wd[id] = select(0.0, 1.0, maxU2 > 1e-8);
}
`;

// Update potential K: nuclear attraction + electron repulsion from all other atom densities
// V(r) = sum_b Z_b/r_b (nuclear) - sum_{b!=myLabel} Z_b/r_b (electron repulsion as point charges)
const updatePotentialWGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> K: array<f32>;
@group(0) @binding(2) var<storage, read> atoms: array<Atom>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }
  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  var V: f32 = 0.0;
  let soft = 0.04 * p.h2;
  for (var b: u32 = 0u; b < p.NATOM; b++) {
    let ab = atoms[b];
    if (ab.Z < 0.5 && ab.Zn < 0.5) { continue; }
    let di = (f32(i) - f32(ab.posI)) * p.h;
    let dj = (f32(j) - f32(ab.posJ)) * p.h;
    let dk = (f32(k) - f32(ab.posK)) * p.h;
    let r = sqrt(di*di + dj*dj + dk*dk + soft);
    let ir = 1.0 / r;

    // Smooth cutoff near core
    var sc: f32 = 1.0;
    if (r < ab.rc) {
      let edge = ab.rc - 3.0 * p.h;
      let t = clamp((r - edge) / (ab.rc - edge), 0.0, 1.0);
      sc = t * t * (3.0 - 2.0 * t);
    }

    // Nuclear attraction using Zn (nuclear charge)
    V += ab.Zn * ir * sc;
  }
  K[id] = V;
}
`;

// Recompute nearest-atom labels after nuclear motion
const updateLabelsWGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> atoms: array<Atom>;
@group(0) @binding(2) var<storage, read_write> label: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }
  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;

  var bestD: f32 = 1e20;
  var best: u32 = 0u;
  for (var b: u32 = 0u; b < p.NATOM; b++) {
    let ab = atoms[b];
    if (ab.Z < 0.5) { continue; }
    let di = f32(i) - f32(ab.posI);
    let dj = f32(j) - f32(ab.posJ);
    let dk = f32(k) - f32(ab.posK);
    let d2 = di*di + dj*dj + dk*dk;
    if (d2 < bestD) { bestD = d2; best = b; }
  }
  let id = i * p.S2 + j * p.S + k;
  label[id] = best;
}
`;

// Compute forces on nuclei: gradient of Hartree potential P at nuclear positions
// Force from electrons on nucleus a: F_a = Z_a * grad(P)
// One thread per atom, tiny dispatch
const computeForcesWGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Pv: array<f32>;
@group(0) @binding(2) var<storage, read> atoms: array<Atom>;
@group(0) @binding(3) var<storage, read_write> forces: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let a = gid.x;
  if (a >= p.NATOM) { return; }
  let at = atoms[a];
  if (at.Zn < 0.5) {
    forces[a * 3u] = 0.0;
    forces[a * 3u + 1u] = 0.0;
    forces[a * 3u + 2u] = 0.0;
    return;
  }
  let i = at.posI;
  let j = at.posJ;
  let k = at.posK;
  let id = i * p.S2 + j * p.S + k;
  let inv2h = 0.5 * p.inv_h;

  // Central difference gradient of Hartree potential P
  let dPx = Pv[id + p.S2] - Pv[id - p.S2];
  let dPy = Pv[id + p.S]  - Pv[id - p.S];
  let dPz = Pv[id + 1u]   - Pv[id - 1u];

  // Electron force on nucleus: F = Zn * grad(P)
  forces[a * 3u]      = at.Zn * dPx * inv2h;
  forces[a * 3u + 1u] = at.Zn * dPy * inv2h;
  forces[a * 3u + 2u] = at.Zn * dPz * inv2h;
}
`;

// Compute total electron density from U, W, labels, atoms
const computeRhoWGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read_write> rho: array<f32>;
@group(0) @binding(3) var<storage, read> atoms: array<Atom>;
@group(0) @binding(4) var<storage, read> label: array<u32>;
@group(0) @binding(5) var<storage, read> W: array<f32>;

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
  rho[id] = atoms[label[id]].Z * u * u;
}
`;

// Jacobi smoother for Poisson: ∇²P = -4π ρ
const jacobiWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Pin: array<f32>;
@group(0) @binding(2) var<storage, read_write> Pout: array<f32>;
@group(0) @binding(3) var<storage, read> rho: array<f32>;

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
  let rhs = p.h2 * p.TWO_PI * rho[id];
  Pout[id] = 0.3333 * Pc + (sum_nbr + rhs) / 9.0;
}
`;

// Atomic per-atom norm accumulation — each thread stores val+label, thread 0 reduces and CAS
const NORM_WG = 256;
const atomicNormWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> label: array<u32>;
@group(0) @binding(4) var<storage, read_write> norms: array<atomic<u32>>;

var<workgroup> wgVal: array<f32, ${NORM_WG}>;
var<workgroup> wgLbl: array<u32, ${NORM_WG}>;

@compute @workgroup_size(${NORM_WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;

  wgVal[lid] = 0.0;
  wgLbl[lid] = 0xFFFFFFFFu;

  if (gid.x < tot) {
    let k = (gid.x % NM) + 1u;
    let j = ((gid.x / NM) % NM) + 1u;
    let i = (gid.x / (NM * NM)) + 1u;
    let id = i * p.S2 + j * p.S + k;
    let v = U[id];
    let val = v * v * p.h3;
    if (val > 0.0) {
      wgVal[lid] = val;
      wgLbl[lid] = label[id];
    }
  }
  workgroupBarrier();

  // Thread 0 reduces all 256 values by label, then one CAS per label
  if (lid == 0u) {
    // Collect unique labels and sum
    var labels: array<u32, 16>;
    var sums: array<f32, 16>;
    var cnt: u32 = 0u;
    for (var t: u32 = 0u; t < ${NORM_WG}u; t++) {
      let lbl = wgLbl[t];
      if (lbl == 0xFFFFFFFFu) { continue; }
      var found = false;
      for (var s: u32 = 0u; s < cnt; s++) {
        if (labels[s] == lbl) { sums[s] += wgVal[t]; found = true; break; }
      }
      if (!found && cnt < 16u) {
        labels[cnt] = lbl;
        sums[cnt] = wgVal[t];
        cnt++;
      }
    }
    // CAS each label sum to global
    for (var s: u32 = 0u; s < cnt; s++) {
      var old_bits = atomicLoad(&norms[labels[s]]);
      loop {
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + sums[s];
        let new_bits = bitcast<u32>(new_val);
        let result = atomicCompareExchangeWeak(&norms[labels[s]], old_bits, new_bits);
        if (result.exchanged) { break; }
        old_bits = result.old_value;
      }
    }
  }
}
`;

// Energy reduce: NRED=3 (T, V_eK, V_ee)
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
@group(0) @binding(7) var<storage, read> W: array<f32>;

var<workgroup> sn: array<f32, ${NRED * REDUCE_WG}>;

@compute @workgroup_size(${REDUCE_WG})
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

    let v = U[id];
    let Zm = atoms[label[id]].Z;

    // Kinetic energy
    let v_ip = U[id + p.S2];
    let v_jp = U[id + p.S];
    let v_kp = U[id + 1u];
    let a = v_ip - v; let b = v_jp - v; let c = v_kp - v;
    sn[lid * ${NRED}u + 0u] = Zm * 0.5 * (a * a + b * b + c * c) * p.h;
    sn[lid * ${NRED}u + 1u] = -Zm * K[id] * v * v * p.h3;
    sn[lid * ${NRED}u + 2u] = Zm * Pv[id] * v * v * p.h3;
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
@group(0) @binding(2) var<storage, read> norms: array<u32>;
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

  let n = bitcast<f32>(norms[label[id]]);
  if (n > 0.0) { U[id] *= inverseSqrt(n); }
}
`;

// Extract: 3 planes — density (u²), label, boundary
const extractWGSL = `
${paramStructWGSL}
${atomStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> label: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> W: array<f32>;
@group(0) @binding(5) var<storage, read> atoms: array<Atom>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  let j = g.y;
  let SS = p.NN + 1u;
  if (i > p.NN || j > p.NN) { return; }

  let idx = i * p.S2 + j * p.S + p.N2;
  // Plane offsets: 0=density, 1=label(as Z), 2=boundary
  let off0 = i * SS + j;
  let off1 = SS * SS + i * SS + j;
  let off2 = 2u * SS * SS + i * SS + j;

  if (i < 1u || i >= p.NN || j < 1u || j >= p.NN) {
    out[off0] = 0.0; out[off1] = 0.0; out[off2] = 0.0;
    return;
  }
  let u = U[idx];
  let lbl = label[idx];
  out[off0] = u * u;
  out[off1] = atoms[lbl].Z;  // store Z for CPU coloring

  // Boundary: density edge + inter-atom
  let thr = 1e-6;
  let phi2 = u * u;
  let here = phi2 > thr;
  var bnd = 0.0;
  if (i > 1u && i < p.NN - 1u) {
    let phiR = U[idx + p.S2];
    let phiR2 = phiR * phiR;
    let lblR = label[idx + p.S2];
    if (here != (phiR2 > thr)) { bnd = 1.0; }
    if (lbl != lblR && phi2 > thr && phiR2 > thr) { bnd = 1.0; }
  }
  if (j > 1u && j < p.NN - 1u) {
    let phiD = U[idx + p.S];
    let phiD2 = phiD * phiD;
    let lblD = label[idx + p.S];
    if (here != (phiD2 > thr)) { bnd = 1.0; }
    if (lbl != lblD && phi2 > thr && phiD2 > thr) { bnd = 1.0; }
  }
  out[off2] = bnd;
}
`;

// ===== GPU STATE =====
let device, paramsBuf, atomBuf, K_buf, sumsBuf, sumsReadBuf, sliceBuf, sliceReadBuf, partialsBuf, numWGBuf, normsBuf, forcesBuf, forcesReadBuf;
let U_buf = [], P_buf = [], labelBuf, W_buf = [], Wd_buf, rhoBuf;
let updatePL, updateWPL, updatePotentialPL, updateLabelsPL, computeRhoPL, jacobiPL, computeForcesPL, reducePL, finalizePL, normalizePL, extractPL, atomicNormPL, computeSharpPL;
let updateBG = [], updateWBG = [], reduceBG = [], finalizeBG, normalizeBG = [], extractBG = [];
let atomicNormBG = [], updatePotentialBG, updateLabelsBG, computeRhoBG = [], jacobiBG = [], computeForcesBG, computeSharpBG = [];
let cur = 0, gpuReady = false, computing = false;
let tStep = 0, E = 0, lastMs = 0;
let E_T = 0, E_eK = 0, E_ee = 0, E_KK = 0;
let gpuError = null;

// Single phase run
let phase = 0, phaseSteps = 0;
const TOTAL_STEPS = window.USER_STEPS || 20000;
let addNucRepulsion = true;

const SLICE_SIZE = 3 * S * S * 4;  // 3 planes: density, Z, boundary
const WG_UPDATE = Math.ceil(INTERIOR / 256);
const WG_REDUCE = Math.ceil(INTERIOR / REDUCE_WG);
const WG_NORM = Math.ceil(INTERIOR / 256);
const WG_EXTRACT = Math.ceil(S / 16);
const SUMS_BYTES = NRED * 4;

let sliceData = null;
let lastForces = null;  // for display

function smoothCut(r, rc) {
  if (r >= rc) return 1;
  const edge = rc - 3 * hv;
  const t = Math.max(0, Math.min(1, (r - edge) / (rc - edge)));
  return t * t * (3 - 2 * t);
}

function fillParamsBuf(pb) {
  const pu = new Uint32Array(pb);
  const pf = new Float32Array(pb);
  pu[0] = NN; pu[1] = S; pu[2] = S2; pu[3] = S3;
  pu[4] = N2; pu[5] = NELEC; pf[6] = R_out; pf[7] = 2 * Math.PI;
  pf[8] = hv; pf[9] = h2v; pf[10] = 1 / hv; pf[11] = 1 / h2v;
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
    af[off + 5] = Zn[n];
  }
  device.queue.writeBuffer(atomBuf, 0, ab);
}

function uploadInitialData() {
  console.log("Init: nuclei at", nucPos.map((p,i) => i+"=("+p+")").join(" "));

  const Kd = new Float32Array(S3);
  const Ud = new Float32Array(S3);
  const Ld = new Uint32Array(S3);
  const Wd = new Float32Array(S3);
  const soft = 0.04 * h2v;
  const NA = NELEC;
  const SMOOTH_WIDTH = 3.0 * hv;

  for (let i = 0; i <= NN; i++) {
    const dx = [];
    for (let n = 0; n < NA; n++) dx[n] = (i - nucPos[n][0]) * hv;
    for (let j = 0; j <= NN; j++) {
      const dy = [];
      for (let n = 0; n < NA; n++) dy[n] = (j - nucPos[n][1]) * hv;
      for (let k = 0; k <= NN; k++) {
        const dz = [];
        for (let n = 0; n < NA; n++) dz[n] = (k - nucPos[n][2]) * hv;
        const id = i * S2 + j * S + k;

        const r = [], ir = [], u = [];
        for (let n = 0; n < NA; n++) {
          r[n] = Math.sqrt(dx[n]*dx[n] + dy[n]*dy[n] + dz[n]*dz[n] + soft);
          ir[n] = 1 / r[n];
          u[n] = (r[n] > r_cut[n] && r[n] < R_out) ? Math.exp(-r[n]) : 0.0;
        }

        let Kval = 0;
        for (let n = 0; n < NA; n++) Kval += Zn[n] * ir[n] * smoothCut(r[n], r_cut[n]);
        Kd[id] = Kval;

        // Assign to nearest atom (simple distance)
        let best = -1, bestD = Infinity;
        for (let n = 0; n < NA; n++) {
          if (Z[n] > 0 && r[n] < bestD) { bestD = r[n]; best = n; }
        }
        if (best >= 0) {
          Ld[id] = best;
          const rb = r[best];
          // Find distance to nearest OTHER atom for boundary gradient
          let secondD = Infinity;
          for (let n = 0; n < NA; n++) {
            if (n !== best && Z[n] > 0 && r[n] < secondD) secondD = r[n];
          }
          // W transitions from 1 (deep inside) to 0 (at Voronoi boundary)
          // Voronoi boundary is where rb == secondD; margin = (secondD - rb)
          const margin = secondD - rb;
          const bw = 3.0 * hv; // boundary transition width
          let wVal = (rb > r_cut[best] && rb < R_out) ? 1.0 : 0.0;
          if (margin < bw && wVal > 0) {
            const t = Math.max(0, margin / bw);
            wVal = t * t * (3 - 2 * t); // smooth Hermite
          }
          Wd[id] = wVal;
          Ud[id] = u[best];  // store u directly
        }

      }
    }
  }

  // Initialize sharp W (Wd): 1 where U > 0, 0 elsewhere
  const WdSharp = new Float32Array(S3);
  for (let id = 0; id < S3; id++) {
    WdSharp[id] = (Ud[id] !== 0) ? 1.0 : 0.0;
  }

  console.log("Uploading to GPU...");
  device.queue.writeBuffer(K_buf, 0, Kd);
  device.queue.writeBuffer(labelBuf, 0, Ld);
  for (let i = 0; i < 2; i++) device.queue.writeBuffer(W_buf[i], 0, Wd);
  device.queue.writeBuffer(Wd_buf, 0, WdSharp);
  for (let i = 0; i < 2; i++) {
    device.queue.writeBuffer(U_buf[i], 0, Ud);
  }
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
  R_out = window.USER_ROUT || 2.0;
  addNucRepulsion = true;
  updateParamsBuf();
  uploadInitialData();
  tStep = 0;
  phaseSteps = 0;
  E_min = Infinity;
  cur = 0;
  phase = 0;
  console.log("=== Molecule: " + atomLabels.join("-") + " ===");
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
    console.log("WebGPU device ready, maxStorage=" + device.limits.maxStorageBufferBindingSize);

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
    labelBuf = device.createBuffer({ size: bs, usage: usage });
    for (let i = 0; i < 2; i++) {
      W_buf[i] = device.createBuffer({ size: bs, usage: usage });
    }
    Wd_buf = device.createBuffer({ size: bs, usage: usage });
    rhoBuf = device.createBuffer({ size: bs, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    forcesBuf = device.createBuffer({ size: FORCES_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    forcesReadBuf = device.createBuffer({ size: FORCES_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    const partialSize = WG_REDUCE * NRED * 4;
    partialsBuf = device.createBuffer({ size: partialSize, usage: GPUBufferUsage.STORAGE });
    sumsBuf = device.createBuffer({ size: SUMS_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    sumsReadBuf = device.createBuffer({ size: SUMS_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    // Per-atom norms via atomicAdd (u32 for atomic ops)
    const normsBufSize = Math.max(MAX_ATOMS * 4, 16);
    normsBuf = device.createBuffer({ size: normsBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    sliceBuf = device.createBuffer({ size: SLICE_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    sliceReadBuf = device.createBuffer({ size: SLICE_SIZE, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    numWGBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(numWGBuf, 0, new Uint32Array([WG_REDUCE, 0, 0, 0]));

    const pb = new ArrayBuffer(PARAM_BYTES);
    fillParamsBuf(pb);
    paramsBuf = device.createBuffer({ size: PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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

    const atomicNormMod = await compileShader('atomicNorm', atomicNormWGSL);
    const updateMod = await compileShader('updateU', updateU_WGSL);
    const updateWMod = await compileShader('updateW', updateW_WGSL);
    const updatePotentialMod = await compileShader('updatePotential', updatePotentialWGSL);
    const updateLabelsMod = await compileShader('updateLabels', updateLabelsWGSL);
    const computeRhoMod = await compileShader('computeRho', computeRhoWGSL);
    const jacobiMod = await compileShader('jacobi', jacobiWGSL);
    const computeForcesMod = await compileShader('computeForces', computeForcesWGSL);
    const reduceMod = await compileShader('reduce', reduceWGSL);
    const finalizeMod = await compileShader('finalize', finalizeWGSL);
    const normalizeMod = await compileShader('normalize', normalizeWGSL);
    const extractMod = await compileShader('extract', extractWGSL);
    const computeSharpMod = await compileShader('computeSharp', computeSharpWGSL);

    atomicNormPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: atomicNormMod, entryPoint: 'main' } });
    updatePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: updateMod, entryPoint: 'main' } });
    updateWPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: updateWMod, entryPoint: 'main' } });
    updatePotentialPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: updatePotentialMod, entryPoint: 'main' } });
    updateLabelsPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: updateLabelsMod, entryPoint: 'main' } });
    computeRhoPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: computeRhoMod, entryPoint: 'main' } });
    jacobiPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: jacobiMod, entryPoint: 'main' } });
    computeForcesPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: computeForcesMod, entryPoint: 'main' } });
    reducePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: reduceMod, entryPoint: 'main' } });
    finalizePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: finalizeMod, entryPoint: 'main' } });
    normalizePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: normalizeMod, entryPoint: 'main' } });
    extractPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: extractMod, entryPoint: 'main' } });
    computeSharpPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: computeSharpMod, entryPoint: 'main' } });

    for (let c = 0; c < 2; c++) {
      const n = 1 - c;
      updateBG[c] = device.createBindGroup({ layout: updatePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: K_buf } },
        { binding: 2, resource: { buffer: U_buf[c] } },
        { binding: 3, resource: { buffer: W_buf[0] } },
        { binding: 4, resource: { buffer: P_buf[0] } },
        { binding: 5, resource: { buffer: U_buf[n] } },
        { binding: 6, resource: { buffer: atomBuf } },
        { binding: 7, resource: { buffer: Wd_buf } },
      ]});
      reduceBG[c] = device.createBindGroup({ layout: reducePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: labelBuf } },
        { binding: 3, resource: { buffer: P_buf[0] } },
        { binding: 4, resource: { buffer: K_buf } },
        { binding: 5, resource: { buffer: partialsBuf } },
        { binding: 6, resource: { buffer: atomBuf } },
        { binding: 7, resource: { buffer: W_buf[0] } },
      ]});
      computeRhoBG[c] = device.createBindGroup({ layout: computeRhoPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: rhoBuf } },
        { binding: 3, resource: { buffer: atomBuf } },
        { binding: 4, resource: { buffer: labelBuf } },
        { binding: 5, resource: { buffer: W_buf[0] } },
      ]});
      atomicNormBG[c] = device.createBindGroup({ layout: atomicNormPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: W_buf[0] } },
        { binding: 3, resource: { buffer: labelBuf } },
        { binding: 4, resource: { buffer: normsBuf } },
      ]});
      normalizeBG[c] = device.createBindGroup({ layout: normalizePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: normsBuf } },
        { binding: 3, resource: { buffer: labelBuf } },
      ]});
      extractBG[c] = device.createBindGroup({ layout: extractPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: U_buf[c] } },
        { binding: 2, resource: { buffer: labelBuf } },
        { binding: 3, resource: { buffer: sliceBuf } },
        { binding: 4, resource: { buffer: W_buf[0] } },
        { binding: 5, resource: { buffer: atomBuf } },
      ]});
    }
    // Compute sharp W from smooth W + U
    for (let c = 0; c < 2; c++) {
      computeSharpBG[c] = device.createBindGroup({ layout: computeSharpPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: W_buf[0] } },
        { binding: 2, resource: { buffer: U_buf[c] } },
        { binding: 3, resource: { buffer: Wd_buf } },
      ]});
    }
    // W update: ping-pong level set at boundaries
    for (let d = 0; d < 2; d++) {
      for (let u = 0; u < 2; u++) {
        updateWBG[d * 2 + u] = device.createBindGroup({ layout: updateWPL.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: paramsBuf } },
          { binding: 1, resource: { buffer: W_buf[d] } },
          { binding: 2, resource: { buffer: W_buf[1 - d] } },
          { binding: 3, resource: { buffer: U_buf[u] } },
          { binding: 4, resource: { buffer: atomBuf } },
        ]});
      }
    }
    // Update potential: recompute K from atom positions
    updatePotentialBG = device.createBindGroup({ layout: updatePotentialPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: K_buf } },
      { binding: 2, resource: { buffer: atomBuf } },
    ]});
    updateLabelsBG = device.createBindGroup({ layout: updateLabelsPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: atomBuf } },
      { binding: 2, resource: { buffer: labelBuf } },
    ]});
    // Jacobi Poisson smoother: P[0]→P[1] and P[1]→P[0]
    jacobiBG[0] = device.createBindGroup({ layout: jacobiPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: P_buf[0] } },
      { binding: 2, resource: { buffer: P_buf[1] } },
      { binding: 3, resource: { buffer: rhoBuf } },
    ]});
    jacobiBG[1] = device.createBindGroup({ layout: jacobiPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: P_buf[1] } },
      { binding: 2, resource: { buffer: P_buf[0] } },
      { binding: 3, resource: { buffer: rhoBuf } },
    ]});
    // Force computation on nuclei
    computeForcesBG = device.createBindGroup({ layout: computeForcesPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: P_buf[0] } },
      { binding: 2, resource: { buffer: atomBuf } },
      { binding: 3, resource: { buffer: forcesBuf } },
    ]});

    finalizeBG = device.createBindGroup({ layout: finalizePL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: partialsBuf } },
      { binding: 1, resource: { buffer: sumsBuf } },
      { binding: 2, resource: { buffer: numWGBuf } },
    ]});

    console.log("Ready! dispatch(" + WG_UPDATE + ") direct potential sum");
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

  // Recompute nuclear potential from atom positions (once per frame)
  let cp = enc.beginComputePass();
  cp.setPipeline(updatePotentialPL);
  cp.setBindGroup(0, updatePotentialBG);
  cp.dispatchWorkgroups(WG_UPDATE);
  cp.end();

  for (let s = 0; s < n; s++) {
    const next = 1 - cur;

    // --- Compute electron density and Jacobi Poisson solve ---
    cp = enc.beginComputePass();
    cp.setPipeline(computeRhoPL);
    cp.setBindGroup(0, computeRhoBG[cur]);
    cp.dispatchWorkgroups(WG_UPDATE);
    cp.end();
    for (let js = 0; js < 2; js++) {
      cp = enc.beginComputePass();
      cp.setPipeline(jacobiPL);
      cp.setBindGroup(0, jacobiBG[js]);
      cp.dispatchWorkgroups(WG_UPDATE);
      cp.end();
    }

    // --- U update ---
    cp = enc.beginComputePass();
    cp.setPipeline(updatePL);
    cp.setBindGroup(0, updateBG[cur]);
    cp.dispatchWorkgroups(WG_UPDATE);
    cp.end();

    if ((s + 1) % NORM_INTERVAL === 0 || s === n - 1) {
      // Zero norms buffer then accumulate per-atom norms via atomicAdd
      enc.clearBuffer(normsBuf);
      cp = enc.beginComputePass();
      cp.setPipeline(atomicNormPL);
      cp.setBindGroup(0, atomicNormBG[next]);
      cp.dispatchWorkgroups(WG_UPDATE);
      cp.end();

      // Normalize U using per-atom norms
      cp = enc.beginComputePass();
      cp.setPipeline(normalizePL);
      cp.setBindGroup(0, normalizeBG[next]);
      cp.dispatchWorkgroups(WG_NORM);
      cp.end();
    }

    // Energy reduce (last step only)
    if (s === n - 1) {
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
    }

    cur = next;
  }

  // --- W smoothing at boundaries (even count → lands back in W_buf[0]) ---
  const wSteps = W_STEPS_PER_FRAME - (W_STEPS_PER_FRAME % 2);
  for (let s = 0; s < wSteps; s++) {
    let wp = enc.beginComputePass();
    wp.setPipeline(updateWPL);
    wp.setBindGroup(0, updateWBG[(s % 2) * 2 + cur]);
    wp.dispatchWorkgroups(WG_UPDATE);
    wp.end();
  }

  // Compute sharp W from smooth W — defines support of U
  cp = enc.beginComputePass();
  cp.setPipeline(computeSharpPL);
  cp.setBindGroup(0, computeSharpBG[cur]);
  cp.dispatchWorkgroups(WG_UPDATE);
  cp.end();

  // Compute forces on nuclei from gradient of (K - 2P)
  cp = enc.beginComputePass();
  cp.setPipeline(computeForcesPL);
  cp.setBindGroup(0, computeForcesBG);
  cp.dispatchWorkgroups(Math.ceil(NELEC / 64));
  cp.end();

  cp = enc.beginComputePass();
  cp.setPipeline(extractPL);
  cp.setBindGroup(0, extractBG[cur]);
  cp.dispatchWorkgroups(WG_EXTRACT, WG_EXTRACT);
  cp.end();

  enc.copyBufferToBuffer(sumsBuf, 0, sumsReadBuf, 0, SUMS_BYTES);
  enc.copyBufferToBuffer(sliceBuf, 0, sliceReadBuf, 0, SLICE_SIZE);
  enc.copyBufferToBuffer(forcesBuf, 0, forcesReadBuf, 0, FORCES_BYTES);
  device.queue.submit([enc.finish()]);

  await sumsReadBuf.mapAsync(GPUMapMode.READ);
  const sumsData = new Float32Array(sumsReadBuf.getMappedRange().slice(0));
  sumsReadBuf.unmap();
  E_T = sumsData[0];
  E_eK = sumsData[1];
  E_ee = sumsData[2];

  await sliceReadBuf.mapAsync(GPUMapMode.READ);
  sliceData = new Float32Array(sliceReadBuf.getMappedRange().slice(0));
  sliceReadBuf.unmap();

  await forcesReadBuf.mapAsync(GPUMapMode.READ);
  const forcesData = new Float32Array(forcesReadBuf.getMappedRange().slice(0));
  forcesReadBuf.unmap();
  lastForces = forcesData;

  tStep += n;
  lastMs = performance.now() - t0;

  // Nuclear repulsion energy + forces
  E_KK = 0;
  const soft_nuc = 0.04 * h2v;
  const nucForces = nucPos.map(() => [0, 0, 0]);
  if (addNucRepulsion) {
    for (let a = 0; a < NELEC; a++) {
      for (let b = a + 1; b < NELEC; b++) {
        if (Zn[a] === 0 || Zn[b] === 0) continue;
        const dx = (nucPos[a][0]-nucPos[b][0])*hv;
        const dy = (nucPos[a][1]-nucPos[b][1])*hv;
        const dz = (nucPos[a][2]-nucPos[b][2])*hv;
        const d = Math.sqrt(dx*dx + dy*dy + dz*dz + soft_nuc);
        E_KK += Zn[a]*Zn[b]/d;
        // Repulsive force between nuclei
        const f = Zn[a]*Zn[b]/(d*d*d);
        nucForces[a][0] += f * dx; nucForces[a][1] += f * dy; nucForces[a][2] += f * dz;
        nucForces[b][0] -= f * dx; nucForces[b][1] -= f * dy; nucForces[b][2] -= f * dz;
      }
    }
  }

  // Add electron forces from GPU (gradient of K-2P) + nuclear repulsion forces
  // Update nuclear velocities and positions
  for (let a = 0; a < NELEC; a++) {
    if (Zn[a] === 0) continue;
    // Total force = electron force (from GPU) + nuclear repulsion force
    const fx = forcesData[a*3]     + nucForces[a][0];
    const fy = forcesData[a*3 + 1] + nucForces[a][1];
    const fz = forcesData[a*3 + 2] + nucForces[a][2];
    // Velocity Verlet with damping (grid units)
    nucVel[a][0] = (nucVel[a][0] + NUC_DT * fx / hv) * NUC_FRICTION;
    nucVel[a][1] = (nucVel[a][1] + NUC_DT * fy / hv) * NUC_FRICTION;
    nucVel[a][2] = (nucVel[a][2] + NUC_DT * fz / hv) * NUC_FRICTION;
    // Update positions (grid coordinates)
    nucPos[a][0] = Math.max(2, Math.min(NN-2, nucPos[a][0] + nucVel[a][0]));
    nucPos[a][1] = Math.max(2, Math.min(NN-2, nucPos[a][1] + nucVel[a][1]));
    nucPos[a][2] = Math.max(2, Math.min(NN-2, nucPos[a][2] + nucVel[a][2]));
  }
  // Upload updated atom positions to GPU
  fillAtomBuf();

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

  if (!computing && phase === 0) {
    computing = true;
    doSteps(STEPS_PER_FRAME).then(() => {
      computing = false;
      phaseSteps += STEPS_PER_FRAME;
      if (isFinite(E) && E < E_min) E_min = E;

      if (phaseSteps >= TOTAL_STEPS) {
        console.log("=== DONE: E=" + E.toFixed(6) + " ===");
        phase = 1;  // done
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
    // 3 planes: density (u²), Z value, boundary
    const zRGB = {1:[1,1,0], 2:[1,0,0], 3:[0,0.5,1], 4:[0,1,0]};
    // Auto-scale: find max density
    let maxDens = 0;
    for (let i = 1; i < NN; i++) {
      for (let j = 1; j < NN; j++) {
        const v = sliceData[i * SS + j];
        if (v > maxDens) maxDens = v;
      }
    }
    const sc = maxDens > 0 ? 1.0 / maxDens : 1.0;
    for (let i = 1; i < NN; i++) {
      const px0 = Math.floor(PX * i * d);
      const px1 = Math.floor(PX * (i + 1) * d);
      for (let j = 1; j < NN; j++) {
        const py0 = Math.floor(PX * j * d);
        const py1 = Math.floor(PX * (j + 1) * d);
        const b = i * SS + j;
        const dens = sliceData[b];
        const zVal = Math.round(sliceData[SS * SS + b]);
        const bnd = sliceData[2 * SS * SS + b];
        let ri, gi, bi;
        if (bnd > 0.5) {
          ri = 255; gi = 255; bi = 255;
        } else {
          const ev = 255 * Math.sqrt(dens * sc);
          const col = zRGB[zVal] || [0.5, 0.5, 0.5];
          ri = Math.min(255, Math.floor(ev * col[0]));
          gi = Math.min(255, Math.floor(ev * col[1]));
          bi = Math.min(255, Math.floor(ev * col[2]));
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

    // Density line cuts through each nucleus row
    const plotH = 60;  // height of line plot in pixels
    // Horizontal cut at j = N2 (mid slice)
    noFill(); stroke(0, 255, 255); strokeWeight(1);
    let maxLine = 0;
    for (let i = 1; i < NN; i++) {
      const v = sliceData[i * SS + N2];
      if (v > maxLine) maxLine = v;
    }
    const lsc = maxLine > 0 ? plotH / maxLine : 1;
    const yBase = 400 - 5;
    beginShape();
    for (let i = 1; i < NN; i++) {
      const v = sliceData[i * SS + N2];
      vertex(i * PX, yBase - v * lsc);
    }
    endShape();
    // Vertical cut at i = N2
    stroke(255, 200, 0); strokeWeight(1);
    maxLine = 0;
    for (let j = 1; j < NN; j++) {
      const v = sliceData[N2 * SS + j];
      if (v > maxLine) maxLine = v;
    }
    const lsc2 = maxLine > 0 ? plotH / maxLine : 1;
    const xBase = 400 - 5;
    beginShape();
    for (let j = 1; j < NN; j++) {
      const v = sliceData[N2 * SS + j];
      vertex(xBase - v * lsc2, j * PX);
    }
    endShape();
    // Draw cut lines
    stroke(0, 255, 255, 60); strokeWeight(1);
    line(0, N2 * PX, 400, N2 * PX);
    stroke(255, 200, 0, 60);
    line(N2 * PX, 0, N2 * PX, 400);
  }

  // Draw nuclear positions and force arrows
  fill(255); stroke(255); strokeWeight(1);
  for (let n = 0; n < NELEC; n++) {
    if (Z[n] > 0) {
      const px = nucPos[n][0] * PX;
      const py = nucPos[n][1] * PX;
      circle(px, py, 6);
      // Draw force arrow
      if (lastForces) {
        const fx = lastForces[n*3];
        const fy = lastForces[n*3 + 1];
        const fmag = Math.sqrt(fx*fx + fy*fy);
        if (fmag > 1e-6) {
          const scale = 20.0 / Math.max(fmag, 0.1);  // scale arrows to ~20px max
          const ex = px + fx * scale;
          const ey = py + fy * scale;
          stroke(0, 255, 0); strokeWeight(2);
          line(px, py, ex, ey);
          // Arrowhead
          const ang = Math.atan2(fy, fx);
          const hl = 5;
          line(ex, ey, ex - hl*Math.cos(ang-0.4), ey - hl*Math.sin(ang-0.4));
          line(ex, ey, ex - hl*Math.cos(ang+0.4), ey - hl*Math.sin(ang+0.4));
          stroke(255); strokeWeight(1);
        }
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
  if (lastForces) {
    let fStr = "";
    for (let n = 0; n < Math.min(NELEC, 6); n++) {
      if (Z[n] === 0) continue;
      const fx = lastForces[n*3], fy = lastForces[n*3+1], fz = lastForces[n*3+2];
      const fm = Math.sqrt(fx*fx + fy*fy + fz*fz);
      fStr += atomLabels[n] + ":" + fm.toFixed(3) + " ";
    }
    fill(0, 255, 0);
    text("F " + fStr, 5, 65);
  }
}

function keyPressed() {
}
