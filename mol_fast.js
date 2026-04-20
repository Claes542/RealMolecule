// mol_fast.js — Fast small-molecule quantum solver (≤8 atoms)
// Ports h2o.js's fused-update pattern (single dispatch handling all electrons via gid.y)
// to a generic geometry. Each atom gets one orbital domain; NELEC = number of atoms.
//
// Input:  window.USER_NUCLEI = [{ i, j, k, Z, rc }, ...]   (1..8 entries)
// Optional: USER_NN, USER_SCREEN, USER_STEPS, USER_NORM_TARGETS (per-electron occupation)
//
// Output: energies E_T, E_eK, E_ee, E_KK, and total E on the canvas; density + line plots.

console.log("[mol_fast.js] script loaded");
(function () {
  "use strict";

  const NN = window.USER_NN || 200;
  const S = NN + 1;
  const S2 = S * S;
  const S3 = S * S * S;
  const N2 = Math.round(NN / 2);
  const MAX_ATOMS = 8;

  const rawNuclei = window.USER_NUCLEI || [
    { i: N2 - 20, j: N2, k: N2, Z: 1, rc: 0 },
    { i: N2 + 20, j: N2, k: N2, Z: 1, rc: 0 },
  ];
  if (rawNuclei.length > MAX_ATOMS) {
    throw new Error("mol_fast: MAX_ATOMS = " + MAX_ATOMS + ", got " + rawNuclei.length);
  }
  const nuclei = rawNuclei.map(n => ({ i: n.i, j: n.j, k: n.k, Z: n.Z, rc: n.rc || 0 }));
  const N_ATOMS = nuclei.length;
  const NELEC = N_ATOMS;
  const normTargets = window.USER_NORM_TARGETS || nuclei.map(() => 1);

  const screenAu = window.USER_SCREEN || 15;
  const hv = screenAu / NN;
  const h2v = hv * hv;
  const h3v = hv * hv * hv;
  const dv = 0.12;
  const dtv = dv * h2v;
  // Kinetic coefficient: H = -K_COEFF·∇². Standard 0.5.
  const KINETIC_COEFF = (window.USER_KINETIC_COEFF !== undefined) ? window.USER_KINETIC_COEFF : 0.5;
  // Per-orbital kinetic multiplier for multi-occupancy (target > 1). Default 1 (no change).
  // Set to 2, 3 etc. to inflate kinetic for N-like orbitals while leaving H at 0.5.
  const KC_MULTI = (window.USER_KC_MULTI !== undefined) ? window.USER_KC_MULTI : 1.0;
  const half_dv = KINETIC_COEFF * dv;
  const INTERIOR = (NN - 1) * (NN - 1) * (NN - 1);
  const STEPS_PER_FRAME = window.USER_STEPS_PER_FRAME || 500;
  const NORM_INTERVAL = 20;
  const MAX_STEPS = window.USER_STEPS || 100000;
  const PX = 400 / NN;
  const NRED = 2 + NELEC;  // T, V_eK, V_ee, plus one slot per electron norm (but packed as 3+NELEC)
  // Layout: [0..NELEC-1] = norm² per electron, [NELEC] = T, [NELEC+1] = V_eK, [NELEC+2] = V_ee,
  //         [NELEC+3] = dipEx (electronic dipole x), [NELEC+4] = dipEy, [NELEC+5] = dipEz
  const NRED_SLOTS = NELEC + 6;

  // ===== Multigrid V-cycle constants =====
  if (NN % 2 !== 0) throw new Error("mol_fast: NN must be even for V-cycle");
  const NC = NN / 2;
  const SC = NC + 1;
  const SC2 = SC * SC;
  const SC3 = SC * SC * SC;
  const INTERIOR_C = (NC - 1) * (NC - 1) * (NC - 1);
  const WG_COARSE = Math.ceil(INTERIOR_C / 256);
  const VCYCLE_INTERVAL = window.USER_VCYCLE_INTERVAL || 50;
  const COARSE_ITERS = window.USER_COARSE_ITERS || 10;
  const VCYCLE_ENABLED = window.USER_VCYCLE !== false;

  console.log("mol_fast: NN=" + NN + ", NELEC=" + NELEC + ", screen=" + screenAu + " au, h=" + hv.toFixed(4) +
              ", V-cycle=" + (VCYCLE_ENABLED ? "ON (every " + VCYCLE_INTERVAL + " steps, NC=" + NC + ")" : "OFF"));

  // ===== WGSL SHADERS (N_ATOMS templated) =====

  const paramStructWGSL = `
struct P {
  NN: u32, S: u32, S2: u32, S3: u32,
  N2: u32, _p0: u32, _p1: u32, _p2: u32,
  h: f32, h2: f32, inv_h: f32, inv_h2: f32,
  dt: f32, half_d: f32, h3: f32, TWO_PI: f32,
  R_out: f32, voronoi: f32, alpha_TF: f32, full_self: f32,
  atoms: array<vec4<f32>, ${MAX_ATOMS}>,   // (i, j, k, Z)
  rcs:   array<vec4<f32>, ${MAX_ATOMS}>,   // (rc, norm_target, _, _)
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
  if (m >= ${NELEC}u) { return; }
  let o = m * p.S3;

  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;

  // --- Competition term: cm = u_self - max(u_other) — drives advection of w front ---
  let uc = Ui[o + id];
  var u_max_other: f32 = 0.0;
  for (var n: u32 = 0u; n < ${NELEC}u; n++) {
    if (n == m) { continue; }
    let un = Ui[n * p.S3 + id];
    if (un > u_max_other) { u_max_other = un; }
  }
  let cm = uc - u_max_other;

  // --- w evolution (free-boundary advection + diffusion + cusps + Voronoi) ---
  let wc  = Wi[o + id];
  let wip = Wi[o + id + p.S2]; let wim = Wi[o + id - p.S2];
  let wjp = Wi[o + id + p.S];  let wjm = Wi[o + id - p.S];
  let wkp = Wi[o + id + 1u];   let wkm = Wi[o + id - 1u];
  let lw = (wip + wim + wjp + wjm + wkp + wkm - 6.0 * wc) * p.inv_h2;
  let gx = (wip - wim) * p.inv_h;
  let gy = (wjp - wjm) * p.inv_h;
  let gz = (wkp - wkm) * p.inv_h;
  var nw = wc
    + 0.25 * p.dt * abs(cm) * lw
    + 5.0  * p.dt * cm * sqrt(gx * gx + gy * gy + gz * gz);
  nw = clamp(nw, 0.0, 1.0);

  // --- Per-atom smooth rc cusp on w (no distance-based Voronoi —
  //     front-tracking via c = u_self - max(u_other) handles domain competition)
  for (var a: u32 = 0u; a < ${N_ATOMS}u; a++) {
    let atom = p.atoms[a];
    let dx = f32(i) - atom.x;
    let dy = f32(j) - atom.y;
    let dz = f32(k) - atom.z;
    let r  = sqrt(dx*dx + dy*dy + dz*dz) * p.h;
    let rc_a = p.rcs[a].x;
    if (rc_a > 0.0 && r < rc_a) {
      let edge = rc_a - 3.0 * p.h;
      let t = clamp((r - edge) / (rc_a - edge), 0.0, 1.0);
      nw = min(nw, t * t * (3.0 - 2.0 * t));
    }
  }
  Wo[o + id] = nw;

  // --- u evolution: w-weighted Laplacian + (K - V_Hartree_other)·u·w ---
  // V_Hartree_other (felt by electron m) = 2·(P_total - f_m·P_m)
  // where f_m = 1/N_m accounts for pair counting in multi-occupancy orbitals.
  var P_total_id: f32 = 0.0;
  for (var n: u32 = 0u; n < ${NELEC}u; n++) {
    P_total_id += Pi[n * p.S3 + id];
  }
  let selfN = p.rcs[m].y;
  // fm: fraction of P_m to subtract from V_felt for SIC. full_self=1 keeps all self-Hartree (fm=0) for multi-occ.
  let fm_multi = select(1.0 / selfN, 0.0, p.full_self > 0.5);
  let fm = select(1.0, fm_multi, selfN > 1.0);
  let Vother = 2.0 * (P_total_id - fm * Pi[o + id]);

  // Thomas-Fermi Pauli-like penalty: V_TF = alpha·ρ^(2/3), active only for multi-occupancy
  let rho_m = uc * uc;
  let V_TF = select(0.0, p.alpha_TF * pow(max(rho_m, 1e-20), 0.6666667), selfN > 1.0 && p.alpha_TF > 0.0);

  // Per-atom directional tilt: V_tilt = -α·y (biases density toward +y for α>0, -y for α<0)
  let tilt_y = p.rcs[m].w;
  let y_au = (f32(j) - f32(p.N2)) * p.h;
  let V_tilt = -tilt_y * y_au;

  let uip = Ui[o + id + p.S2]; let uim = Ui[o + id - p.S2];
  let ujp = Ui[o + id + p.S];  let ujm = Ui[o + id - p.S];
  let ukp = Ui[o + id + 1u];   let ukm = Ui[o + id - 1u];
  // Per-orbital kinetic multiplier: multi-occupancy gets KC_MULTI× kinetic cost.
  let k_mult = select(1.0, ${KC_MULTI.toFixed(6)}, selfN > 1.0);
  let hd = p.half_d * k_mult;
  var u_new: f32 = uc
    + hd * ((uip - uc) * (wip + nw) * 0.5 - (uc - uim) * (nw + wim) * 0.5)
    + hd * ((ujp - uc) * (wjp + nw) * 0.5 - (uc - ujm) * (nw + wjm) * 0.5)
    + hd * ((ukp - uc) * (wkp + nw) * 0.5 - (uc - ukm) * (nw + wkm) * 0.5)
    + p.dt * (K[id] - Vother - V_TF - V_tilt) * uc * wc;
  Uo[o + id] = u_new;

  // --- Poisson: each P_m sourced by electron m's OWN density u_m² ---
  // V felt = 2·(P_total - f_m·P_m) done above; self-interaction removed in update.
  let Pc = Pi[o + id];
  Po[o + id] = Pc
    + p.dt * (Pi[o + id + p.S2] + Pi[o + id - p.S2]
            + Pi[o + id + p.S]  + Pi[o + id - p.S]
            + Pi[o + id + 1u]   + Pi[o + id - 1u]
            - 6.0 * Pc) * p.inv_h2
    + p.TWO_PI * p.dt * uc * uc;
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

const NRED: u32 = ${NRED_SLOTS}u;
var<workgroup> sn: array<f32, ${128 * NRED_SLOTS}>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32,
        @builtin(workgroup_id) wgid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  for (var x: u32 = 0u; x < NRED; x++) { sn[lid * NRED + x] = 0.0; }
  if (gid.x < tot) {
    let k = (gid.x % NM) + 1u;
    let j = ((gid.x / NM) % NM) + 1u;
    let i = (gid.x / (NM * NM)) + 1u;
    let id = i * p.S2 + j * p.S + k;
    var T: f32 = 0.0;
    var VeK: f32 = 0.0;
    var Vee: f32 = 0.0;
    var dipEx: f32 = 0.0;
    var dipEy: f32 = 0.0;
    var dipEz: f32 = 0.0;
    // Cell coordinates relative to grid center (N2, N2, N2), in au
    let xc = (f32(i) - f32(p.N2)) * p.h;
    let yc = (f32(j) - f32(p.N2)) * p.h;
    let zc = (f32(k) - f32(p.N2)) * p.h;
    // P_total at this cell (same for all electrons)
    var P_tot: f32 = 0.0;
    for (var n: u32 = 0u; n < ${NELEC}u; n++) { P_tot += Pv[n * p.S3 + id]; }
    for (var m: u32 = 0u; m < ${NELEC}u; m++) {
      let o = m * p.S3;
      let v = U[o + id];
      sn[lid * NRED + m] = v * v * p.h3;
      if (W[o + id] > 0.1) {
        let a = U[o + id + p.S2] - v;
        let b = U[o + id + p.S]  - v;
        let c = U[o + id + 1u]   - v;
        let selfN_T = p.rcs[m].y;
        let k_mult_T = select(1.0, ${KC_MULTI.toFixed(6)}, selfN_T > 1.0);
        T += k_mult_T * ${KINETIC_COEFF.toFixed(6)} * (a * a + b * b + c * c) * p.h;
      }
      VeK -= K[id] * v * v * p.h3;
      // Electronic dipole contribution: -∫ρ_m·r dV  (electron charge = -1)
      let rho_cell = v * v * p.h3;
      dipEx -= rho_cell * xc;
      dipEy -= rho_cell * yc;
      dipEz -= rho_cell * zc;
      // Vee: E_Hartree = 0.5·∫ V_tot·ρ_tot - SIC
      //   Use E_ee = Σ_m ∫ (P_tot - f_m·P_m)·u_m² dV  (removes self-Hartree per orbital)
      let selfN = p.rcs[m].y;
      let fm = select(1.0, 1.0 / selfN, selfN > 1.0);
      Vee += (P_tot - fm * Pv[o + id]) * v * v * p.h3;
      // Thomas-Fermi kinetic contribution: E_TF = (3/5)·α·ρ^(5/3), multi-occupancy only
      if (selfN > 1.0 && p.alpha_TF > 0.0) {
        let rho_m = v * v;
        T += 0.6 * p.alpha_TF * pow(max(rho_m, 1e-20), 1.6666667) * p.h3;
      }
      // Tilt potential contribution: E_tilt = ∫ V_tilt · ρ_m = -α·∫ρ_m·y dV
      let tilt_y_r = p.rcs[m].y;  // unused placeholder; real tilt in .w slot below
      let tilt = p.rcs[m].w;
      if (tilt != 0.0) {
        T += (-tilt) * yc * v * v * p.h3;
      }
    }
    sn[lid * NRED + ${NELEC}u]      = T;
    sn[lid * NRED + ${NELEC + 1}u]  = VeK;
    sn[lid * NRED + ${NELEC + 2}u]  = Vee;
    sn[lid * NRED + ${NELEC + 3}u]  = dipEx;
    sn[lid * NRED + ${NELEC + 4}u]  = dipEy;
    sn[lid * NRED + ${NELEC + 5}u]  = dipEz;
  }
  workgroupBarrier();
  for (var s: u32 = 64u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var x: u32 = 0u; x < NRED; x++) { sn[lid * NRED + x] += sn[(lid + s) * NRED + x]; }
    }
    workgroupBarrier();
  }
  if (lid == 0u) {
    let base = wgid.x * NRED;
    for (var x: u32 = 0u; x < NRED; x++) { partials[base + x] = sn[x]; }
  }
}
`;

  const finalizeWGSL = `
struct NWG { count: u32 }
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> sums: array<f32>;
@group(0) @binding(2) var<uniform> nwg: NWG;

const NRED: u32 = ${NRED_SLOTS}u;
var<workgroup> wg: array<f32, ${128 * NRED_SLOTS}>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lid: u32) {
  for (var x: u32 = 0u; x < NRED; x++) { wg[lid * NRED + x] = 0.0; }
  for (var i: u32 = lid; i < nwg.count; i += 128u) {
    for (var x: u32 = 0u; x < NRED; x++) { wg[lid * NRED + x] += partials[i * NRED + x]; }
  }
  workgroupBarrier();
  for (var s: u32 = 64u; s > 0u; s >>= 1u) {
    if (lid < s) {
      for (var x: u32 = 0u; x < NRED; x++) { wg[lid * NRED + x] += wg[(lid + s) * NRED + x]; }
    }
    workgroupBarrier();
  }
  if (lid == 0u) {
    for (var x: u32 = 0u; x < NRED; x++) { sums[x] = wg[x]; }
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
  for (var m: u32 = 0u; m < ${NELEC}u; m++) {
    let n = sums[m];
    let tgt = p.rcs[m].y;
    if (n > 0.0 && tgt > 0.0) { U[m * p.S3 + id] *= sqrt(tgt / n); }
  }
}
`;

  // --- Nuclear force: F_α = Z_α · ∇V_H(R_α) where V_H = 2·Σ_m P_m ---
  // Per-atom, sphere-averaged finite-difference gradient of total P.
  const FORCE_RADIUS = 4;  // grid cells
  const forceWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> P_all: array<f32>;  // per-electron P, contiguous
@group(0) @binding(2) var<storage, read_write> forceSums: array<f32>;  // N_ATOMS*3 floats

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let atom = gid.x;
  if (atom >= ${N_ATOMS}u) { return; }
  let Z_a = p.atoms[atom].w;
  if (Z_a <= 0.0) {
    forceSums[atom * 3u] = 0.0; forceSums[atom * 3u + 1u] = 0.0; forceSums[atom * 3u + 2u] = 0.0;
    return;
  }
  let ci = i32(p.atoms[atom].x);
  let cj = i32(p.atoms[atom].y);
  let ck = i32(p.atoms[atom].z);
  let R = ${FORCE_RADIUS}i;
  let R2f = f32(R * R);
  let inv2h = 0.5 * p.inv_h;
  let S_i = i32(p.S);
  var sumFi: f32 = 0.0; var sumFj: f32 = 0.0; var sumFk: f32 = 0.0;
  var count: f32 = 0.0;
  for (var di: i32 = -R; di <= R; di++) {
    let ii = ci + di;
    if (ii < 1 || ii >= S_i - 1) { continue; }
    for (var dj: i32 = -R; dj <= R; dj++) {
      let jj = cj + dj;
      if (jj < 1 || jj >= S_i - 1) { continue; }
      for (var dk: i32 = -R; dk <= R; dk++) {
        let kk = ck + dk;
        if (kk < 1 || kk >= S_i - 1) { continue; }
        let r2 = f32(di*di + dj*dj + dk*dk);
        if (r2 > R2f) { continue; }
        let ui = u32(ii); let uj = u32(jj); let uk = u32(kk);
        // Sum P across all electrons at each neighbor
        var Pxp: f32 = 0.0; var Pxm: f32 = 0.0;
        var Pyp: f32 = 0.0; var Pym: f32 = 0.0;
        var Pzp: f32 = 0.0; var Pzm: f32 = 0.0;
        for (var m: u32 = 0u; m < ${NELEC}u; m++) {
          let o = m * p.S3;
          Pxp += P_all[o + (ui + 1u) * p.S2 + uj * p.S + uk];
          Pxm += P_all[o + (ui - 1u) * p.S2 + uj * p.S + uk];
          Pyp += P_all[o + ui * p.S2 + (uj + 1u) * p.S + uk];
          Pym += P_all[o + ui * p.S2 + (uj - 1u) * p.S + uk];
          Pzp += P_all[o + ui * p.S2 + uj * p.S + (uk + 1u)];
          Pzm += P_all[o + ui * p.S2 + uj * p.S + (uk - 1u)];
        }
        sumFi += (Pxp - Pxm) * inv2h;
        sumFj += (Pyp - Pym) * inv2h;
        sumFk += (Pzp - Pzm) * inv2h;
        count += 1.0;
      }
    }
  }
  let avgInv = select(0.0, 1.0 / count, count > 0.0);
  // F = 2·Z·<∇P> (factor 2 converts 2π-source P to full V_Hartree)
  forceSums[atom * 3u]      = 2.0 * Z_a * sumFi * avgInv;
  forceSums[atom * 3u + 1u] = 2.0 * Z_a * sumFj * avgInv;
  forceSums[atom * 3u + 2u] = 2.0 * Z_a * sumFk * avgInv;
}
`;

  // --- Rebuild K from current atom positions (after nuclei move) ---
  const recomputeK_WGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> K: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  if (id >= p.S3) { return; }
  let k = id % p.S;
  let j = (id / p.S) % p.S;
  let i = id / p.S2;
  var Kval: f32 = 0.0;
  for (var a: u32 = 0u; a < ${N_ATOMS}u; a++) {
    let Z_a = p.atoms[a].w;
    if (Z_a <= 0.0) { continue; }
    let di = (f32(i) - p.atoms[a].x) * p.h;
    let dj = (f32(j) - p.atoms[a].y) * p.h;
    let dk = (f32(k) - p.atoms[a].z) * p.h;
    let r_soft = sqrt(di*di + dj*dj + dk*dk + 0.04 * p.h2);
    Kval += Z_a / r_soft;
  }
  K[id] = Kval;
}
`;

  // Output layout for slice:
  //   [0 .. NELEC*S*S-1]           : U·W (2D density in z=N2 slice, color image)
  //   [NELEC*S*S .. 2*NELEC*S*S-1] : U alone along j=N2 line (1D line plot)
  //   [2*NELEC*S*S .. 3*NELEC*S*S-1]: W alone along j=N2 line (1D line plot)
  const extractWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> Pv: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  let j = g.y;
  let SS = p.NN + 1u;
  if (i > p.NN || j > p.NN) { return; }
  for (var m: u32 = 0u; m < ${NELEC}u; m++) {
    let idx2D = m * p.S3 + i * p.S2 + j * p.S + p.N2;
    let uv = U[idx2D];
    let wv = W[idx2D];
    out[m * SS * SS + i * SS + j] = select(0.0, uv, wv > 0.0);
    // Line through center along i-axis at j=N2, k=N2.
    if (j == 0u) {
      let idxLine = i * p.S2 + p.N2 * p.S + p.N2;
      let idxMLine = m * p.S3 + idxLine;
      out[${NELEC}u * SS * SS + m * SS + i] = U[idxMLine];
      out[${2 * NELEC}u * SS * SS + m * SS + i] = W[idxMLine];
      // --- Potential lines, packed in the UNUSED tail of section 1 (U-line region) ---
      //   [N*SS*SS + N*SS          .. N*SS*SS + N*SS + SS - 1]  = 2·P_total (V_Hartree total)
      //   [N*SS*SS + N*SS + SS     .. N*SS*SS + (N+1)*SS + m*SS + i] = V_pauli per electron
      var P_sum: f32 = 0.0;
      for (var n: u32 = 0u; n < ${NELEC}u; n++) {
        P_sum += Pv[n * p.S3 + idxLine];
      }
      if (m == 0u) {
        out[${NELEC}u * SS * SS + ${NELEC}u * SS + i] = 2.0 * P_sum;
      }
      let selfN = p.rcs[m].y;
      var V_pauli: f32 = 0.0;
      if (selfN > 1.0 && p.alpha_TF > 0.0) {
        let rho_m = U[idxMLine] * U[idxMLine];
        V_pauli = p.alpha_TF * pow(max(rho_m, 1e-20), 0.6666667);
      }
      out[${NELEC}u * SS * SS + (${NELEC + 1}u + m) * SS + i] = V_pauli;
    }
  }
}
`;

  // ===== V-cycle shaders =====
  // Source rho for electron m's Poisson: now just electron m's OWN density u_m².
  // (Self-interaction is subtracted at the update step via V_felt = 2·(P_total - f_m·P_m).)
  const rhoSourceWGSL = `
${paramStructWGSL}
struct VIDX { m: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> vidx: VIDX;
@group(0) @binding(2) var<storage, read> U: array<f32>;
@group(0) @binding(3) var<storage, read_write> rhoSrc: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.S3) { return; }
  let id = gid.x;
  let uc = U[vidx.m * p.S3 + id];
  rhoSrc[id] = uc * uc;
}
`;

  // Residual r = -2π·rho - ∇²P_m
  const residualPmWGSL = `
${paramStructWGSL}
struct VIDX { m: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> vidx: VIDX;
@group(0) @binding(2) var<storage, read> P_all: array<f32>;
@group(0) @binding(3) var<storage, read> rhoSrc: array<f32>;
@group(0) @binding(4) var<storage, read_write> res: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let NM = p.NN - 1u;
  let tot = NM * NM * NM;
  if (gid.x >= tot) { return; }
  let k = (gid.x % NM) + 1u;
  let j = ((gid.x / NM) % NM) + 1u;
  let i = (gid.x / (NM * NM)) + 1u;
  let id = i * p.S2 + j * p.S + k;
  let o = vidx.m * p.S3;
  let Pc = P_all[o + id];
  let lap = (P_all[o + id + p.S2] + P_all[o + id - p.S2]
           + P_all[o + id + p.S]  + P_all[o + id - p.S]
           + P_all[o + id + 1u]   + P_all[o + id - 1u]
           - 6.0 * Pc) * p.inv_h2;
  res[id] = -p.TWO_PI * rhoSrc[id] - lap;
}
`;

  // Full-weighting restriction: fine (S³) → coarse (SC³)
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

  // Weighted Jacobi on coarse grid: (I/3)·E + (Σnbr + 4h²·rhs)/9
  const coarseSmoothWGSL = `
${paramStructWGSL}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Ein: array<f32>;
@group(0) @binding(2) var<storage, read_write> Eout: array<f32>;
@group(0) @binding(3) var<storage, read> rhsC: array<f32>;

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
  Eout[cid] = 0.3333 * ec + (sum_nbr + hc2 * rhsC[cid]) / 9.0;
}
`;

  // Trilinear prolongation: coarse error → P_m (additive, damped 0.5)
  const prolongCorrectPmWGSL = `
${paramStructWGSL}
struct VIDX { m: u32, _p0: u32, _p1: u32, _p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> vidx: VIDX;
@group(0) @binding(2) var<storage, read> Ec: array<f32>;
@group(0) @binding(3) var<storage, read_write> P_all: array<f32>;

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
  P_all[vidx.m * p.S3 + fid] += 0.5 * corr;
}
`;

  // ===== GPU STATE =====
  let device, paramsBuf, K_buf, sumsBuf, sumsReadBuf, sliceBuf, sliceReadBuf, partialsBuf, numWGBuf;
  let U_buf = [], W_buf = [], P_buf = [];
  let updatePL, reducePL, finalizePL, normalizePL, extractPL, forcePL, recomputeKPL;
  let updateBG = [], reduceBG = [], finalizeBG, normalizeBG = [], extractBG = [];
  let forceBG = [], recomputeKBG;
  let forceBuf, forceReadBuf;
  // V-cycle state
  let vcycleIdxBuf = [];  // one 16-byte uniform per electron m (holds index)
  let rhoSourceBuf, residualBuf, rhsCoarseBuf;
  let errCoarseBuf = [];  // ping-pong on coarse grid
  let rhoSourcePL, residualPmPL, restrictPL, coarseSmoothPL, prolongCorrectPmPL;
  let rhoSourceBG = [], residualPmBG = [], prolongCorrectPmBG = [];
  let restrictBG, coarseSmoothBG = [];
  let cur = 0, gpuReady = false, computing = false;
  let tStep = 0, E = 0, lastMs = 0;
  let E_T = 0, E_eK = 0, E_ee = 0, E_KK = 0;
  let dipole = [0, 0, 0], dipoleMag = 0;  // total dipole in atomic units (1 au = 2.5417 Debye)
  let gpuError = null;
  let sliceData = null;
  // Nuclear dynamics state (CPU)
  // DT_NUC in atomic time units. Mass per atom = Z·1836 (roughly, proton mass × charge number)
  // — good enough for H (Z=1 → mass 1836) and light pseudopotential atoms. For Z=2 O pseudo,
  // this gives 3672 which is too light vs real O (~29000); override with USER_MASS if needed.
  const DT_NUC = window.USER_DT_NUC || 2.0;
  const NUC_DAMPING = window.USER_NUC_DAMPING || 0.9;
  const PROTON_MASS = 1836;
  const nucMass = (window.USER_MASS || nuclei.map(n => Math.max(1, n.Z) * PROTON_MASS));
  let nucVel = Array.from({length: N_ATOMS}, () => [0, 0, 0]);
  let nucForce = Array.from({length: N_ATOMS}, () => [0, 0, 0]);
  let dynamicsEnabled = window.USER_DYNAMICS || false;
  let forcesReadbackPending = false;

  const WG_UPDATE = Math.ceil(INTERIOR / 256);
  const WG_REDUCE = Math.ceil(INTERIOR / 128);
  const WG_NORM = Math.ceil(INTERIOR / 256);
  const WG_EXTRACT = Math.ceil(S / 16);
  const SLICE_SIZE = 3 * NELEC * S * S * 4;  // 2D image + U-line + W-line
  const SUMS_BYTES = NRED_SLOTS * 4;

  // ===== PARAMS BUFFER =====
  // Layout: 80 bytes fixed header + MAX_ATOMS*16 (atoms) + MAX_ATOMS*16 (rcs)
  const PARAM_BYTES = 80 + MAX_ATOMS * 16 * 2;

  function writeParams() {
    const buf = new ArrayBuffer(PARAM_BYTES);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    u[0] = NN; u[1] = S; u[2] = S2; u[3] = S3;
    u[4] = N2;
    f[8] = hv; f[9] = h2v; f[10] = 1 / hv; f[11] = 1 / h2v;
    f[12] = dtv; f[13] = half_dv; f[14] = h3v; f[15] = 2 * Math.PI;
    f[16] = 3.0;  // R_out (u/w cutoff radius in au — large default)
    f[17] = 1.0;  // voronoi on
    f[18] = (window.USER_ALPHA_TF !== undefined) ? window.USER_ALPHA_TF : 0.0;  // Thomas-Fermi Pauli coeff (default off)
    f[19] = window.USER_FULL_SELF ? 1.0 : 0.0;  // 1 ⇒ keep full self-Hartree for multi-occupancy (no SIC reduction)
    // atoms at offset 80 bytes (index 20 in f32 array)
    for (let a = 0; a < MAX_ATOMS; a++) {
      const off = 20 + a * 4;
      if (a < N_ATOMS) {
        f[off + 0] = nuclei[a].i;
        f[off + 1] = nuclei[a].j;
        f[off + 2] = nuclei[a].k;
        f[off + 3] = nuclei[a].Z;
      }
    }
    // rcs at offset 80 + MAX_ATOMS*16 = 80 + 128 = 208 bytes (index 52)
    const rcsOff = 20 + MAX_ATOMS * 4;
    for (let a = 0; a < MAX_ATOMS; a++) {
      const off = rcsOff + a * 4;
      if (a < N_ATOMS) {
        f[off + 0] = nuclei[a].rc;
        f[off + 1] = normTargets[a];
        f[off + 3] = nuclei[a].tilt_y || 0;  // per-atom directional-tilt strength (V = -α·y-offset)
      }
    }
    // line-plot j offset (grid cells from N2) packed into last rcs slot .z
    f[rcsOff + (MAX_ATOMS - 1) * 4 + 2] = window.USER_LINE_J_OFFSET || 0;
    device.queue.writeBuffer(paramsBuf, 0, buf);
  }

  function uploadInitialData() {
    const Kd = new Float32Array(S3);
    const Ud = new Float32Array(NELEC * S3);
    const Wd = new Float32Array(NELEC * S3);
    const Pd = new Float32Array(NELEC * S3);
    const soft = 0.04 * h2v;
    const R_init = 3.0;  // au — initial u/w extent

    for (let i = 0; i <= NN; i++) {
      for (let j = 0; j <= NN; j++) {
        for (let k = 0; k <= NN; k++) {
          const id = i * S2 + j * S + k;
          let K_acc = 0;
          // Precompute per-atom distance (to each nucleus) for this cell
          const rList = [];
          const uList = [];
          let bestM = -1, bestU = -1;
          for (let a = 0; a < N_ATOMS; a++) {
            const dx = (i - nuclei[a].i) * hv;
            const dy = (j - nuclei[a].j) * hv;
            const dz = (k - nuclei[a].k) * hv;
            const r_soft = Math.sqrt(dx * dx + dy * dy + dz * dz + soft);
            const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
            K_acc += nuclei[a].Z / r_soft;
            const uHere = Math.exp(-nuclei[a].Z * r);
            rList.push(r);
            uList.push(uHere);
            if (uHere > bestU) { bestU = uHere; bestM = a; }
          }
          Kd[id] = K_acc;
          // u-init: each u_m = exp(-Z_m·r_m) at every cell (smooth, with rc smoothcut
          // at m's own nucleus). No Voronoi on u — tails visible everywhere on any line.
          for (let m = 0; m < N_ATOMS; m++) {
            const rc_m = nuclei[m].rc;
            let uCut = uList[m];
            if (rc_m > 0 && rList[m] < rc_m) {
              const edge = rc_m - 3 * hv;
              const t = Math.max(0, Math.min(1, (rList[m] - edge) / (rc_m - edge)));
              uCut *= t * t * (3 - 2 * t);
            }
            Ud[m * S3 + id] = uCut;
          }
          // w-init: best-wins hard partition (only the winner gets w=1; others 0).
          if (bestM >= 0 && rList[bestM] < R_init) {
            const rc_b = nuclei[bestM].rc;
            let wCut = 1.0;
            if (rc_b > 0 && rList[bestM] < rc_b) {
              const edge = rc_b - 3 * hv;
              const t = Math.max(0, Math.min(1, (rList[bestM] - edge) / (rc_b - edge)));
              wCut = t * t * (3 - 2 * t);
            }
            Wd[bestM * S3 + id] = wCut;
          }
          // P_m init = 0.5·Z_m / (r_m + h²)
          // The Hartree potential generated by electron m's OWN density,
          // approximated as a point charge Z_m at atom m's nucleus.
          // Factor 0.5 matches our ∇²P = -2π·ρ convention (Green's function is 1/(2r)).
          for (let m = 0; m < N_ATOMS; m++) {
            Pd[m * S3 + id] = 0.5 * nuclei[m].Z / (rList[m] + h2v);
          }
        }
      }
    }
    device.queue.writeBuffer(K_buf, 0, Kd);
    for (let i = 0; i < 2; i++) {
      device.queue.writeBuffer(U_buf[i], 0, Ud);
      device.queue.writeBuffer(W_buf[i], 0, Wd);
      device.queue.writeBuffer(P_buf[i], 0, Pd);
    }
    cur = 0;
  }

  async function initGPU() {
    if (!navigator.gpu) { gpuError = "WebGPU not supported"; return; }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) { gpuError = "No GPU adapter"; return; }
    // Request the adapter's maximum limits so we can allocate large buffers (NN=200, NELEC up to 8).
    const requiredLimits = {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    };
    device = await adapter.requestDevice({ requiredLimits });
    console.log("GPU limits: maxStorageBuffer=" + (adapter.limits.maxStorageBufferBindingSize >> 20) + " MB, maxBuffer=" + (adapter.limits.maxBufferSize >> 20) + " MB");
    device.lost.then(info => { gpuError = "GPU lost: " + info.message; gpuReady = false; });

    const bs = S3 * 4, bN = NELEC * S3 * 4;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    K_buf = device.createBuffer({ size: bs, usage });
    for (let i = 0; i < 2; i++) {
      U_buf[i] = device.createBuffer({ size: bN, usage: usage | GPUBufferUsage.COPY_SRC });
      W_buf[i] = device.createBuffer({ size: bN, usage: usage | GPUBufferUsage.COPY_SRC });
      P_buf[i] = device.createBuffer({ size: bN, usage: usage | GPUBufferUsage.COPY_SRC });
    }
    partialsBuf = device.createBuffer({ size: WG_REDUCE * NRED_SLOTS * 4, usage: GPUBufferUsage.STORAGE });
    sumsBuf = device.createBuffer({ size: SUMS_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    sumsReadBuf = device.createBuffer({ size: SUMS_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    sliceBuf = device.createBuffer({ size: SLICE_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    sliceReadBuf = device.createBuffer({ size: SLICE_SIZE, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    numWGBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(numWGBuf, 0, new Uint32Array([WG_REDUCE, 0, 0, 0]));
    paramsBuf = device.createBuffer({ size: PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    writeParams();
    uploadInitialData();
    // Nuclear force buffer: N_ATOMS × 3 floats (fx, fy, fz per atom)
    forceBuf = device.createBuffer({ size: N_ATOMS * 3 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    forceReadBuf = device.createBuffer({ size: N_ATOMS * 3 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    // V-cycle buffers (one rhoSource + residual on fine grid; rhs + err ping-pong on coarse)
    rhoSourceBuf = device.createBuffer({ size: S3 * 4, usage: GPUBufferUsage.STORAGE });
    residualBuf  = device.createBuffer({ size: S3 * 4, usage: GPUBufferUsage.STORAGE });
    rhsCoarseBuf = device.createBuffer({ size: SC3 * 4, usage: GPUBufferUsage.STORAGE });
    errCoarseBuf[0] = device.createBuffer({ size: SC3 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    errCoarseBuf[1] = device.createBuffer({ size: SC3 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    // Per-electron index uniforms (written once at init)
    for (let m = 0; m < NELEC; m++) {
      vcycleIdxBuf[m] = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(vcycleIdxBuf[m], 0, new Uint32Array([m, 0, 0, 0]));
    }

    async function compile(name, code) {
      const mod = device.createShaderModule({ code });
      const info = await mod.getCompilationInfo();
      for (const msg of info.messages) {
        if (msg.type === 'error') throw new Error(name + " shader: " + msg.message + " (line " + msg.lineNum + ")");
      }
      return mod;
    }
    const updateMod = await compile("update", updateWGSL);
    const reduceMod = await compile("reduce", reduceWGSL);
    const finalizeMod = await compile("finalize", finalizeWGSL);
    const normalizeMod = await compile("normalize", normalizeWGSL);
    const extractMod = await compile("extract", extractWGSL);
    const forceMod = await compile("force", forceWGSL);
    const recomputeKMod = await compile("recomputeK", recomputeK_WGSL);
    const rhoSourceMod = await compile("rhoSource", rhoSourceWGSL);
    const residualPmMod = await compile("residualPm", residualPmWGSL);
    const restrictMod = await compile("restrict", restrictWGSL);
    const coarseSmoothMod = await compile("coarseSmooth", coarseSmoothWGSL);
    const prolongCorrectPmMod = await compile("prolongCorrectPm", prolongCorrectPmWGSL);
    updatePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: updateMod, entryPoint: 'main' } });
    reducePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: reduceMod, entryPoint: 'main' } });
    finalizePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: finalizeMod, entryPoint: 'main' } });
    normalizePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: normalizeMod, entryPoint: 'main' } });
    extractPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: extractMod, entryPoint: 'main' } });
    forcePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: forceMod, entryPoint: 'main' } });
    recomputeKPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: recomputeKMod, entryPoint: 'main' } });
    rhoSourcePL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: rhoSourceMod, entryPoint: 'main' } });
    residualPmPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: residualPmMod, entryPoint: 'main' } });
    restrictPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: restrictMod, entryPoint: 'main' } });
    coarseSmoothPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: coarseSmoothMod, entryPoint: 'main' } });
    prolongCorrectPmPL = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: prolongCorrectPmMod, entryPoint: 'main' } });

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
        { binding: 3, resource: { buffer: sliceBuf } },
        { binding: 4, resource: { buffer: P_buf[c] } },
      ]});
    }
    finalizeBG = device.createBindGroup({ layout: finalizePL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: partialsBuf } },
      { binding: 1, resource: { buffer: sumsBuf } },
      { binding: 2, resource: { buffer: numWGBuf } },
    ]});
    // Force bind group (one per buffer-parity, since P_buf alternates)
    for (let c = 0; c < 2; c++) {
      forceBG[c] = device.createBindGroup({ layout: forcePL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: P_buf[c] } },
        { binding: 2, resource: { buffer: forceBuf } },
      ]});
    }
    recomputeKBG = device.createBindGroup({ layout: recomputeKPL.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: K_buf } },
    ]});

    // V-cycle bind groups: rhoSource/residual/prolong need (cur parity) × (electron m)
    for (let c = 0; c < 2; c++) {
      rhoSourceBG[c] = [];
      residualPmBG[c] = [];
      prolongCorrectPmBG[c] = [];
      for (let m = 0; m < NELEC; m++) {
        rhoSourceBG[c][m] = device.createBindGroup({
          layout: rhoSourcePL.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: vcycleIdxBuf[m] } },
            { binding: 2, resource: { buffer: U_buf[c] } },
            { binding: 3, resource: { buffer: rhoSourceBuf } },
          ]});
        residualPmBG[c][m] = device.createBindGroup({
          layout: residualPmPL.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: vcycleIdxBuf[m] } },
            { binding: 2, resource: { buffer: P_buf[c] } },
            { binding: 3, resource: { buffer: rhoSourceBuf } },
            { binding: 4, resource: { buffer: residualBuf } },
          ]});
        prolongCorrectPmBG[c][m] = device.createBindGroup({
          layout: prolongCorrectPmPL.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: vcycleIdxBuf[m] } },
            { binding: 2, resource: { buffer: errCoarseBuf[0] } },  // final result lives in [0] after even iters
            { binding: 3, resource: { buffer: P_buf[c] } },
          ]});
      }
    }
    restrictBG = device.createBindGroup({
      layout: restrictPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: residualBuf } },
        { binding: 2, resource: { buffer: rhsCoarseBuf } },
      ]});
    // Coarse smooth ping-pong: [0] reads errCoarseBuf[0] → writes [1]; [1] reads [1] → writes [0]
    coarseSmoothBG[0] = device.createBindGroup({
      layout: coarseSmoothPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: errCoarseBuf[0] } },
        { binding: 2, resource: { buffer: errCoarseBuf[1] } },
        { binding: 3, resource: { buffer: rhsCoarseBuf } },
      ]});
    coarseSmoothBG[1] = device.createBindGroup({
      layout: coarseSmoothPL.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: paramsBuf } },
        { binding: 1, resource: { buffer: errCoarseBuf[1] } },
        { binding: 2, resource: { buffer: errCoarseBuf[0] } },
        { binding: 3, resource: { buffer: rhsCoarseBuf } },
      ]});

    computeEKK();

    gpuReady = true;
    console.log("mol_fast ready — dispatch(" + WG_UPDATE + "," + NELEC + ",1), E_KK=" + E_KK.toFixed(4));
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
        cp = enc.beginComputePass(); cp.setPipeline(reducePL); cp.setBindGroup(0, reduceBG[next]);
        cp.dispatchWorkgroups(WG_REDUCE); cp.end();
        cp = enc.beginComputePass(); cp.setPipeline(finalizePL); cp.setBindGroup(0, finalizeBG);
        cp.dispatchWorkgroups(1); cp.end();
        cp = enc.beginComputePass(); cp.setPipeline(normalizePL); cp.setBindGroup(0, normalizeBG[next]);
        cp.dispatchWorkgroups(WG_NORM); cp.end();
      }
      // V-cycle correction every VCYCLE_INTERVAL steps, per electron
      if (VCYCLE_ENABLED && (s + 1) % VCYCLE_INTERVAL === 0) {
        for (let m = 0; m < NELEC; m++) {
          // Compute rho source for electron m
          cp = enc.beginComputePass(); cp.setPipeline(rhoSourcePL); cp.setBindGroup(0, rhoSourceBG[next][m]);
          cp.dispatchWorkgroups(Math.ceil(S3 / 256)); cp.end();
          // Residual (fine) = -2π·rho - ∇²P_m
          cp = enc.beginComputePass(); cp.setPipeline(residualPmPL); cp.setBindGroup(0, residualPmBG[next][m]);
          cp.dispatchWorkgroups(WG_UPDATE); cp.end();
          // Restrict residual → rhsCoarse
          cp = enc.beginComputePass(); cp.setPipeline(restrictPL); cp.setBindGroup(0, restrictBG);
          cp.dispatchWorkgroups(WG_COARSE); cp.end();
          // Zero both coarse error buffers (start from 0 correction)
          enc.clearBuffer(errCoarseBuf[0]);
          enc.clearBuffer(errCoarseBuf[1]);
          // Coarse Jacobi iterations, ping-pong
          for (let cs = 0; cs < COARSE_ITERS; cs++) {
            cp = enc.beginComputePass(); cp.setPipeline(coarseSmoothPL); cp.setBindGroup(0, coarseSmoothBG[cs % 2]);
            cp.dispatchWorkgroups(WG_COARSE); cp.end();
          }
          // After even # of iters, final lives in errCoarseBuf[0] (as bound in prolong BG).
          // If COARSE_ITERS is odd, the final would be in errCoarseBuf[1] — we keep it even above.
          // Prolongate correction onto P_m
          cp = enc.beginComputePass(); cp.setPipeline(prolongCorrectPmPL); cp.setBindGroup(0, prolongCorrectPmBG[next][m]);
          cp.dispatchWorkgroups(WG_UPDATE); cp.end();
        }
      }
      cur = next;
    }
    let cp = enc.beginComputePass(); cp.setPipeline(extractPL); cp.setBindGroup(0, extractBG[cur]);
    cp.dispatchWorkgroups(WG_EXTRACT, WG_EXTRACT); cp.end();
    enc.copyBufferToBuffer(sumsBuf, 0, sumsReadBuf, 0, SUMS_BYTES);
    enc.copyBufferToBuffer(sliceBuf, 0, sliceReadBuf, 0, SLICE_SIZE);
    device.queue.submit([enc.finish()]);
    await sumsReadBuf.mapAsync(GPUMapMode.READ);
    const sums = new Float32Array(sumsReadBuf.getMappedRange().slice(0));
    sumsReadBuf.unmap();
    E_T  = sums[NELEC];
    E_eK = sums[NELEC + 1];
    E_ee = sums[NELEC + 2];
    E = E_T + E_eK + E_ee + E_KK;
    // Dipole: μ = Σ_a Z_a·R_a + μ_elec (electron part already summed with minus sign)
    let dipNx = 0, dipNy = 0, dipNz = 0;
    for (let a = 0; a < N_ATOMS; a++) {
      dipNx += nuclei[a].Z * (nuclei[a].i - N2) * hv;
      dipNy += nuclei[a].Z * (nuclei[a].j - N2) * hv;
      dipNz += nuclei[a].Z * (nuclei[a].k - N2) * hv;
    }
    dipole[0] = dipNx + sums[NELEC + 3];
    dipole[1] = dipNy + sums[NELEC + 4];
    dipole[2] = dipNz + sums[NELEC + 5];
    dipoleMag = Math.hypot(dipole[0], dipole[1], dipole[2]);
    await sliceReadBuf.mapAsync(GPUMapMode.READ);
    sliceData = new Float32Array(sliceReadBuf.getMappedRange().slice(0));
    sliceReadBuf.unmap();
    tStep += n;
    lastMs = performance.now() - t0;
    if (!isFinite(E)) { gpuError = "numerical instability at step " + tStep; return; }
  }

  // ===== Nuclear dynamics =====

  function computeEKK() {
    E_KK = 0;
    for (let a = 0; a < N_ATOMS; a++) {
      for (let b = a + 1; b < N_ATOMS; b++) {
        const dx = (nuclei[a].i - nuclei[b].i) * hv;
        const dy = (nuclei[a].j - nuclei[b].j) * hv;
        const dz = (nuclei[a].k - nuclei[b].k) * hv;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (r > 0) E_KK += nuclei[a].Z * nuclei[b].Z / r;
      }
    }
  }

  async function computeForces() {
    if (forcesReadbackPending) return;
    forcesReadbackPending = true;
    const enc = device.createCommandEncoder();
    let cp = enc.beginComputePass();
    cp.setPipeline(forcePL);
    cp.setBindGroup(0, forceBG[cur]);
    cp.dispatchWorkgroups(N_ATOMS);
    cp.end();
    enc.copyBufferToBuffer(forceBuf, 0, forceReadBuf, 0, N_ATOMS * 3 * 4);
    device.queue.submit([enc.finish()]);
    await forceReadBuf.mapAsync(GPUMapMode.READ);
    const fd = new Float32Array(forceReadBuf.getMappedRange().slice(0));
    forceReadBuf.unmap();
    for (let a = 0; a < N_ATOMS; a++) {
      // Electronic force: F_elec = 2·Z·∇P (from shader)
      nucForce[a][0] = fd[a * 3];
      nucForce[a][1] = fd[a * 3 + 1];
      nucForce[a][2] = fd[a * 3 + 2];
      // Nuclear-nuclear force: F_nuc = Σ_b Z_a·Z_b·(R_a-R_b)/|R_a-R_b|³
      for (let b = 0; b < N_ATOMS; b++) {
        if (b === a) continue;
        const dx = (nuclei[a].i - nuclei[b].i) * hv;
        const dy = (nuclei[a].j - nuclei[b].j) * hv;
        const dz = (nuclei[a].k - nuclei[b].k) * hv;
        const r2 = dx * dx + dy * dy + dz * dz + 0.04 * h2v;
        const inv_r3 = 1 / (r2 * Math.sqrt(r2));
        const pref = nuclei[a].Z * nuclei[b].Z * inv_r3;
        nucForce[a][0] += pref * dx;
        nucForce[a][1] += pref * dy;
        nucForce[a][2] += pref * dz;
      }
    }
    forcesReadbackPending = false;
  }

  function stepNuclei() {
    // Velocity-Verlet-lite with damping. Acceleration = F / m; velocity += dt·a; pos += dt·v.
    const dt = DT_NUC;
    let moved = false;
    for (let a = 0; a < N_ATOMS; a++) {
      if (nuclei[a].Z <= 0) continue;
      const m = nucMass[a];
      for (let d = 0; d < 3; d++) {
        nucVel[a][d] = NUC_DAMPING * (nucVel[a][d] + dt * nucForce[a][d] / m);
      }
      // Convert velocity (au/time) → grid cells via 1/h
      const di = (nucVel[a][0] * dt) / hv;
      const dj = (nucVel[a][1] * dt) / hv;
      const dk = (nucVel[a][2] * dt) / hv;
      if (Math.abs(di) > 0.001 || Math.abs(dj) > 0.001 || Math.abs(dk) > 0.001) {
        nuclei[a].i += di;
        nuclei[a].j += dj;
        nuclei[a].k += dk;
        moved = true;
      }
    }
    if (moved) {
      // Rewrite paramsBuf atoms + dispatch recomputeK
      writeParams();
      computeEKK();
      const enc = device.createCommandEncoder();
      const cp = enc.beginComputePass();
      cp.setPipeline(recomputeKPL);
      cp.setBindGroup(0, recomputeKBG);
      cp.dispatchWorkgroups(Math.ceil(S3 / 256));
      cp.end();
      device.queue.submit([enc.finish()]);
    }
  }

  // ===== p5.js integration =====
  const ELEC_COLORS = [[255,50,50],[50,255,50],[50,100,255],[255,200,0],[200,50,255],[0,255,200],[255,100,200],[150,255,150]];

  // 3D view state (WEBGL buffer)
  let view3D = null;
  let rotX = -0.4, rotY = 0.5;
  let dragStart = null;

  window.setup = function () {
    createCanvas(620, 620);
    textSize(11);
    view3D = createGraphics(200, 400, WEBGL);
    view3D.textSize(10);
    initGPU();
  };

  window.draw = function () {
    background(0);
    if (gpuError) { fill(255,80,80); text("ERR: " + gpuError, 5, 20); return; }
    if (!gpuReady) { fill(255); text("Initializing WebGPU...", 5, 20); return; }
    if (!computing && tStep < MAX_STEPS) {
      computing = true;
      doSteps(STEPS_PER_FRAME).then(() => {
        // Always compute forces so arrows display; only move atoms when dynamics is enabled
        computeForces().then(() => {
          if (dynamicsEnabled) stepNuclei();
          computing = false;
        });
      });
    }
    if (sliceData) {
      noStroke();
      // 2D density image (z=N2 slice)
      for (let i = 1; i < NN; i++) {
        for (let j = 1; j < NN; j++) {
          const px = PX * i, py = PX * j;
          let r = 0, g = 0, b = 0;
          for (let m = 0; m < NELEC; m++) {
            const v = 500 * sliceData[m * S * S + i * S + j];
            const col = ELEC_COLORS[m % ELEC_COLORS.length];
            r += v * col[0] / 255;
            g += v * col[1] / 255;
            b += v * col[2] / 255;
          }
          fill(Math.min(255, r), Math.min(255, g), Math.min(255, b));
          rect(px, py, PX + 1, PX + 1);
        }
      }
      // Nuclei
      fill(255); stroke(255); strokeWeight(1);
      for (let a = 0; a < N_ATOMS; a++) circle(nuclei[a].i * PX, nuclei[a].j * PX, 6);
      // Force arrows (red) — always shown; dimmer when dynamics is off
      stroke(dynamicsEnabled ? 255 : 180, dynamicsEnabled ? 0 : 80, dynamicsEnabled ? 0 : 80);
      strokeWeight(1.5);
      const scale = 200;
      for (let a = 0; a < N_ATOMS; a++) {
        if (nuclei[a].Z <= 0) continue;
        const x0 = nuclei[a].i * PX, y0 = nuclei[a].j * PX;
        const fx = nucForce[a][0], fy = nucForce[a][1];
        if (Math.hypot(fx, fy) < 1e-8) continue;
        line(x0, y0, x0 + fx * scale, y0 + fy * scale);
      }
      noStroke();
      // 1D line plots along i-axis (k=N2, j=N2): w (black dot), u (electron color)
      // Located below the 2D image (y=410..530 area). Same baseline, same ×100 scale.
      fill(30); noStroke(); rect(0, 405, 420, 130);     // plot background
      fill(80); rect(0, 525, 420, 1);                   // zero line
      const baseY = 525;
      const lineOffsetU = NELEC * S * S;
      const lineOffsetW = 2 * NELEC * S * S;
      // Potentials packed in unused tail of section 1:
      //   [N*S*S + N*S       ..  N*S*S + (N+1)*S - 1]  = 2·P_total
      //   [N*S*S + (N+1+m)*S ..  .. + S - 1]           = V_pauli per electron m
      const lineOffsetPtot = NELEC * S * S + NELEC * S;
      const lineOffsetVp   = NELEC * S * S + (NELEC + 1) * S;
      // Line indexed by i (x-axis)
      for (let ii = 1; ii < NN; ii++) {
        const x = PX * ii;
        for (let m = 0; m < NELEC; m++) {
          const uVal = sliceData[lineOffsetU + m * S + ii];
          const wVal = sliceData[lineOffsetW + m * S + ii];
          const col = ELEC_COLORS[m % ELEC_COLORS.length];
          fill(180);                    ellipse(x, baseY - 100 * wVal, 3);        // w (gray)
          fill(col[0], col[1], col[2]); ellipse(x, baseY - 100 * uVal * wVal, 3); // u·w (color)
          // V_pauli per electron — cyan dots, clipped
          const vp = sliceData[lineOffsetVp + m * S + ii];
          if (vp > 0) {
            fill(0, 220, 255);
            ellipse(x, baseY - Math.min(120, 20 * vp), 2);
          }
        }
        // 2·P_total (orange), logish compression so it fits: 10 * sqrt(P)
        const pt = sliceData[lineOffsetPtot + ii];
        if (pt > 0 && isFinite(pt)) {
          fill(255, 150, 50);
          ellipse(x, baseY - Math.min(120, 10 * Math.sqrt(pt)), 2);
        }
      }
      fill(200); textSize(10);
      text("u·w (colors), w (gray), 2·P_tot (orange, √-compressed), V_pauli (cyan)", 8, 418);
      textSize(11);
    }
    fill(255);
    text("mol_fast " + NN + "³ / " + screenAu + " au | " + NELEC + "e step " + tStep, 5, 550);
    text("E=" + E.toFixed(4) + " | T=" + E_T.toFixed(3) + " VeK=" + E_eK.toFixed(3) + " Vee=" + E_ee.toFixed(3) + " VKK=" + E_KK.toFixed(3), 5, 565);
    if (lastMs > 0) text((lastMs / STEPS_PER_FRAME).toFixed(2) + " ms/step", 330, 550);
    fill(180, 220, 255);
    var distLines = [];
    for (var a = 0; a < N_ATOMS; a++) {
      for (var b = a + 1; b < N_ATOMS; b++) {
        var dx = (nuclei[a].i - nuclei[b].i) * hv;
        var dy = (nuclei[a].j - nuclei[b].j) * hv;
        var dz = (nuclei[a].k - nuclei[b].k) * hv;
        var d = Math.sqrt(dx*dx + dy*dy + dz*dz);
        distLines.push(a + "-" + b + ":" + d.toFixed(2));
      }
    }
    text(distLines.join("  "), 5, 400);
    // Dynamics status — on its own line (y=578) below the energy line (y=565)
    fill(dynamicsEnabled ? [0, 255, 0] : [150, 150, 150]);
    text("dyn=" + (dynamicsEnabled ? "ON" : "off") + "  [press D to toggle]", 5, 578);
    // Force magnitudes per atom (atomic units of force ≈ Hartree/Bohr)
    var Fparts = [];
    var Fmax = 0;
    for (var a = 0; a < N_ATOMS; a++) {
      var Fm = Math.hypot(nucForce[a][0], nucForce[a][1], nucForce[a][2]);
      if (Fm > Fmax) Fmax = Fm;
      Fparts.push(a + ":" + Fm.toFixed(3));
    }
    fill(255, 150, 150);
    text("|F| " + Fparts.join("  ") + "  max=" + Fmax.toFixed(4) + " au", 5, 593);
    // Dipole moment display (atomic units and Debye)
    fill(150, 220, 255);
    var dipD = dipoleMag * 2.5417;  // au → Debye
    text("dipole=" + dipoleMag.toFixed(4) + " au (" + dipD.toFixed(3) + " D)  dir=("
      + dipole[0].toFixed(3) + ", " + dipole[1].toFixed(3) + ", " + dipole[2].toFixed(3) + ")", 5, 608);
    // 3D view (right side)
    draw3D();
    image(view3D, 420, 0);
  };

  function draw3D() {
    view3D.background(10);
    view3D.push();
    view3D.rotateX(rotX);
    view3D.rotateY(rotY);
    // Scale grid units so the molecule fits — 60 px per au roughly
    const pxPerAu = 60;
    // Draw nuclei
    view3D.noStroke();
    for (let a = 0; a < N_ATOMS; a++) {
      const x = (nuclei[a].i - N2) * hv * pxPerAu;
      const y = (nuclei[a].j - N2) * hv * pxPerAu;
      const z = (nuclei[a].k - N2) * hv * pxPerAu;
      const col = ELEC_COLORS[a % ELEC_COLORS.length];
      view3D.fill(col[0], col[1], col[2]);
      view3D.push();
      view3D.translate(x, y, z);
      // Sphere radius scales with Z
      const rad = 6 + 3 * Math.sqrt(Math.max(1, nuclei[a].Z));
      view3D.sphere(rad);
      view3D.pop();
    }
    // Bond lines between all atom pairs
    view3D.stroke(180, 180, 180, 150); view3D.strokeWeight(1.5);
    view3D.noFill();
    for (let a = 0; a < N_ATOMS; a++) {
      for (let b = a + 1; b < N_ATOMS; b++) {
        const ax = (nuclei[a].i - N2) * hv * pxPerAu;
        const ay = (nuclei[a].j - N2) * hv * pxPerAu;
        const az = (nuclei[a].k - N2) * hv * pxPerAu;
        const bx = (nuclei[b].i - N2) * hv * pxPerAu;
        const by = (nuclei[b].j - N2) * hv * pxPerAu;
        const bz = (nuclei[b].k - N2) * hv * pxPerAu;
        view3D.line(ax, ay, az, bx, by, bz);
      }
    }
    // Force arrows — always; dimmer when dynamics is off
    view3D.stroke(255, dynamicsEnabled ? 80 : 160, dynamicsEnabled ? 80 : 160);
    view3D.strokeWeight(2);
    const fScale = 150;
    for (let a = 0; a < N_ATOMS; a++) {
      const x = (nuclei[a].i - N2) * hv * pxPerAu;
      const y = (nuclei[a].j - N2) * hv * pxPerAu;
      const z = (nuclei[a].k - N2) * hv * pxPerAu;
      view3D.line(x, y, z,
        x + nucForce[a][0] * fScale, y + nucForce[a][1] * fScale, z + nucForce[a][2] * fScale);
    }
    view3D.pop();
  }

  window.mousePressed = function() {
    if (mouseX >= 420 && mouseX < 620 && mouseY >= 0 && mouseY < 400) {
      dragStart = [mouseX, mouseY, rotX, rotY];
    }
  };
  window.mouseDragged = function() {
    if (dragStart) {
      rotY = dragStart[3] + (mouseX - dragStart[0]) * 0.01;
      rotX = dragStart[2] + (mouseY - dragStart[1]) * 0.01;
    }
  };
  window.mouseReleased = function() { dragStart = null; };

  window.keyPressed = function() {
    if (key === 'd' || key === 'D') {
      dynamicsEnabled = !dynamicsEnabled;
      console.log("dynamics:", dynamicsEnabled ? "ON" : "off");
    }
  };
})();
